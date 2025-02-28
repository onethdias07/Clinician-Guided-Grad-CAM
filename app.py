import os
import base64
from io import BytesIO
import logging
import random
import uuid
import sys
import subprocess
import threading
import time

import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

from flask import Flask, request, render_template, jsonify
import torch
import torchvision.transforms as T
import torchxrayvision as xrv

from utils.grad_cam import GradCAM, upsample_cam, overlay_cam_on_image
from utils.feedback_utils import do_finetuning_step

import torch.nn.functional as F
import torchxrayvision.utils as xrv_utils

app = Flask(__name__)
app.config['SECRET_KEY'] = 'QimQxIXuNwyGeq1zF935xt2w76Ks0WT0'
logging.basicConfig(level=logging.DEBUG)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

model = None

current_label_idx = None
current_unique_id = None
uploaded_image_bytes = None
uploaded_img_np = None

def patched_features2(self, x):
    x = xrv_utils.fix_resolution(x, 224, self)
    xrv_utils.warn_normalization(x)
    features = self.features(x)
    out = F.relu(features, inplace=False)
    out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
    return out

def disable_inplace_relu(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.ReLU):
            mod.inplace = False

def should_invert(img_np):
    mean_val = img_np.mean()
    return (mean_val > 0.7)

def load_model():
    global model
    app.logger.debug("Loading TorchXrayvision DenseNet model...")

    try:
        xrv.models.DenseNet.features2 = patched_features2
        model_local = xrv.models.DenseNet(weights="densenet121-res224-all")
        model_local.eval()
        disable_inplace_relu(model_local)

        for param in model_local.parameters():
            param.requires_grad = True

        if torch.cuda.is_available():
            model_local.cuda()

        model = model_local
        app.logger.debug("Model loaded. Pathologies: %s", model.pathologies)
    except Exception as e:
        app.logger.error("Error loading model: %s", e, exc_info=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_image", methods=["POST"])
def upload_image():
    global current_label_idx, current_unique_id
    global uploaded_image_bytes, uploaded_img_np

    uploaded_file = request.files.get("file")
    if not uploaded_file:
        return jsonify({"error": "No file provided"}), 400

    uploaded_image_bytes = uploaded_file.read()
    current_unique_id = str(uuid.uuid4())

    try:
        pil_img = Image.open(BytesIO(uploaded_image_bytes)).convert("L")
        img_np = np.array(pil_img, dtype=np.float32)
        img_np = xrv.datasets.normalize(img_np, 255)
        if should_invert(img_np):
            img_np = 1.0 - img_np

        img_np = img_np[None, ...]
    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image file"}), 400

    transform = T.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img_np = transform(img_np)
    uploaded_img_np = img_np.copy()

    image_tensor = torch.from_numpy(img_np).unsqueeze(0)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        outputs = model(image_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
    logits = outputs[0].cpu().numpy()

    best_idx = int(np.argmax(logits))
    best_logit = float(logits[best_idx])
    current_label_idx = best_idx
    label_name = model.pathologies[best_idx]

    prob = 1.0 / (1.0 + np.exp(-best_logit))
    app.logger.info(f"upload_image => Pathology {label_name}, logit={best_logit:.4f}, prob={prob:.4f}")

    cam_np = GradCAM.generate_cam_smooth(
        model=model,
        input_tensor=image_tensor,
        target_label=best_idx,
        num_runs=5,
        threshold=0.05
    )

    cam_224 = upsample_cam(cam_np, (224,224))
    xray_np = img_np.squeeze()
    xray_np_color = np.stack([xray_np]*3, axis=-1)
    overlay = overlay_cam_on_image(xray_np_color, cam_224, alpha=0.5)
    overlay_b64 = array_to_b64(overlay)

    return jsonify({
        "pred_class": label_name,
        "pred_conf": prob,
        "gradcam": overlay_b64
    })

@app.route("/submit_mask", methods=["POST"])
def submit_mask():
    global current_label_idx, current_unique_id
    global uploaded_image_bytes, uploaded_img_np

    data = request.get_json()
    if not data or "mask" not in data:
        return jsonify({"error": "No mask data found"}), 400

    if uploaded_image_bytes is None or uploaded_img_np is None:
        return jsonify({"error": "No uploaded image found in memory."}), 400

    try:
        expert_mask = b64_to_array(data["mask"], target_size=(224,224))
        user_present = data.get("pathology_present", True)
        real_label = 1 if user_present else 0

        os.makedirs("data/images", exist_ok=True)
        xray_filename = f"{current_unique_id}_xray.png"
        xray_path = os.path.join("data/images", xray_filename)
        with open(xray_path, "wb") as f:
            f.write(uploaded_image_bytes)

        os.makedirs("data/masks", exist_ok=True)
        mask_filename = f"{current_unique_id}_mask.png"
        mask_path = os.path.join("data/masks", mask_filename)
        mask_pil = Image.fromarray((expert_mask*255).astype(np.uint8))
        mask_pil.save(mask_path)

        os.makedirs("data", exist_ok=True)
        annotations_csv = os.path.join("data", "annotations.csv")
        row_to_append = f"{xray_path},{mask_path},{current_label_idx},{real_label}\n"

        if not os.path.exists(annotations_csv):
            with open(annotations_csv, "w") as f:
                f.write("xray_path,mask_path,pathology_idx,real_label\n")

        with open(annotations_csv, "a") as f:
            f.write(row_to_append)
        app.logger.info(f"submit_mask => appended row: {row_to_append.strip()}")

        image_tensor = torch.from_numpy(uploaded_img_np).unsqueeze(0)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        with torch.no_grad():
            out_before = model(image_tensor)
            if isinstance(out_before, tuple):
                out_before = out_before[0]
            prob_before = float(torch.sigmoid(out_before[0, current_label_idx]))

        steps = 10
        lr = 1e-5
        lambda_feedback = 5.0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for step in range(steps):
            loss_val = do_finetuning_step(
                model=model,
                input_tensor=image_tensor,
                real_label=real_label,
                user_mask=expert_mask,
                target_label=current_label_idx,
                optimizer=optimizer,
                lambda_feedback=lambda_feedback
            )
            app.logger.debug("Fine-tune step %d/%d, loss=%.4f", step+1, steps, loss_val)

        with torch.no_grad():
            out_after = model(image_tensor)
            if isinstance(out_after, tuple):
                out_after = out_after[0]
            prob_after = float(torch.sigmoid(out_after[0, current_label_idx]))

        updated_cam = GradCAM.generate_cam_smooth(
            model=model,
            input_tensor=image_tensor,
            target_label=current_label_idx,
            num_runs=5,
            threshold=0.05
        )
        cam_224 = upsample_cam(updated_cam, (224,224))
        xray_np = uploaded_img_np.squeeze()
        xray_np_color = np.stack([xray_np]*3, axis=-1)
        overlay = overlay_cam_on_image(xray_np_color, cam_224, alpha=0.5)
        updated_b64 = array_to_b64(overlay)

        best_logit = float(out_after[0, current_label_idx].cpu().numpy())
        final_prob = 1.0 / (1.0 + np.exp(-best_logit))
        label_name = model.pathologies[current_label_idx]

        return jsonify({
            "updated_gradcam": updated_b64,
            "pred_class": label_name,
            "pred_conf": final_prob
        })

    except Exception as e:
        app.logger.error("Error in /submit_mask: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500

def array_to_b64(np_img):
    buffer = BytesIO()
    if np_img.dtype != np.uint8:
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    if np_img.ndim == 2:
        img = Image.fromarray(np_img, mode="L")
    else:
        img = Image.fromarray(np_img, mode="RGB")
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def b64_to_array(b64_str, target_size=None):
    decoded = base64.b64decode(b64_str)
    with BytesIO(decoded) as buf:
        pil_img = Image.open(buf).convert("L")
        if target_size is not None:
            pil_img = pil_img.resize(target_size)
        arr = np.array(pil_img, dtype=np.float32)/255.
    return arr

training_status = {
    "running": False,
    "epoch": 0,
    "total_epochs": 0,
    "message": "Idle"
}
status_lock = threading.Lock()
training_process = None

def parse_epoch_line(line_str):
    if "Epoch " not in line_str or "/" not in line_str:
        return
    try:
        after_epoch = line_str.split("Epoch ")[1]
        part = after_epoch.split()[0]
        current_str, total_str = part.split("/")
        current_epoch = int(current_str)
        tot_epochs = int(total_str)
        with status_lock:
            training_status["epoch"] = current_epoch
            training_status["total_epochs"] = tot_epochs
    except:
        pass

def run_offline_train_subprocess():
    global training_process
    cmd = [sys.executable, "-u", "offline_train.py", "--config", "config.yaml"]
    with status_lock:
        training_status["running"] = True
        training_status["epoch"] = 0
        training_status["total_epochs"] = 0
        training_status["message"] = "Training started (subprocess)"

    training_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )

    for line in training_process.stdout:
        line_str = line.strip()
        with status_lock:
            training_status["message"] = line_str
        parse_epoch_line(line_str)

    training_process.wait()
    with status_lock:
        training_status["running"] = False
        training_status["message"] = "Training finished (subprocess)"
    training_process = None

@app.route("/start_training", methods=["POST"])
def start_training():
    with status_lock:
        if training_status["running"]:
            return jsonify({"message": "Training is already in progress."}), 200
        training_status["running"] = True
        training_status["message"] = "Spawning training process..."

    t = threading.Thread(target=run_offline_train_subprocess, daemon=True)
    t.start()
    return jsonify({"message": "Offline training started."}), 200

@app.route("/training_status", methods=["GET"])
def get_training_status():
    with status_lock:
        status_copy = dict(training_status)
    return jsonify(status_copy)

if __name__ == "__main__":
    load_model()
    finetuned_path = "checkpoints/model_finetuned.pth"
    if os.path.exists(finetuned_path):
        model.load_state_dict(torch.load(
            finetuned_path,
            map_location="cuda" if torch.cuda.is_available() else "cpu"
        ))
        model.eval()
        app.logger.info(f"Loaded fine-tuned model from {finetuned_path}")

    app.run(debug=True, use_reloader=False, port=8000)
