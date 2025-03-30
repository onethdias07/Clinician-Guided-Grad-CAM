import os
import csv
import base64
import time
import uuid
import sys
import json
import logging
import subprocess
import threading
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import torch
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from werkzeug.security import generate_password_hash, check_password_hash
import jwt

from attention_model import SimpleAttentionCNN, SpatialAttention
from grad_cam.grad_cam import HeatMapper, show_grad_cam, find_good_layer

# Track status of model refinement process
finetuning_process = None
finetuning_status = {
    "running": False,
    "message": "No refinement process running",
    "current_epoch": 0,
    "total_epochs": 10,
    "start_time": None,
    "feedback_count": 0
}

# Cache models to avoid reloading them for each request
loaded_models = {
    "original": None,
    "finetuned": None
}

# In-memory user store (would be replaced by database in production)
users = {}
JWT_SECRET = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
JWT_EXPIRATION = 24 * 60 * 60  # 24 hours in seconds

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({
                'message': 'Authentication token is missing',
                'authenticated': False
            }), 401
        
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            current_user = users.get(data['username'])
            if not current_user:
                return jsonify({
                    'message': 'User not found',
                    'authenticated': False
                }), 401
        except jwt.ExpiredSignatureError:
            return jsonify({
                'message': 'Token has expired',
                'authenticated': False
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                'message': 'Invalid token',
                'authenticated': False
            }), 401
            
        return f(current_user, *args, **kwargs)
    
    return decorated

def get_probability(img, model_type="current"):
    # Simplified input handling - convert all inputs to tensor in a consistent way
    if isinstance(img, Image.Image):
        # Convert PIL Image to tensor
        img_np = np.array(img.convert("L"))
        img_np = cv2.resize(img_np, (256, 256))
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    elif isinstance(img, np.ndarray):
        # Convert numpy array to tensor
        img_np = cv2.resize(img, (256, 256)) if img.shape[0] != 256 else img
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    elif isinstance(img, torch.Tensor):
        # Already a tensor
        img_tensor = img
    else:
        raise TypeError("Image must be PIL Image, numpy array, or torch Tensor")
    
    # Use dictionary lookup instead of conditionals to select model
    model_dict = {
        "original": loaded_models["original"],
        "finetuned": loaded_models["finetuned"],
        "current": attention_model
    }
    
    model = model_dict.get(model_type, attention_model)
    if model is None:
        model = attention_model
    
    with torch.no_grad():
        output, _ = model(img_tensor)
        pred_prob = output.item()
        pred_prob_percent = round(pred_prob * 100, 2)
        
    return pred_prob_percent

def load_attention_model(model_path=None):
    # Simplify path resolution
    if not model_path or not os.path.exists(model_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'model', 'tb_chest_xray_attention_best.pt')

    model = SimpleAttentionCNN()
    
    try:
        # Simplify model loading with a fallback mechanism
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise

def load_both_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if loaded_models["original"] is None:
        original_path = os.path.join(base_dir, 'model', 'tb_chest_xray_attention_best.pt')
        try:
            loaded_models["original"] = load_attention_model(original_path)
            logging.info(f"Loaded original model from {original_path}")
        except Exception as e:
            logging.error(f"Error loading original model: {e}")
    
    if loaded_models["finetuned"] is None:
        latest_model = find_latest_refined_model()
        if latest_model:
            finetuned_path = os.path.join(base_dir, 'finetuning', latest_model)
            try:
                loaded_models["finetuned"] = load_attention_model(finetuned_path)
                logging.info(f"Loaded finetuned model from {finetuned_path}")
            except Exception as e:
                logging.error(f"Error loading finetuned model: {e}")
    
    return loaded_models["original"] is not None, loaded_models["finetuned"] is not None

def create_overlay(map_2d, original_gray, size=256, colormap=cv2.COLORMAP_INFERNO, alpha=0.7):
    map_2d = np.asarray(map_2d, dtype=np.float32)
    map_resized = cv2.resize(map_2d, (size, size))
    
    if (map_resized.max() - map_resized.min()) > 1e-5:
        map_resized = (map_resized - map_resized.min()) / (map_resized.max() - map_resized.min())
        threshold = 0.4
        map_resized[map_resized < threshold] = 0
        
        if map_resized.max() > 0:
            map_resized = map_resized / map_resized.max()
        
        map_resized = np.power(map_resized, 1.5)
    else:
        h, w = map_resized.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h/2, w/2
        map_resized = 1 - (((y - center_y)/(h/2))**2 + ((x - center_x)/(w/2))**2) / 2
        map_resized = np.clip(map_resized, 0, 1) * 0.7

    heatmap = cv2.applyColorMap((map_resized * 255).astype(np.uint8), colormap)

    if len(original_gray.shape) == 2:
        original_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = original_gray
    original_bgr = cv2.resize(original_bgr, (size, size))

    overlay = cv2.addWeighted(original_bgr, 1-alpha, heatmap, alpha, 0)
    return overlay

# Simplified image processing function with common functionality extracted
def process_base_image(file):
    # Process image file to PIL and numpy formats
    pil_img = Image.open(file).convert("RGB")
    pil_img_gray = pil_img.convert("L")

    # Create base64 of original image
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    original_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Prepare tensor for model
    img_np = np.array(pil_img_gray)
    img_np = cv2.resize(img_np, (256, 256))
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    
    return {
        'pil_img': pil_img,
        'pil_img_gray': pil_img_gray,
        'img_np': img_np,
        'img_tensor': img_tensor,
        'original_base64': original_base64
    }

def process_image(file):
    # Use the base image processor
    base_result = process_base_image(file)
    img_np = base_result['img_np']
    img_tensor = base_result['img_tensor']
    
    # Get tuberculosis probability
    pred_prob_percent = get_probability(img_tensor)

    # Generate attention map
    with torch.no_grad():
        _, attn_map = attention_model(img_tensor)

    attn_map_np = attn_map[0].squeeze().cpu().numpy()
    
    # Create attention overlay
    attn_overlay_bgr = create_overlay(attn_map_np, img_np, size=256)
    _, buffer = cv2.imencode('.jpg', attn_overlay_bgr)
    attn_overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    # Generate Grad-CAM visualization
    try:
        try:
            focus_layer = find_good_layer(attention_model, img_tensor)
        except Exception:
            # Fallback to a default layer based on probability
            if pred_prob_percent > 50:
                focus_layer = attention_model.backbone[2]
            else:
                focus_layer = attention_model.backbone[4] if len(attention_model.backbone) >= 4 else attention_model.backbone[0]
                    
        _, grad_cam_overlay_bgr = show_grad_cam(
            img_np, 
            attention_model, 
            target_layer=focus_layer,
            use_relu=True,
            smooth_factor=0.3,
            alpha=0.65
        )
    except Exception:
        # Use attention overlay as fallback
        grad_cam_overlay_bgr = attn_overlay_bgr
    
    # Convert Grad-CAM to base64
    _, buffer = cv2.imencode('.jpg', grad_cam_overlay_bgr)
    grad_cam_overlay_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return {
        'original_image': base_result['original_base64'],
        'grad_cam_image': grad_cam_overlay_base64,
        'attention_image': attn_overlay_base64,
        'tb_probability': pred_prob_percent
    }

def process_image_with_comparison(file):
    pil_img = Image.open(file).convert("RGB")
    pil_img_gray = pil_img.convert("L")

    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    original_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    img_np = np.array(pil_img_gray)
    img_np = cv2.resize(img_np, (256, 256))
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor / 255.0

    original_loaded, finetuned_loaded = load_both_models()
    
    results = {
        'original_image': original_base64,
        'has_comparison': original_loaded and finetuned_loaded
    }
    
    if original_loaded:
        try:
            original_prob = get_probability(img_tensor, "original")
            
            with torch.no_grad():
                original_output, original_attn = loaded_models["original"](img_tensor)
            
            original_attn_np = original_attn[0].squeeze().cpu().numpy()
            original_overlay = create_overlay(original_attn_np, img_np, size=256)
            _, buffer = cv2.imencode('.jpg', original_overlay)
            original_attn_base64 = base64.b64encode(buffer).decode("utf-8")
            
            try:
                focus_layer = find_good_layer(loaded_models["original"], img_tensor)
                _, original_gradcam = show_grad_cam(
                    img_np, 
                    loaded_models["original"], 
                    target_layer=focus_layer,
                    use_relu=True,
                    smooth_factor=0.3,
                    alpha=0.65
                )
            except Exception:
                original_gradcam = original_overlay
            
            _, buffer = cv2.imencode('.jpg', original_gradcam)
            original_gradcam_base64 = base64.b64encode(buffer).decode("utf-8")
            
            results.update({
                'original_probability': original_prob,
                'original_attention': original_attn_base64,
                'original_gradcam': original_gradcam_base64
            })
        except Exception as e:
            logging.error(f"Error processing with original model: {e}")
            results['original_error'] = str(e)
    
    if finetuned_loaded:
        try:
            finetuned_prob = get_probability(img_tensor, "finetuned")
            
            with torch.no_grad():
                finetuned_output, finetuned_attn = loaded_models["finetuned"](img_tensor)
            
            finetuned_attn_np = finetuned_attn[0].squeeze().cpu().numpy()
            finetuned_overlay = create_overlay(finetuned_attn_np, img_np, size=256)
            _, buffer = cv2.imencode('.jpg', finetuned_overlay)
            finetuned_attn_base64 = base64.b64encode(buffer).decode("utf-8")
            
            try:
                focus_layer = find_good_layer(loaded_models["finetuned"], img_tensor)
                _, finetuned_gradcam = show_grad_cam(
                    img_np, 
                    loaded_models["finetuned"], 
                    target_layer=focus_layer,
                    use_relu=True,
                    smooth_factor=0.3,
                    alpha=0.65
                )
            except Exception:
                finetuned_gradcam = finetuned_overlay
            
            _, buffer = cv2.imencode('.jpg', finetuned_gradcam)
            finetuned_gradcam_base64 = base64.b64encode(buffer).decode("utf-8")
            
            results.update({
                'finetuned_probability': finetuned_prob,
                'finetuned_attention': finetuned_attn_base64,
                'finetuned_gradcam': finetuned_gradcam_base64
            })
        except Exception as e:
            logging.error(f"Error processing with finetuned model: {e}")
            results['finetuned_error'] = str(e)
    
    normal_results = process_image(file)
    results.update({
        'tb_probability': normal_results['tb_probability'],
        'grad_cam_image': normal_results['grad_cam_image'],
        'attention_image': normal_results['attention_image']
    })
    
    return results

def save_feedback(image_data, mask_data, user_label):
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    unique_id = f"{timestamp_str}_{uuid.uuid4().hex[:6]}"

    image_filename = f"{unique_id}_image.jpg"
    mask_filename = f"{unique_id}_mask.png"

    image_path = os.path.join(images_dir, image_filename)
    mask_path = os.path.join(masks_dir, mask_filename)

    image_bytes = base64.b64decode(image_data)
    with open(image_path, "wb") as f_img:
        f_img.write(image_bytes)

    mask_bytes = base64.b64decode(mask_data)
    with open(mask_path, "wb") as f_mask:
        f_mask.write(mask_bytes)

    with open(log_csv_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([image_filename, mask_filename, user_label, timestamp_str])

    return {
        "status": "ok",
        "image_saved": image_filename,
        "mask_saved": mask_filename,
        "label": user_label,
        "timestamp": timestamp_str
    }

def monitor_finetuning_output(process):
    global finetuning_status
    
    for line in process.stdout:
        logging.info(f"Finetuning output: {line.strip()}")
        
        if "[Epoch" in line:
            try:
                parts = line.split('[Epoch ')[1].split(']')[0].split('/')
                current_epoch = int(parts[0])
                total_epochs = int(parts[1])
                
                finetuning_status["current_epoch"] = current_epoch
                finetuning_status["total_epochs"] = total_epochs
                finetuning_status["message"] = f"Refining model (Epoch {current_epoch}/{total_epochs})"
                
            except Exception as e:
                logging.error(f"Error parsing epoch info: {e}")

    if process.poll() is not None:
        exit_code = process.returncode
        if exit_code == 0:
            finetuning_status["message"] = "Model refinement completed successfully"
        else:
            finetuning_status["message"] = f"Model refinement failed with exit code {exit_code}"
        
        finetuning_status["running"] = False
        finetuning_status["current_epoch"] = finetuning_status["total_epochs"]

def find_latest_refined_model():
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetuning')
        if not os.path.exists(model_dir):
            return None
            
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and 'refined' in f]
        if not model_files:
            return None
            
        newest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
        return newest_model
    except Exception as e:
        logging.error(f"Error finding latest model: {e}")
        return None

def find_all_refined_models():
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetuning')
        if not os.path.exists(model_dir):
            return []
            
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and 'refined' in f]
        if not model_files:
            return []
            
        model_files.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
        return model_files
    except Exception as e:
        logging.error(f"Error finding refined models: {e}")
        return []

app = Flask(__name__)
CORS(app)

base_dir = os.path.dirname(os.path.abspath(__file__))
feedback_dir = os.path.join(base_dir, 'feedback')
images_dir = os.path.join(feedback_dir, 'images')
masks_dir = os.path.join(feedback_dir, 'masks')
log_csv_path = os.path.join(feedback_dir, 'feedback_log.csv')

os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

if not os.path.isfile(log_csv_path):
    with open(log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "mask_filename", "label", "timestamp"])

model_path = os.path.join(base_dir, 'model', 'tb_chest_xray_attention_best.pt')
attention_model = load_attention_model(model_path)

def count_feedback_items():
    if not os.path.exists(log_csv_path):
        return 0
    
    try:
        with open(log_csv_path, 'r') as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception as e:
        print(f"Error counting feedback items: {e}")
        return 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(base_dir, "app_debug.log"))
    ]
)
logger = logging.getLogger("App")

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Username and password are required'}), 400
        
    username = data.get('username')
    
    if username in users:
        return jsonify({'message': 'User already exists'}), 409
    
    hashed_password = generate_password_hash(data.get('password'))
    users[username] = {
        'username': username,
        'password': hashed_password,
        'role': data.get('role', 'user')
    }
    
    return jsonify({
        'message': 'User registered successfully',
        'username': username
    }), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Username and password are required'}), 400
        
    username = data.get('username')
    
    if username not in users:
        return jsonify({'message': 'Invalid username or password'}), 401
    
    user = users[username]
    if not check_password_hash(user['password'], data.get('password')):
        return jsonify({'message': 'Invalid username or password'}), 401
    
    token_expiration = datetime.utcnow() + timedelta(seconds=JWT_EXPIRATION)
    payload = {
        'username': username,
        'role': user['role'],
        'exp': token_expiration
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    
    return jsonify({
        'message': 'Login successful',
        'token': token,
        'username': username,
        'role': user['role'],
        'expires': JWT_EXPIRATION
    })

@app.route('/api/auth/verify', methods=['GET'])
@token_required
def verify_token(current_user):

    return jsonify({
        'authenticated': True,
        'username': current_user['username'],
        'role': current_user['role']
    })

@app.route('/')
def home():
    feedback_count = count_feedback_items()
    return render_template('index.html', 
                          feedback_count=feedback_count,
                          current_model="tb_chest_xray_attention_best.pt")

@app.route('/api/predict', methods=['POST'])
@token_required
def api_predict(current_user):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        compare = request.args.get('compare', 'false').lower() == 'true'
        
        if compare:
            result = process_image_with_comparison(file)
        else:
            result = process_image(file)
            
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/api/feedback', methods=['POST'])
@token_required
def api_feedback(current_user):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload"}), 400

    base64_image = data.get("image", None)
    base64_mask = data.get("mask", None)
    user_label = data.get("label", "")

    if not base64_image or not base64_mask:
        return jsonify({"error": "Missing 'image' or 'mask' field in JSON"}), 400

    try:
        result = save_feedback(base64_image, base64_mask, user_label)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error saving feedback: {str(e)}"}), 500

@app.route('/api/feedback_count', methods=['GET'])
@token_required
def api_feedback_count(current_user):
    count = count_feedback_items()
    return jsonify({"count": count})

@app.route('/api/finetuning_status', methods=['GET'])
@token_required
def check_finetuning_status(current_user):
    global finetuning_status
    finetuning_status["feedback_count"] = count_feedback_items()
    
    if finetuning_process is not None and finetuning_status["running"]:
        if finetuning_process.poll() is not None:
            finetuning_status["running"] = False
            finetuning_status["message"] = "Model refinement completed"
            finetuning_status["current_epoch"] = finetuning_status["total_epochs"]
    
    return jsonify(finetuning_status)

@app.route('/api/run_finetuning', methods=['POST'])
@token_required
def run_finetuning(current_user):
    if current_user['role'] != 'admin':
        return jsonify({
            "success": False,
            "message": "Insufficient permissions. Only administrators can run model refinement."
        }), 403
    
    global finetuning_process, finetuning_status
    
    logging.info("Received request to /api/run_finetuning endpoint")
    
    if finetuning_status["running"]:
        return jsonify({
            "success": False,
            "message": "A refinement process is already running"
        })
    
    feedback_count = count_feedback_items()
    if feedback_count == 0:
        return jsonify({
            "success": False,
            "message": "No feedback data available for refinement"
        })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_path = f"finetuning/tb_chest_xray_refined_{timestamp}.pt"
    
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetuning'), exist_ok=True)
    
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetune.py'),
        "--old-model-path", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "tb_chest_xray_attention_best.pt"),
        "--new-model-path", os.path.join(os.path.dirname(os.path.abspath(__file__)), new_model_path),
        "--feedback-log", os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback", "feedback_log.csv"),
        "--feedback-images-dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback", "images"),
        "--feedback-masks-dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback", "masks"),
        "--epochs", "10",
        "--batch-size", "8"
    ]
    
    logging.info(f"Running finetuning command: {' '.join(cmd)}")
    
    try:
        finetuning_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        finetuning_status["running"] = True
        finetuning_status["message"] = "Model refinement in progress..."
        finetuning_status["current_epoch"] = 0
        finetuning_status["total_epochs"] = 10
        finetuning_status["start_time"] = time.time()
        finetuning_status["feedback_count"] = feedback_count
        
        thread = threading.Thread(target=monitor_finetuning_output, args=(finetuning_process,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Model refinement started successfully",
        })
    
    except Exception as e:
        logging.error(f"Error starting finetuning process: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        })

@app.route('/api/available_models', methods=['GET'])
@token_required
def get_available_models(current_user):
    default_model = "tb_chest_xray_attention_best.pt"
    
    refined_models = find_all_refined_models()
    
    return jsonify({
        "default_model": default_model,
        "refined_models": refined_models
    })

@app.route('/api/switch_model', methods=['POST'])
@token_required
def switch_model(current_user):
    if current_user['role'] != 'admin':
        return jsonify({
            "success": False,
            "message": "Insufficient permissions. Only administrators can switch models."
        }), 403
    
    global attention_model
    
    if finetuning_status["running"]:
        return jsonify({
            "success": False,
            "message": "Cannot switch models while refinement is in progress"
        })
    
    data = request.get_json()
    if not data or "model_name" not in data:
        return jsonify({
            "success": False,
            "message": "Model name not specified"
        })
        
    requested_model = data["model_name"]
    logging.info(f"Attempting to switch to model: {requested_model}")
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if requested_model == "tb_chest_xray_attention_best.pt":
            model_path = os.path.join(base_dir, "model", requested_model)
        else:
            model_path = os.path.join(base_dir, "finetuning", requested_model)
            
        if not os.path.isfile(model_path):
            return jsonify({
                "success": False,
                "message": f"Model file not found: {requested_model}"
            })
        
        attention_model = load_attention_model(model_path)
        
        return jsonify({
            "success": True,
            "message": f"Switched to model: {requested_model}",
            "model_name": requested_model
        })
    
    except Exception as e:
        logging.error(f"Error switching model: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        })

@app.route('/api/current_model', methods=['GET'])
@token_required
def get_current_model(current_user):
    model_name = "tb_chest_xray_attention_best.pt"
    
    return jsonify({"model_name": model_name})

if __name__ == "__main__":
    admin_password = os.environ.get('ADMIN_PASSWORD', 'admin')
    users['admin'] = {
        'username': 'admin',
        'password': generate_password_hash(admin_password),
        'role': 'admin'
    }
    
    users['user'] = {
        'username': 'user',
        'password': generate_password_hash('password'),
        'role': 'user'
    }
    
    print("Default users created:")
    print("- Admin user: username='admin', password='admin' (or set by ADMIN_PASSWORD env var)")
    print("- Regular user: username='user', password='password'")
    
    load_both_models()
    
    app.run(debug=True, host='0.0.0.0', port=8000)