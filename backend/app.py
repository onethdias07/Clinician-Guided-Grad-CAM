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
# Front end msgs for the finetuning
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

# In-memory user store (would be replaced by database) -> do this if pushing to cloud
users = {}
JWT_SECRET = os.environ.get('JWT_SECRET_KEY', 'i-shall-replace-this-later')
JWT_EXPIRATION = 24 * 60 * 60  # this is 24 hours in seconds

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
    # this is a simplified input handling -> convert all inputs to tensor in a consistent way
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
    
    # Use dictionary lookup instead of conditionals to select model for better code clarity
    model_dict = {
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

# this the model loader
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

# this creates the grad-cam overlay
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

def process_base_image(file):
    # Process image file to PIL and numpy formats
    pil_img = Image.open(file).convert("RGB")
    pil_img_gray = pil_img.convert("L")

    # Create base64 of original image for UI display
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
    
    # Get tuberculosis probability 0 means normal and 1 means TB
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
        # Use attention overlay as fallback incase of grad-cam fails
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

# this is to save feedback to the feedback dir (its in backend/feedback)
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
        if "[Epoch" in line:
            try:
                parts = line.split('[Epoch ')[1].split(']')[0].split('/')
                current_epoch = int(parts[0])
                total_epochs = int(parts[1])
                
                finetuning_status["current_epoch"] = current_epoch
                finetuning_status["total_epochs"] = total_epochs
                finetuning_status["message"] = f"Refining model (Epoch {current_epoch}/{total_epochs})"
                
            except Exception as e:
                pass

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
        return None
# this will be used to fine and display all refined models in the UI
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
        return 0

# this is the main route for the app
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
# this is the login route for the app
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
# this is the token verification route for the app
@app.route('/api/auth/verify', methods=['GET'])
@token_required
def verify_token(current_user):

    return jsonify({
        'authenticated': True,
        'username': current_user['username'],
        'role': current_user['role']
    })
# this is the route for the home page of the app
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
        result = process_image(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

# this is the route for the feedback saving
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

# this is to display the feedback count in the UI
@app.route('/api/feedback_count', methods=['GET'])
@token_required
def api_feedback_count(current_user):
    count = count_feedback_items()
    return jsonify({"count": count})
# this is to display the finetuning status in the UI
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
# this is to run the finetuning process
@app.route('/api/run_finetuning', methods=['POST'])
@token_required
def run_finetuning(current_user):
    if current_user['role'] != 'admin':
        return jsonify({
            "success": False,
            "message": "Insufficient permissions. Only administrators can run model refinement."
        }), 403
    
    global finetuning_process, finetuning_status
    
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
    # this is all the cmd commands to run the finetuning process
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
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        })

# this is to display the select in the frontend
@app.route('/api/available_models', methods=['GET'])
@token_required
def get_available_models(current_user):
    default_model = "tb_chest_xray_attention_best.pt"
    
    refined_models = find_all_refined_models()
    
    return jsonify({
        "default_model": default_model,
        "refined_models": refined_models
    })

# this is to switchj models
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
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        })

# this is to display the current model in the frontend
@app.route('/api/current_model', methods=['GET'])
@token_required
def get_current_model(current_user):
    model_name = "tb_chest_xray_attention_best.pt"
    
    return jsonify({"model_name": model_name})

if __name__ == "__main__":
    admin_password = os.environ.get('ADMIN_PASSWORD', 'admin')
    # admin users who can run finetuning 
    users['admin'] = {
        'username': 'admin',
        'password': generate_password_hash(admin_password),
        'role': 'admin'
    }
    # this is for regular users like docs
    users['user'] = {
        'username': 'user',
        'password': generate_password_hash('password'),
        'role': 'user'
    }
    
    app.run(host='0.0.0.0', port=8000)