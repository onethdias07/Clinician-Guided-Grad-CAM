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
from grad_cam.grad_cam import GradCAM, show_grad_cam, find_best_target_layer

############################################################
# GLOBAL STATE
############################################################
# Add finetuning global state
finetuning_process = None
finetuning_status = {
    "running": False,
    "message": "No refinement process running",
    "current_epoch": 0,
    "total_epochs": 10,
    "start_time": None,
    "feedback_count": 0
}

# Add models dictionary to store loaded models
loaded_models = {
    "original": None,
    "finetuned": None
}

############################################################
# USER MANAGEMENT
############################################################
# Simple in-memory user storage (in production, use a database)
users = {}
JWT_SECRET = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
JWT_EXPIRATION = 24 * 60 * 60  # 24 hours in seconds

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        # Return error if no token
        if not token:
            return jsonify({
                'message': 'Authentication token is missing',
                'authenticated': False
            }), 401
        
        # Verify token
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
            
        # If we get here, token is valid
        return f(current_user, *args, **kwargs)
    
    return decorated

############################################################
# UTILITY FUNCTIONS
############################################################
def get_probability(img, model_type="current"):
    """
    Calculate TB probability from image (accepts PIL, numpy, or tensor)
    Returns probability percentage (0-100)
    
    Parameters:
    - img: The image to analyze
    - model_type: "current" (active model), "original", or "finetuned"
    """
    if isinstance(img, Image.Image):
        img_np = np.array(img.convert("L"))
        img_np = cv2.resize(img_np, (256, 256))
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor / 255.0
    elif isinstance(img, np.ndarray):
        img_np = cv2.resize(img_np, (256, 256)) if img.shape[0] != 256 else img
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor / 255.0
    elif isinstance(img, torch.Tensor):
        img_tensor = img
    else:
        raise TypeError("Image must be PIL Image, numpy array, or torch Tensor")
    
    # Select the appropriate model
    if model_type == "original" and loaded_models["original"] is not None:
        model = loaded_models["original"]
    elif model_type == "finetuned" and loaded_models["finetuned"] is not None:
        model = loaded_models["finetuned"]
    else:
        model = attention_model  # Default to current model
    
    with torch.no_grad():
        output, _ = model(img_tensor)
        pred_prob = output.item()
        pred_prob_percent = round(pred_prob * 100, 2)
        
    return pred_prob_percent

def load_attention_model(model_path=None):
    if (model_path is None or not os.path.exists(model_path)):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'model', 'tb_chest_xray_attention_best.pt')

    model = SimpleAttentionCNN()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    model.eval()
    return model

def load_both_models():
    """Load both original and latest finetuned models for comparison"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load original model if not already loaded
    if loaded_models["original"] is None:
        original_path = os.path.join(base_dir, 'model', 'tb_chest_xray_attention_best.pt')
        try:
            loaded_models["original"] = load_attention_model(original_path)
            logging.info(f"Loaded original model from {original_path}")
        except Exception as e:
            logging.error(f"Error loading original model: {e}")
    
    # Find and load latest finetuned model if not already loaded
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
        threshold = 0.4  # Only show activations that are at least 40% of the max
        map_resized[map_resized < threshold] = 0
        
        if map_resized.max() > 0:
            map_resized = map_resized / map_resized.max()
        
        map_resized = np.power(map_resized, 1.5)  # Suppress weak signals
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

def process_image(file):
    """Common image processing logic for both web and API routes"""
    # Load and preprocess
    pil_img = Image.open(file).convert("RGB")
    pil_img_gray = pil_img.convert("L")

    # Convert original to base64
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    original_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Prepare for model input
    img_np = np.array(pil_img_gray)
    img_np = cv2.resize(img_np, (256, 256))
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor / 255.0

    # Get probability 
    pred_prob_percent = get_probability(img_tensor)

    # Forward pass for attention map
    with torch.no_grad():
        _, attn_map = attention_model(img_tensor)

    # Attention overlay
    attn_map_np = attn_map[0].squeeze().cpu().numpy()
    attn_overlay_bgr = create_overlay(attn_map_np, img_np, size=256)
    _, buffer = cv2.imencode('.jpg', attn_overlay_bgr)
    attn_overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    # Generate Grad-CAM
    try:
        # Try to find the best target layer dynamically
        try:
            target_layer = find_best_target_layer(attention_model, img_tensor)
        except Exception:
            if pred_prob_percent > 50:
                target_layer = attention_model.feature_extractor[2]
            else:
                if len(attention_model.feature_extractor) >= 4:
                    target_layer = attention_model.feature_extractor[4]
                else:
                    target_layer = attention_model.feature_extractor[0]
                    
        _, grad_cam_overlay_bgr = show_grad_cam(
            img_np, 
            attention_model, 
            target_layer=target_layer,
            use_relu=True,
            smooth_factor=0.3,
            alpha=0.65
        )
    except Exception:
        # Use attention map as fallback
        grad_cam_overlay_bgr = attn_overlay_bgr
    
    # Encode Grad-CAM overlay to base64
    _, buffer = cv2.imencode('.jpg', grad_cam_overlay_bgr)
    grad_cam_overlay_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return {
        'original_image': original_base64,
        'grad_cam_image': grad_cam_overlay_base64,
        'attention_image': attn_overlay_base64,
        'tb_probability': pred_prob_percent
    }

def process_image_with_comparison(file):
    """Process image and return results from both original and finetuned models"""
    # Load and preprocess
    pil_img = Image.open(file).convert("RGB")
    pil_img_gray = pil_img.convert("L")

    # Convert original to base64
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    original_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Prepare for model input
    img_np = np.array(pil_img_gray)
    img_np = cv2.resize(img_np, (256, 256))
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor / 255.0

    # Ensure both models are loaded
    original_loaded, finetuned_loaded = load_both_models()
    
    results = {
        'original_image': original_base64,
        'has_comparison': original_loaded and finetuned_loaded
    }
    
    # Process with original model if available
    if original_loaded:
        try:
            # Get probability
            original_prob = get_probability(img_tensor, "original")
            
            # Get attention maps
            with torch.no_grad():
                original_output, original_attn = loaded_models["original"](img_tensor)
            
            # Create attention overlay
            original_attn_np = original_attn[0].squeeze().cpu().numpy()
            original_overlay = create_overlay(original_attn_np, img_np, size=256)
            _, buffer = cv2.imencode('.jpg', original_overlay)
            original_attn_base64 = base64.b64encode(buffer).decode("utf-8")
            
            # Generate Grad-CAM
            try:
                target_layer = find_best_target_layer(loaded_models["original"], img_tensor)
                _, original_gradcam = show_grad_cam(
                    img_np, 
                    loaded_models["original"], 
                    target_layer=target_layer,
                    use_relu=True,
                    smooth_factor=0.3,
                    alpha=0.65
                )
            except Exception:
                # Use attention map as fallback
                original_gradcam = original_overlay
            
            # Encode Grad-CAM overlay to base64
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
    
    # Process with finetuned model if available
    if finetuned_loaded:
        try:
            # Get probability
            finetuned_prob = get_probability(img_tensor, "finetuned")
            
            # Get attention maps
            with torch.no_grad():
                finetuned_output, finetuned_attn = loaded_models["finetuned"](img_tensor)
            
            # Create attention overlay
            finetuned_attn_np = finetuned_attn[0].squeeze().cpu().numpy()
            finetuned_overlay = create_overlay(finetuned_attn_np, img_np, size=256)
            _, buffer = cv2.imencode('.jpg', finetuned_overlay)
            finetuned_attn_base64 = base64.b64encode(buffer).decode("utf-8")
            
            # Generate Grad-CAM
            try:
                target_layer = find_best_target_layer(loaded_models["finetuned"], img_tensor)
                _, finetuned_gradcam = show_grad_cam(
                    img_np, 
                    loaded_models["finetuned"], 
                    target_layer=target_layer,
                    use_relu=True,
                    smooth_factor=0.3,
                    alpha=0.65
                )
            except Exception:
                # Use attention map as fallback
                finetuned_gradcam = finetuned_overlay
            
            # Encode Grad-CAM overlay to base64
            _, buffer = cv2.imencode('.jpg', finetuned_gradcam)
            finetuned_gradcam_base64 = base64.b64encode(buffer).decode("utf-8")
            
            results.update({
                'finetuned_probability': finetuned_prob,
                'finetuned_attention': finetuned_attn_base64,
                'finetuned_gradcam': finetuned_gradcam_base64
            })
            
            # Calculate IoU and correlation if both models available
            if original_loaded:
                try:
                    # Calculate metrics between attention maps
                    iou = calculate_iou(original_attn_np, finetuned_attn_np)
                    correlation, _ = calculate_correlation(original_attn_np, finetuned_attn_np)
                    
                    results.update({
                        'attention_iou': round(float(iou), 4),
                        'attention_correlation': round(float(correlation), 4)
                    })
                except Exception as e:
                    logging.error(f"Error calculating attention metrics: {e}")
        except Exception as e:
            logging.error(f"Error processing with finetuned model: {e}")
            results['finetuned_error'] = str(e)
    
    # Also add normal processing results
    normal_results = process_image(file)
    results.update({
        'tb_probability': normal_results['tb_probability'],
        'grad_cam_image': normal_results['grad_cam_image'],
        'attention_image': normal_results['attention_image']
    })
    
    return results

def calculate_iou(attn_map1, attn_map2, threshold=0.5):
    """Calculate IoU between two attention maps"""
    # Normalize maps to [0,1]
    if attn_map1.max() > 0:
        attn_map1 = (attn_map1 - attn_map1.min()) / (attn_map1.max() - attn_map1.min())
    if attn_map2.max() > 0:
        attn_map2 = (attn_map2 - attn_map2.min()) / (attn_map2.max() - attn_map2.min())
        
    # Resize to same shape if needed
    if attn_map1.shape != attn_map2.shape:
        attn_map2 = cv2.resize(attn_map2, (attn_map1.shape[1], attn_map1.shape[0]))
        
    # Apply threshold
    binary_map1 = (attn_map1 >= threshold).astype(np.float32)
    binary_map2 = (attn_map2 >= threshold).astype(np.float32)
    
    # Calculate IoU
    intersection = np.logical_and(binary_map1, binary_map2).sum()
    union = np.logical_or(binary_map1, binary_map2).sum()
    
    if union == 0:
        return 0
    return intersection / union

def calculate_correlation(attn_map1, attn_map2):
    """Calculate Pearson correlation between two attention maps"""
    from scipy import stats
    
    # Normalize maps to [0,1]
    if attn_map1.max() > 0:
        attn_map1 = (attn_map1 - attn_map1.min()) / (attn_map1.max() - attn_map1.min())
    if attn_map2.max() > 0:
        attn_map2 = (attn_map2 - attn_map2.min()) / (attn_map2.max() - attn_map2.min())
    
    # Resize to same shape if needed
    if attn_map1.shape != attn_map2.shape:
        attn_map2 = cv2.resize(attn_map2, (attn_map1.shape[1], attn_map1.shape[0]))
    
    # Flatten arrays
    flat1 = attn_map1.flatten()
    flat2 = attn_map2.flatten()
    
    # Calculate correlation
    corr, p_value = stats.pearsonr(flat1, flat2)
    return corr, p_value

def save_feedback(image_data, mask_data, user_label):
    """Common feedback saving logic for both web and API routes"""
    # Generate a timestamp-based unique ID
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    unique_id = f"{timestamp_str}_{uuid.uuid4().hex[:6]}"

    image_filename = f"{unique_id}_image.jpg"
    mask_filename = f"{unique_id}_mask.png"

    image_path = os.path.join(images_dir, image_filename)
    mask_path = os.path.join(masks_dir, mask_filename)

    # Decode & save the image
    image_bytes = base64.b64decode(image_data)
    with open(image_path, "wb") as f_img:
        f_img.write(image_bytes)

    # Decode & save the mask
    mask_bytes = base64.b64decode(mask_data)
    with open(mask_path, "wb") as f_mask:
        f_mask.write(mask_bytes)

    # Log to CSV
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
    """Monitor the output of the finetuning process and update status"""
    global finetuning_status
    
    for line in process.stdout:
        logging.info(f"Finetuning output: {line.strip()}")
        
        # Check for epoch information in the output
        if "[Epoch" in line:
            try:
                # Extract epoch number from a line like: [Epoch 5/10] Train...
                parts = line.split('[Epoch ')[1].split(']')[0].split('/')
                current_epoch = int(parts[0])
                total_epochs = int(parts[1])
                
                finetuning_status["current_epoch"] = current_epoch
                finetuning_status["total_epochs"] = total_epochs
                finetuning_status["message"] = f"Refining model (Epoch {current_epoch}/{total_epochs})"
                
            except Exception as e:
                logging.error(f"Error parsing epoch info: {e}")

    # Process has finished
    if process.poll() is not None:
        exit_code = process.returncode
        if exit_code == 0:
            finetuning_status["message"] = "Model refinement completed successfully"
        else:
            finetuning_status["message"] = f"Model refinement failed with exit code {exit_code}"
        
        finetuning_status["running"] = False
        finetuning_status["current_epoch"] = finetuning_status["total_epochs"]  # Mark as complete

def find_latest_refined_model():
    """Find the most recently created model file in the finetuning directory"""
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetuning')
        if not os.path.exists(model_dir):
            return None
            
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and 'refined' in f]
        if not model_files:
            return None
            
        # Sort by modification time (newest first)
        newest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
        return newest_model
    except Exception as e:
        logging.error(f"Error finding latest model: {e}")
        return None

def find_all_refined_models():
    """Find all refined model files in the finetuning directory"""
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetuning')
        if not os.path.exists(model_dir):
            return []
            
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and 'refined' in f]
        if not model_files:
            return []
            
        # Sort by modification time (newest first)
        model_files.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
        return model_files
    except Exception as e:
        logging.error(f"Error finding refined models: {e}")
        return []

############################################################
# APP INITIALIZATION
############################################################
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Feedback / Logging directories
base_dir = os.path.dirname(os.path.abspath(__file__))
feedback_dir = os.path.join(base_dir, 'feedback')
images_dir = os.path.join(feedback_dir, 'images')
masks_dir = os.path.join(feedback_dir, 'masks')
log_csv_path = os.path.join(feedback_dir, 'feedback_log.csv')

# Create directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# Ensure feedback_log.csv has a header if not present
if not os.path.isfile(log_csv_path):
    with open(log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "mask_filename", "label", "timestamp"])

# Load the model
model_path = os.path.join(base_dir, 'model', 'tb_chest_xray_attention_best.pt')
attention_model = load_attention_model(model_path)

def count_feedback_items():
    """Count the number of feedback entries in the CSV file"""
    if not os.path.exists(log_csv_path):
        return 0
    
    try:
        with open(log_csv_path, 'r') as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception as e:
        print(f"Error counting feedback items: {e}")
        return 0

# Set up logging near the top of the file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(base_dir, "app_debug.log"))
    ]
)
logger = logging.getLogger("App")

############################################################
# AUTHENTICATION ROUTES
############################################################
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Username and password are required'}), 400
        
    username = data.get('username')
    
    # Check if user already exists
    if username in users:
        return jsonify({'message': 'User already exists'}), 409
    
    # Create new user with hashed password
    hashed_password = generate_password_hash(data.get('password'))
    users[username] = {
        'username': username,
        'password': hashed_password,
        'role': data.get('role', 'user')  # Default to 'user' role
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
    
    # Check if user exists
    if username not in users:
        return jsonify({'message': 'Invalid username or password'}), 401
    
    # Verify password
    user = users[username]
    if not check_password_hash(user['password'], data.get('password')):
        return jsonify({'message': 'Invalid username or password'}), 401
    
    # Generate JWT token
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

############################################################
# WEB ROUTES
############################################################
@app.route('/')
def home():
    feedback_count = count_feedback_items()
    return render_template('index.html', 
                          feedback_count=feedback_count,
                          current_model="tb_chest_xray_attention_best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    """Web route for prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        result = process_image(file)
        return render_template('index.html',
                            original_image=result['original_image'],
                            grad_cam_image=result['grad_cam_image'],
                            attention_image=result['attention_image'],
                            tb_probability=result['tb_probability'],
                            feedback_count=count_feedback_items(),
                            current_model="tb_chest_xray_attention_best.pt")
    except Exception as e:
        return render_template('index.html',
                            error=f"Error processing image: {str(e)}",
                            feedback_count=count_feedback_items(),
                            current_model="tb_chest_xray_attention_best.pt")

@app.route('/submit_mask', methods=['POST'])
def submit_mask():
    """Web route for submitting feedback"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload"}), 400

        base64_image = data.get("image", None)
        base64_mask = data.get("mask", None)
        user_label = data.get("label", "")

        if not base64_image or not base64_mask:
            return jsonify({"error": "Missing 'image' or 'mask' field in JSON"}), 400

        result = save_feedback(base64_image, base64_mask, user_label)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

############################################################
# PROTECTED API ENDPOINTS
############################################################
@app.route('/api/predict', methods=['POST'])
@token_required
def api_predict(current_user):
    """Protected API endpoint for image prediction, requires authentication"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Add support for interpretability comparison
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
    """Protected API endpoint for submitting feedback (mask + label)"""
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
    """Protected API endpoint to get the current feedback count"""
    count = count_feedback_items()
    return jsonify({"count": count})

@app.route('/api/finetuning_status', methods=['GET'])
@token_required
def check_finetuning_status(current_user):
    """Protected endpoint to get the current status of the finetuning process"""
    global finetuning_status
    # Update feedback count on status check
    finetuning_status["feedback_count"] = count_feedback_items()
    
    # Check if process is still running
    if finetuning_process is not None and finetuning_status["running"]:
        if finetuning_process.poll() is not None:  # Process has completed
            finetuning_status["running"] = False
            finetuning_status["message"] = "Model refinement completed"
            finetuning_status["current_epoch"] = finetuning_status["total_epochs"]
    
    return jsonify(finetuning_status)

@app.route('/api/run_finetuning', methods=['POST'])
@token_required
def run_finetuning(current_user):
    """Protected endpoint to start the finetuning process"""
    # Only allow users with 'admin' role to run finetuning
    if current_user['role'] != 'admin':
        return jsonify({
            "success": False,
            "message": "Insufficient permissions. Only administrators can run model refinement."
        }), 403
    
    global finetuning_process, finetuning_status
    
    # Log the request for debugging
    logging.info("Received request to /api/run_finetuning endpoint")
    
    # Check if already running
    if finetuning_status["running"]:
        return jsonify({
            "success": False,
            "message": "A refinement process is already running"
        })
    
    # Check if we have feedback data
    feedback_count = count_feedback_items()
    if feedback_count == 0:
        return jsonify({
            "success": False,
            "message": "No feedback data available for refinement"
        })
    
    # Create timestamp for model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_path = f"finetuning/tb_chest_xray_refined_{timestamp}.pt"
    
    # Make sure finetuning directory exists
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetuning'), exist_ok=True)
    
    # Set up the command
    cmd = [
        sys.executable,  # Current Python interpreter
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetune.py'),
        "--old-model-path", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "tb_chest_xray_attention_best.pt"),
        "--new-model-path", os.path.join(os.path.dirname(os.path.abspath(__file__)), new_model_path),
        "--feedback-log", os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback", "feedback_log.csv"),
        "--feedback-images-dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback", "images"),
        "--feedback-masks-dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback", "masks"),
        "--epochs", "10",
        "--batch-size", "8"
    ]
    
    # Log the command
    logging.info(f"Running finetuning command: {' '.join(cmd)}")
    
    try:
        # Start the process
        finetuning_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Update status
        finetuning_status["running"] = True
        finetuning_status["message"] = "Model refinement in progress..."
        finetuning_status["current_epoch"] = 0
        finetuning_status["total_epochs"] = 10
        finetuning_status["start_time"] = time.time()
        finetuning_status["feedback_count"] = feedback_count
        
        # Start a thread to monitor the output
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
    """Protected API endpoint to get all available models"""
    # Get the default model
    default_model = "tb_chest_xray_attention_best.pt"  # Always include the default model
    
    # Get all refined models
    refined_models = find_all_refined_models()
    
    # Return the list
    return jsonify({
        "default_model": default_model,
        "refined_models": refined_models
    })

@app.route('/api/switch_model', methods=['POST'])
@token_required
def switch_model(current_user):
    """Protected endpoint to switch to using a specific model"""
    # Only allow users with 'admin' role to switch models
    if current_user['role'] != 'admin':
        return jsonify({
            "success": False,
            "message": "Insufficient permissions. Only administrators can switch models."
        }), 403
    
    global attention_model
    
    # Check if finetuning is in progress
    if finetuning_status["running"]:
        return jsonify({
            "success": False,
            "message": "Cannot switch models while refinement is in progress"
        })
    
    # Get requested model from POST data
    data = request.get_json()
    if not data or "model_name" not in data:
        return jsonify({
            "success": False,
            "message": "Model name not specified"
        })
        
    requested_model = data["model_name"]
    logging.info(f"Attempting to switch to model: {requested_model}")
    
    try:
        # Determine model path based on whether it's the default or a refined model
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if requested_model == "tb_chest_xray_attention_best.pt":
            # Loading the default model
            model_path = os.path.join(base_dir, "model", requested_model)
        else:
            # Loading a refined model
            model_path = os.path.join(base_dir, "finetuning", requested_model)
            
        # Verify file exists
        if not os.path.isfile(model_path):
            return jsonify({
                "success": False,
                "message": f"Model file not found: {requested_model}"
            })
        
        # Load the new model
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
    """Protected API endpoint to get the current model name"""
    model_name = "tb_chest_xray_attention_best.pt"
    
    return jsonify({"model_name": model_name})

@app.route('/api/interpretability/compare', methods=['POST'])
@token_required
def compare_interpretability(current_user):
    """Protected endpoint for comparing interpretability between models"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        # Process image and get comparison results
        result = process_image_with_comparison(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Error comparing models: {str(e)}'}), 500

@app.route('/api/model_status', methods=['GET'])
@token_required
def get_model_status(current_user):
    """Protected endpoint to get the status of loaded models"""
    # Check if models are loaded
    original_loaded, finetuned_loaded = load_both_models()
    
    result = {
        'original_model': {
            'loaded': original_loaded,
            'path': 'model/tb_chest_xray_attention_best.pt'
        },
        'finetuned_model': {
            'loaded': finetuned_loaded,
            'path': find_latest_refined_model() if finetuned_loaded else None
        }
    }
    
    return jsonify(result)

# Public status endpoint - no authentication required
@app.route('/api/status', methods=['GET'])
def api_status():
    """Public API health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "API is running",
        "version": "1.0.0",
        "authentication_required": True
    })

if __name__ == "__main__":
    # Create a default admin user
    admin_password = os.environ.get('ADMIN_PASSWORD', 'admin')
    users['admin'] = {
        'username': 'admin',
        'password': generate_password_hash(admin_password),
        'role': 'admin'
    }
    
    # Create a default regular user
    users['user'] = {
        'username': 'user',
        'password': generate_password_hash('password'),
        'role': 'user'
    }
    
    print("Default users created:")
    print("- Admin user: username='admin', password='admin' (or set by ADMIN_PASSWORD env var)")
    print("- Regular user: username='user', password='password'")
    
    # Preload models for interpretability comparison
    load_both_models()
    
    app.run(debug=True, host='0.0.0.0', port=8000)