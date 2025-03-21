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
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import torch
import numpy as np
from PIL import Image
import cv2
from io import BytesIO

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

############################################################
# UTILITY FUNCTIONS
############################################################
def get_probability(img):
    """
    Calculate TB probability from image (accepts PIL, numpy, or tensor)
    Returns probability percentage (0-100)
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
    
    with torch.no_grad():
        output, _ = attention_model(img_tensor)
        pred_prob = output.item()
        pred_prob_percent = round(pred_prob * 100, 2)
        
    return pred_prob_percent

def load_attention_model(model_path=None):
    if model_path is None or not os.path.exists(model_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'model', 'tb_chest_xray_attention_best.pt')

    model = SimpleAttentionCNN()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    model.eval()
    return model

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

def save_feedback(image_data, mask_data, user_label):
    """Common feedback saving logic for both web and API routes"""
    # Generate a timestamp-based unique ID
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    unique_id = f"{timestamp_str}_{uuid.uuid4().hex[:6]}"

    # Check file extension in the base64 data
    if "image/png" in image_data.split(";")[0]:
        image_filename = f"{unique_id}_image.png"
    else:
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
# API ENDPOINTS
############################################################
@app.route('/api/status', methods=['GET'])
def api_status():
    """API health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "API is running",
        "version": "1.0.0"
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for image prediction, accepts multipart/form-data"""
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

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """API endpoint for submitting feedback (mask + label)"""
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
def api_feedback_count():
    """API endpoint to get the current feedback count"""
    count = count_feedback_items()
    return jsonify({"count": count})

@app.route('/api/finetuning_status', methods=['GET'])
def check_finetuning_status():
    """Get the current status of the finetuning process"""
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
def run_finetuning():
    """Start the finetuning process with enhanced error handling"""
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
    
    # Check for required dependencies
    try:
        import psutil
        logging.info("psutil dependency found")
    except ImportError:
        logging.error("psutil dependency missing - installing")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
            logging.info("psutil installed successfully")
        except Exception as e:
            logging.error(f"Failed to install psutil: {e}")
            return jsonify({
                "success": False,
                "message": f"Failed to install required dependency: {str(e)}"
            })
    
    # Set up the command with the new parameters
    cmd = [
        sys.executable,  # Current Python interpreter
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetune.py'),
        "--old-model-path", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "tb_chest_xray_attention_best.pt"),
        "--new-model-path", os.path.join(os.path.dirname(os.path.abspath(__file__)), new_model_path),
        "--feedback-log", os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback", "feedback_log.csv"),
        "--feedback-images-dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback", "images"),
        "--feedback-masks-dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback", "masks"),
        "--include-original-data",
        "--balance-datasets",
        "--freeze-layers", "9",
        "--gradual-unfreeze",
        "--initial-lr", "1e-5",
        "--final-lr", "1e-4",
        "--epochs", "25",
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
def get_available_models():
    """API endpoint to get all available models"""
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
def switch_model():
    """Switch to using a specific model"""
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
def get_current_model():
    """API endpoint to get the current model name"""
    model_name = "tb_chest_xray_attention_best.pt"
    
    return jsonify({"model_name": model_name})

@app.route('/api/finetuning_logs', methods=['GET'])
def get_finetuning_logs():
    """API endpoint to get the latest finetuning log file contents"""
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        if not os.path.exists(log_dir):
            return jsonify({
                "success": False,
                "message": "No log directory found",
                "log_content": ""
            })
            
        # Find the most recent log file
        log_files = [f for f in os.listdir(log_dir) if f.startswith('finetune_') and f.endswith('.log')]
        if not log_files:
            return jsonify({
                "success": False,
                "message": "No log files found",
                "log_content": ""
            })
            
        # Sort by modification time (newest first)
        newest_log = max(log_files, key=lambda f: os.path.getmtime(os.path.join(log_dir, f)))
        log_path = os.path.join(log_dir, newest_log)
        
        # Read log file content
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
            
        return jsonify({
            "success": True,
            "message": f"Retrieved log file: {newest_log}",
            "log_content": log_content,
            "log_file": newest_log
        })
        
    except Exception as e:
        logging.error(f"Error retrieving finetuning logs: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "log_content": ""
        })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)