import os
import csv
import base64
import time
import uuid
import sys
import subprocess
import threading

from flask import Flask, request, jsonify, render_template
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
# Finetuning status tracker
finetuning_status = {"running": False, "message": "", "success": False}

############################################################
# UTILITY: Calculate TB Probability
############################################################
def get_probability(img):
    """
    Calculate TB probability from image (accepts PIL, numpy, or tensor)
    Returns probability percentage (0-100)
    """
    # Handle different input types
    if isinstance(img, Image.Image):
        # Convert PIL to numpy
        img_np = np.array(img.convert("L"))
        img_np = cv2.resize(img_np, (256, 256))
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor / 255.0
    elif isinstance(img, np.ndarray):
        # Convert numpy to tensor
        img_np = cv2.resize(img, (256, 256)) if img.shape[0] != 256 else img
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor / 255.0
    elif isinstance(img, torch.Tensor):
        img_tensor = img
    else:
        raise TypeError("Image must be PIL Image, numpy array, or torch Tensor")
    
    # Forward pass
    with torch.no_grad():
        output, _ = attention_model(img_tensor)
        pred_prob = output.item()
        pred_prob_percent = round(pred_prob * 100, 2)
        
    return pred_prob_percent

############################################################
# UTILITY: Load the Trained Attention Model
############################################################
def load_attention_model(model_path=None):
    if model_path is None or not os.path.exists(model_path):
        # Use default model path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'model', 'tb_chest_xray_attention_best.pt')

    model = SimpleAttentionCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

############################################################
# UTILITY: Create a Heatmap Overlay from a Tensor Map
############################################################
def create_overlay(map_2d, original_gray, size=256, colormap=cv2.COLORMAP_INFERNO, alpha=0.7):
    map_2d = np.asarray(map_2d, dtype=np.float32)
    map_resized = cv2.resize(map_2d, (size, size))
    
    # Enhanced normalization with thresholding
    if map_resized.max() - map_resized.min() > 1e-5:
        # Regular min-max normalization
        map_resized = (map_resized - map_resized.min()) / (map_resized.max() - map_resized.min())
        
        # Applying thresholding to suppress weak activations
        threshold = 0.4  # Only show activations that are at least 40% of the max
        map_resized[map_resized < threshold] = 0
        
        # Re-normalize after thresholding if there are non-zero values left
        if map_resized.max() > 0:
            map_resized = map_resized / map_resized.max()
        
        # Apply gamma correction that SUPPRESSES weak signals
        map_resized = np.power(map_resized, 1.5)  # Use gamma > 1 to suppress weak signals
    else:
        # Create a more subtle fallback pattern for flat maps
        h, w = map_resized.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h/2, w/2
        map_resized = 1 - (((y - center_y)/(h/2))**2 + ((x - center_x)/(w/2))**2) / 2
        map_resized = np.clip(map_resized, 0, 1) * 0.7  # Reduce intensity of fallback pattern

    # Apply colormap with enhanced visibility
    heatmap = cv2.applyColorMap((map_resized * 255).astype(np.uint8), colormap)

    if len(original_gray.shape) == 2:
        original_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = original_gray
    original_bgr = cv2.resize(original_bgr, (size, size))

    # Use lower alpha for less overpowering visuals
    overlay = cv2.addWeighted(original_bgr, 1-alpha, heatmap, alpha, 0)
    return overlay

############################################################
# APP INIT + FOLDERS
############################################################
app = Flask(__name__)

# Feedback / Logging directories
base_dir = os.path.dirname(os.path.abspath(__file__))
feedback_dir = os.path.join(base_dir, 'feedback')
images_dir = os.path.join(feedback_dir, 'images')
masks_dir = os.path.join(feedback_dir, 'masks')
log_csv_path = os.path.join(feedback_dir, 'feedback_log.csv')
finetuning_dir = os.path.join(base_dir, 'finetuning')

# Create directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)
os.makedirs(finetuning_dir, exist_ok=True)

# Ensure feedback_log.csv has a header if not present
if not os.path.isfile(log_csv_path):
    with open(log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Example columns: image_filename, mask_filename, label, timestamp
        writer.writerow(["image_filename", "mask_filename", "label", "timestamp"])

# Track the current model path
current_model_path = os.path.join(base_dir, 'model', 'tb_chest_xray_attention_best.pt')
attention_model = load_attention_model(current_model_path)

############################################################
# UTILITY: Count Feedback Items
############################################################
def count_feedback_items():
    """Count the number of feedback entries in the CSV file"""
    if not os.path.exists(log_csv_path):
        return 0
    
    try:
        with open(log_csv_path, 'r') as f:
            # Subtract 1 for the header row
            return max(0, sum(1 for _ in f) - 1)
    except Exception as e:
        print(f"Error counting feedback items: {e}")
        return 0

############################################################
# ROUTES
############################################################
@app.route('/')
def home():
    feedback_count = count_feedback_items()
    return render_template('index.html', 
                          feedback_count=feedback_count,
                          current_model=os.path.basename(current_model_path))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

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

    # Get probability using the utility function
    pred_prob_percent = get_probability(img_tensor)

    # Forward pass (just for attention map)
    with torch.no_grad():
        _, attn_map = attention_model(img_tensor)

    # Attention overlay
    attn_map_np = attn_map[0].squeeze().cpu().numpy()  # (30,30)
    attn_overlay_bgr = create_overlay(attn_map_np, img_np, size=256)
    _, buffer = cv2.imencode('.jpg', attn_overlay_bgr)
    attn_overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    # Generate Grad-CAM - IMPROVED APPROACH
    try:
        # Try to find the best target layer dynamically based on activation strength
        try:
            # This will analyze activation patterns to find the best layer
            target_layer = find_best_target_layer(attention_model, img_tensor)
            print(f"Using dynamically selected layer for Grad-CAM")
        except Exception as e:
            print(f"Dynamic layer selection failed: {e}")
            # Fallback to predetermined layers with known good results
            if pred_prob_percent > 50:  # For TB positive cases
                # For TB positive, look at middle/earlier layers which often show lesions better
                target_layer = attention_model.feature_extractor[2]
                print(f"Using layer 2 for TB positive case")
            else:
                # For TB negative, often deeper layers work better to show absence of patterns
                if len(attention_model.feature_extractor) >= 4:
                    target_layer = attention_model.feature_extractor[4]
                    print(f"Using layer 4 for TB negative case")
                else:
                    target_layer = attention_model.feature_extractor[0]
                    print(f"Using layer 0 (fallback) for TB negative case")
                    
        _, grad_cam_overlay_bgr = show_grad_cam(
            img_np, 
            attention_model, 
            target_layer=target_layer,
            use_relu=True,
            smooth_factor=0.3,  # Reduced for even sharper results
            alpha=0.65  # Slightly increased for better visibility
        )
    except Exception as e:
        print(f"Error with first Grad-CAM attempt: {e}")
        
        try:
            # Second attempt with different parameters and layer
            # Prefer first conv layer for reliable gradients
            target_layer = attention_model.feature_extractor[0]
            print(f"Retrying Grad-CAM with first conv layer")
            
            _, grad_cam_overlay_bgr = show_grad_cam(
                img_np, 
                attention_model, 
                target_layer=target_layer,
                use_relu=False,  # Try without ReLU
                smooth_factor=0.5,
                alpha=0.7
            )
        except Exception as e:
            print(f"Error with second Grad-CAM attempt: {e}")
            
            # Use attention map as fallback
            print("Using attention map as fallback for Grad-CAM")
            grad_cam_overlay_bgr = attn_overlay_bgr
    
    # Encode Grad-CAM overlay to base64
    _, buffer = cv2.imencode('.jpg', grad_cam_overlay_bgr)
    grad_cam_overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    return render_template('index.html',
                           original_image=original_base64,
                           grad_cam_image=grad_cam_overlay_base64,
                           attention_image=attn_overlay_base64,
                           tb_probability=pred_prob_percent,
                           feedback_count=count_feedback_items(),
                           current_model=os.path.basename(current_model_path))

@app.route('/advanced_gradcam', methods=['POST'])
def advanced_gradcam():
    """
    Enhanced visualization endpoint with multiple Grad-CAM options
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'})

    file = request.files['file']
    
    # Load and preprocess
    pil_img = Image.open(file).convert("RGB")
    pil_img_gray = pil_img.convert("L")
    img_np = np.array(pil_img_gray)
    
    # Get probability for consistency
    pred_prob_percent = get_probability(pil_img_gray)
    
    results = []
    
    # IMPROVED: Layer-specific results
    layer_indices = []
    if len(attention_model.feature_extractor) >= 6:
        layer_indices = [0, 2, 4]  # Use first, middle and deeper layers
    elif len(attention_model.feature_extractor) >= 4:
        layer_indices = [0, 2]
    else:
        layer_indices = [0]
    
    # Try different layers
    for layer_idx in layer_indices:
        target_layer = attention_model.feature_extractor[layer_idx]
        
        try:
            # Generate visualization with improved parameters
            _, overlay_bgr = show_grad_cam(
                img_np, 
                attention_model, 
                target_layer=target_layer,
                use_relu=True,
                smooth_factor=0.3,
                alpha=0.65
            )
            
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', overlay_bgr)
            overlay_base64 = base64.b64encode(buffer).decode("utf-8")
            
            results.append({
                'name': f"Layer {layer_idx}",
                'image': overlay_base64
            })
        except Exception as e:
            print(f"Error with layer {layer_idx}: {e}")
    
    # Try different colormap options on the best layer (usually layer 2)
    best_layer_idx = 2 if len(attention_model.feature_extractor) >= 4 else 0
    target_layer = attention_model.feature_extractor[best_layer_idx]
    
    colormaps = [
        ('JET', cv2.COLORMAP_JET),
        ('INFERNO', cv2.COLORMAP_INFERNO),
        ('HOT', cv2.COLORMAP_HOT),
        ('VIRIDIS', cv2.COLORMAP_VIRIDIS)
    ]
    
    for cmap_name, cmap in colormaps:
        try:
            # Generate base visualization
            _, overlay_bgr = show_grad_cam(
                img_np, 
                attention_model, 
                target_layer=target_layer,
                use_relu=True,
                smooth_factor=0.3
            )
            
            # Extract grayscale heatmap
            gc_gray = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2GRAY) / 255.0
            
            # Apply colormap
            heatmap = cv2.applyColorMap(np.uint8(255 * gc_gray), cmap)
            colormap_overlay = cv2.addWeighted(
                cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR), 0.35, 
                heatmap, 0.65, 0
            )
            
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', colormap_overlay)
            overlay_base64 = base64.b64encode(buffer).decode("utf-8")
            
            results.append({
                'name': f"{cmap_name} Colormap",
                'image': overlay_base64
            })
        except Exception as e:
            print(f"Error with {cmap_name}: {e}")
    
    # Include probability in the response
    return jsonify({
        'visualizations': results,
        'tb_probability': pred_prob_percent
    })

@app.route('/debug_gradcam', methods=['POST'])
def debug_gradcam():
    """
    Debug endpoint that returns multiple Grad-CAM visualizations from different layers
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Load and preprocess
    pil_img = Image.open(file).convert("L")
    img_np = np.array(pil_img)
    
    # Calculate probability for consistency
    pred_prob_percent = get_probability(pil_img)
    
    # Try different layers and settings
    results = []
    
    # Try to find the best layer dynamically
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    try:
        best_layer = find_best_target_layer(attention_model, img_tensor)
        
        # Add a special visualization showing the best layer
        _, overlay = show_grad_cam(
            img_np,
            attention_model,
            target_layer=best_layer,
            use_relu=True,
            smooth_factor=0.3,
            alpha=0.65
        )
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', overlay)
        overlay_base64 = base64.b64encode(buffer).decode("utf-8")
        
        results.append({
            'layer': "Auto-Selected Best Layer",
            'relu': 'On',
            'image': overlay_base64
        })
    except Exception as e:
        print(f"Error finding best layer: {e}")
    
    # Also try specific layers
    for layer_idx in [0, 2, 4, 6]:
        if layer_idx < len(attention_model.feature_extractor):
            target_layer = attention_model.feature_extractor[layer_idx]
            
            # Try with and without ReLU
            for use_relu in [True, False]:
                try:
                    _, overlay = show_grad_cam(
                        img_np, 
                        attention_model, 
                        target_layer=target_layer,
                        use_relu=use_relu,
                        smooth_factor=0.3,  # Sharper focus
                        alpha=0.65  # Better visibility
                    )
                    
                    # Encode to base64
                    _, buffer = cv2.imencode('.jpg', overlay)
                    overlay_base64 = base64.b64encode(buffer).decode("utf-8")
                    
                    results.append({
                        'layer': f"Layer {layer_idx}",
                        'relu': 'On' if use_relu else 'Off',
                        'image': overlay_base64
                    })
                except Exception as e:
                    print(f"Error with layer {layer_idx}, relu={use_relu}: {e}")
    
    # If we got no results, add the attention map as fallback
    if not results:
        with torch.no_grad():
            _, attn_map = attention_model(img_tensor)
        
        attn_map_np = attn_map[0].squeeze().cpu().numpy()
        overlay = create_overlay(attn_map_np, img_np)
        
        _, buffer = cv2.imencode('.jpg', overlay)
        overlay_base64 = base64.b64encode(buffer).decode("utf-8")
        
        results.append({
            'layer': "Attention Map (Fallback)",
            'relu': 'N/A',
            'image': overlay_base64
        })
    
    # Include probability in the response
    return jsonify({
        'visualizations': results,
        'tb_probability': pred_prob_percent
    })

@app.route('/submit_mask', methods=['POST'])
def submit_mask():
    """
    Expects JSON like:
      {
        "image": <base64_jpeg>,
        "mask": <base64_png>,
        "label": <string>  # e.g., "TB" or "Normal"
      }
    Saves both image & mask to disk, logs them in feedback_log.csv, 
    and returns a JSON response indicating success.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload"}), 400

    base64_image = data.get("image", None)  # Base64-encoded JPG of the original
    base64_mask = data.get("mask", None)    # Base64-encoded PNG of the user's drawn mask
    user_label = data.get("label", "")      # "TB", "Normal", or another user-provided label

    if not base64_image or not base64_mask:
        return jsonify({"error": "Missing 'image' or 'mask' field in JSON"}), 400

    # Generate a timestamp-based unique ID
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    unique_id = f"{timestamp_str}_{uuid.uuid4().hex[:6]}"

    image_filename = f"{unique_id}_image.jpg"
    mask_filename  = f"{unique_id}_mask.png"

    image_path = os.path.join(images_dir, image_filename)
    mask_path  = os.path.join(masks_dir, mask_filename)

    # Decode & save the image
    try:
        image_data = base64.b64decode(base64_image)
        with open(image_path, "wb") as f_img:
            f_img.write(image_data)
    except Exception as e:
        return jsonify({"error": f"Failed to decode 'image': {str(e)}"}), 400

    # Decode & save the mask
    try:
        mask_data = base64.b64decode(base64_mask)
        with open(mask_path, "wb") as f_mask:
            f_mask.write(mask_data)
    except Exception as e:
        return jsonify({"error": f"Failed to decode 'mask': {str(e)}"}), 400

    # Log to CSV
    with open(log_csv_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Example columns: [image_filename, mask_filename, label, timestamp]
        writer.writerow([image_filename, mask_filename, user_label, timestamp_str])

    return jsonify({
        "status": "ok",
        "image_saved": image_filename,
        "mask_saved": mask_filename,
        "label": user_label,
        "timestamp": timestamp_str
    })

############################################################
# FINETUNING ROUTES
############################################################
@app.route('/run_finetuning', methods=['POST'])
def run_finetuning():
    """Trigger the offline fine-tuning process"""
    global finetuning_status
    
    # Check if fine-tuning is already running
    if finetuning_status.get("running", False):
        return jsonify({
            "success": False, 
            "message": "Fine-tuning already in progress"
        })
    
    # Check if there's any feedback data
    feedback_count = count_feedback_items()
    if feedback_count == 0:
        return jsonify({
            "success": False, 
            "message": "No feedback data found. Please collect feedback before fine-tuning."
        })
    
    # Generate timestamp for this fine-tuning run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Use absolute path and ensure directory exists
    abs_finetuning_dir = os.path.abspath(finetuning_dir)
    os.makedirs(abs_finetuning_dir, exist_ok=True)
    
    new_model_path = os.path.join(abs_finetuning_dir, f'tb_chest_xray_refined_{timestamp}.pt')
    
    # Log the paths for debugging
    print(f"DEBUG: Finetuning directory: {abs_finetuning_dir}")
    print(f"DEBUG: New model will be saved to: {new_model_path}")
    print(f"DEBUG: Directory exists: {os.path.isdir(abs_finetuning_dir)}")
    print(f"DEBUG: Directory is writable: {os.access(abs_finetuning_dir, os.W_OK)}")
    
    # Start fine-tuning in a separate thread to avoid blocking
    finetuning_status = {
        "running": True,
        "message": "Fine-tuning started...",
        "success": False,
        "timestamp": timestamp,
        "new_model_path": new_model_path  # Store the path for later reference
    }
    
    thread = threading.Thread(
        target=run_finetuning_process,
        args=(current_model_path, new_model_path)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "success": True,
        "message": "Fine-tuning process started. This may take several minutes.",
        "timestamp": timestamp
    })

def run_finetuning_process(old_model_path, new_model_path):
    """Run the fine-tuning process in a separate thread"""
    global finetuning_status
    
    try:
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
        
        print(f"DEBUG: Starting finetuning process")
        print(f"DEBUG: Old model path: {old_model_path}")
        print(f"DEBUG: New model path: {new_model_path}")
        
        # Construct the command with absolute paths
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune.py"),
            "--old-model-path", os.path.abspath(old_model_path),
            "--new-model-path", os.path.abspath(new_model_path),
            "--feedback-log", os.path.abspath(log_csv_path),
            "--feedback-images-dir", os.path.abspath(images_dir),
            "--feedback-masks-dir", os.path.abspath(masks_dir),
            "--epochs", "10"
        ]
        
        print(f"DEBUG: Running command: {' '.join(cmd)}")
        
        # Run the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        print(f"DEBUG: Process stdout: {stdout[:500]}...")  # Print beginning of stdout
        
        if stderr:
            print(f"DEBUG: Process stderr: {stderr}")
        
        if process.returncode == 0:
            # Check if file was actually created
            if os.path.exists(new_model_path):
                print(f"DEBUG: Model file was successfully created at {new_model_path}")
                finetuning_status = {
                    "running": False,
                    "message": f"Fine-tuning completed successfully! New model saved to: {os.path.basename(new_model_path)}",
                    "success": True,
                    "new_model_path": new_model_path
                }
            else:
                print(f"DEBUG: Process completed but model file was not found at {new_model_path}")
                finetuning_status = {
                    "running": False,
                    "message": f"Fine-tuning process completed, but the model file was not found. Check server logs.",
                    "success": False
                }
        else:
            finetuning_status = {
                "running": False,
                "message": f"Fine-tuning failed. Error: {stderr}",
                "success": False
            }
    except Exception as e:
        print(f"DEBUG: Exception in finetuning process: {str(e)}")
        finetuning_status = {
            "running": False,
            "message": f"Error during fine-tuning: {str(e)}",
            "success": False
        }

@app.route('/finetuning_status', methods=['GET'])
def get_finetuning_status():
    """Get the status of the current or last fine-tuning process"""
    global finetuning_status
    
    # Add feedback count to status
    status_copy = finetuning_status.copy()
    status_copy['feedback_count'] = count_feedback_items()
    
    # If finetuning is complete, check if file exists
    if status_copy.get('success', False) and 'new_model_path' in status_copy:
        model_path = status_copy['new_model_path']
        status_copy['model_exists'] = os.path.exists(model_path)
        
    return jsonify(status_copy)

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch between the original and fine-tuned model"""
    global attention_model, current_model_path
    
    # Checking if there's a refined model available
    if not os.path.exists(finetuning_dir):
        return jsonify({
            "success": False,
            "message": "No refined models available."
        })
    
    # Find the most recent refined model
    model_files = [f for f in os.listdir(finetuning_dir) if f.endswith('.pt')]
    if not model_files:
        return jsonify({
            "success": False,
            "message": "No refined models found in the finetuning directory."
        })
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(finetuning_dir, f)), reverse=True)
    latest_model = os.path.join(finetuning_dir, model_files[0])
    
    # Check if we're already using this model
    if current_model_path == latest_model:
        # Switch back to original model
        original_model_path = os.path.join(base_dir, 'model', 'tb_chest_xray_attention_best.pt')
        attention_model = load_attention_model(original_model_path)
        current_model_path = original_model_path
        
        return jsonify({
            "success": True,
            "message": "Switched back to original model.",
            "model_name": "tb_chest_xray_attention_best.pt"
        })
    else:
        # Switch to refined model
        try:
            attention_model = load_attention_model(latest_model)
            current_model_path = latest_model
            
            return jsonify({
                "success": True,
                "message": f"Switched to refined model: {os.path.basename(latest_model)}",
                "model_name": os.path.basename(latest_model)
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Error loading refined model: {str(e)}"
            })

if __name__ == "__main__":
    app.run(debug=True, port=8000)