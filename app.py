from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from PIL import Image
import os
import cv2
from io import BytesIO
import base64

# Import your new Attention-based model and Grad-CAM
from attention_model import SimpleAttentionCNN, SpatialAttention
from grad_cam.grad_cam import GradCAM

############################################################
# UTILITY: Load the Trained Attention Model
############################################################
def load_attention_model():
    """
    Load the SimpleAttentionCNN with pre-trained weights.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model', 'tb_chest_xray_attention_best.pt')

    model = SimpleAttentionCNN()
    # Map to CPU if no GPU available
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

############################################################
# UTILITY: Create a Heatmap Overlay from a Tensor Map
############################################################
def create_overlay(map_2d, original_gray, size=256):
    """
    map_2d: 2D NumPy array (e.g., attention or Grad-CAM) with shape (H,W).
    original_gray: The original grayscale image as a NumPy array (H,W).
    size: final size for the overlay (e.g. 256).
    
    Returns: an OpenCV BGR overlay image (np.array) that you can convert to base64.
    """
    # Resize map to match final display size
    map_resized = cv2.resize(map_2d, (size, size))
    # Normalize to [0,1] for color mapping
    if map_resized.max() - map_resized.min() > 1e-5:
        map_resized = (map_resized - map_resized.min()) / (map_resized.max() - map_resized.min())
    else:
        map_resized = np.zeros_like(map_resized)

    heatmap = cv2.applyColorMap((map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Convert grayscale image to BGR
    if len(original_gray.shape) == 2:
        original_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = original_gray

    original_bgr = cv2.resize(original_bgr, (size, size))
    overlay = cv2.addWeighted(original_bgr, 0.5, heatmap, 0.5, 0)
    return overlay

############################################################
# FLASK APP INITIALIZATION
############################################################
app = Flask(__name__)

# Load the Attention-based model globally
attention_model = load_attention_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file part exists
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read image
    img_pil = Image.open(file).convert("RGB")  # Original color (could convert to L for grayscale)
    
    # Convert to grayscale for your model
    img_pil_gray = img_pil.convert("L")

    # Convert original image to base64 for display
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Preprocess for model inference
    img_np = np.array(img_pil_gray)  # shape: (H,W)
    img_np = cv2.resize(img_np, (256, 256))  # match model input
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,256,256]
    img_tensor = img_tensor / 255.0  # normalize to [0,1]

    # 1) Get classification & attention map from the attention model
    with torch.no_grad():
        output, attn_map = attention_model(img_tensor)  # output shape: [1,1], attn_map: [1,1,30,30]
        pred_prob = output.item()  # single scalar
        pred_prob_percent = round(pred_prob * 100, 2)

    # 2) Create an attention overlay (upsample attn_map to 256x256)
    attn_map_np = attn_map[0].squeeze().cpu().numpy()  # shape: (30,30)
    attention_overlay_bgr = create_overlay(attn_map_np, img_np, size=256)
    # Convert overlay to base64
    _, buffer = cv2.imencode('.jpg', attention_overlay_bgr)
    attn_overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    # 3) Use Grad-CAM on the final conv layer
    #    The final conv layer in your attention model is `attention_model.feature_extractor[6]`
    gradcam_tool = GradCAM(attention_model, target_layer=attention_model.feature_extractor[6])
    # Grad-CAM requires a forward + backward pass
    output_forward = attention_model(img_tensor)[0]  # forward pass => (output, attn_map)
    attention_model.zero_grad()
    output_forward[0].backward()

    cam = gradcam_tool.gradients  # captured by GradCAM hooks
    activations = gradcam_tool.activations

    # Flatten out the shape: [1, channels, H, W] => compute Grad-CAM
    # (We replicate logic from grad_cam.py, or we can use the existing GradCAM __call__ if refactored)
    # But let's do it explicitly:
    b, c, h, w = cam.size()
    alpha = cam.view(b, c, -1).mean(2).view(b, c, 1, 1)
    grad_cam_map = (activations * alpha).sum(dim=1, keepdim=True)
    grad_cam_map = torch.relu(grad_cam_map)  # ReLU
    gradcam_tool.remove_hooks()  # remove the hooks

    grad_cam_map_np = grad_cam_map[0, 0].detach().cpu().numpy()  # shape: (H, W) e.g. (30,30)
    # Resize to 256x256 and create overlay
    grad_cam_overlay_bgr = create_overlay(grad_cam_map_np, img_np, size=256)
    _, buffer = cv2.imencode('.jpg', grad_cam_overlay_bgr)
    grad_cam_overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    # Return the template with both images and classification result
    return render_template('index.html',
                           original_image=img_base64,
                           grad_cam_image=grad_cam_overlay_base64,
                           attention_image=attn_overlay_base64,
                           tb_probability=pred_prob_percent)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
