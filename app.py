from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from PIL import Image
import os
import cv2
from io import BytesIO
import base64
from model import load_model
from grad_cam.grad_cam import GradCAM  # Import the Grad-CAM utility class

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model()
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    img_pil = Image.open(file).convert("RGB")  # Convert to RGB

    # Convert the image to grayscale (important for the model)
    img_pil_gray = img_pil.convert("L")  # "L" mode for grayscale

    # Convert the original image to base64 for inline display
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Preprocess image for Grad-CAM
    img = np.array(img_pil_gray)  # Convert to numpy array
    img = cv2.resize(img, (256, 256))  # Resize to model's expected input size
    img_norm = img.astype('float32') / 255.0  # Normalize the image to [0,1]
    img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer=model.features[6])  # Adjust target layer if necessary

    # Get the Grad-CAM heatmap
    cam = gradcam(img_tensor)
    cam_np = cam.detach().numpy()[0, 0]
    cam_np = cv2.resize(cam_np, (256, 256))
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())  # Normalize heatmap

    # Create the overlay
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to BGR for overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_color, 0.5, heatmap, 0.5, 0)

    # Convert the Grad-CAM overlay to base64
    _, buffer = cv2.imencode('.jpg', overlay)
    cam_overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    # Perform classification and display the result
    with torch.no_grad():
        img_tensor = img_tensor.to(torch.device("cpu"))  # Ensure the tensor is on the correct device
        output = model(img_tensor)
        pred_prob = output.item()
        pred_prob_percent = round(pred_prob * 100, 2)

    # Return the template with both images and classification result
    return render_template('index.html', 
                           original_image=img_base64,  # Pass base64 for original image
                           grad_cam_image=cam_overlay_base64,  # Pass base64 for Grad-CAM overlay
                           tb_probability=pred_prob_percent)  # Display probability as percentage

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
