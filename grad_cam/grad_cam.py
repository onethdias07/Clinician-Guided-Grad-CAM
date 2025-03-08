import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register forward and backward hooks to capture activations and gradients
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def __call__(self, x):
        output = self.model(x)
        
        # Backprop to get gradients w.r.t. target class
        self.model.zero_grad()
        output[0].backward()

        gradients = self.gradients
        activations = self.activations
        b, c, h, w = gradients.size()

        # Calculate alpha weights (global average pooling)
        alpha = gradients.view(b, c, -1).mean(2)
        weights = alpha.view(b, c, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        
        # Apply ReLU to the CAM to discard negative values
        cam = F.relu(cam)

        # Remove hooks
        self.remove_hooks()
        
        return cam

def show_grad_cam(img_pil, model):
    img = np.array(img_pil)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (256, 256))
    img_norm = img.astype('float32') / 255.0
    img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0)

    gradcam = GradCAM(model, target_layer=model.features[6])  # Specify the target conv layer
    cam = gradcam(img_tensor)

    cam_np = cam.detach().numpy()[0, 0]
    cam_np = cv2.resize(cam_np, (256, 256))
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_color, 0.5, heatmap, 0.5, 0)

    return img_color, overlay
