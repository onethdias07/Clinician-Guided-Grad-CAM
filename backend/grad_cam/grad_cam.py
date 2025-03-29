import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class GradCAM:
    def __init__(self, model, focus_layer):
        self.model = model
        self.focus_layer = focus_layer
        self.grads = None
        self.acts = None
        
        self.fwd_hook = focus_layer.register_forward_hook(self.store_activation)
        self.bwd_hook = focus_layer.register_full_backward_hook(self.store_gradient)
        
        print(f"GradCAM initialized with focus layer: {focus_layer}")

    def store_activation(self, module, input, output):
        self.acts = output.detach()
        print(f"Saved activation with shape: {output.shape}")
    
    def store_gradient(self, module, grad_input, grad_output):
        self.grads = grad_output[0].detach()
        if self.grads is not None:
            print(f"Saved gradient with shape: {self.grads.shape}")
            print(f"Gradient stats - min: {self.grads.min().item():.6f}, max: {self.grads.max().item():.6f}, mean: {self.grads.mean().item():.6f}")
        else:
            print("Received None gradient")
    
    def cleanup_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()
        print("Removed hooks")

    def __call__(self, x, target_class=None, use_relu=True, smooth_factor=0):
        print(f"Model in training mode: {self.model.training}")
        print(f"Input shape: {x.shape}")
        
        was_training = self.model.training
        self.model.eval()
        
        x = x.clone()
        x.requires_grad_(True)
        
        model_output = self.model(x)
        
        if isinstance(model_output, tuple):
            print("Model output is a tuple, taking first element")
            output = model_output[0]
        else:
            print("Model output is not a tuple")
            output = model_output
        
        print(f"Output shape: {output.shape}, value: {output.detach().cpu().numpy()}")
        
        if target_class is None:
            target = output[0]
            print(f"Using binary target: {target.item()}")
        else:
            target = output[0, target_class]
            print(f"Using target class {target_class}: {target.item()}")
        
        self.model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        
        target.backward(retain_graph=True)

        if self.grads is None or self.acts is None:
            print("WARNING: No gradients or activations captured! Trying one more approach...")
            
            self.model.zero_grad()
            if len(output.shape) > 1 and output.shape[1] > 1:
                one_hot = torch.zeros_like(output)
                one_hot[0, target_class if target_class is not None else 0] = 1
                output.backward(gradient=one_hot, retain_graph=True)
            else:
                output.backward(torch.ones_like(output), retain_graph=True)
            
            if self.grads is None or self.acts is None:
                raise ValueError("Could not capture gradients. Try a different layer or model.")

        grads = self.grads
        acts = self.acts
        b, c, h, w = grads.size()

        abs_grads = torch.abs(grads)
        
        alpha = abs_grads.view(b, c, -1).mean(2)
        weights = alpha.view(b, c, 1, 1)
        print(f"Alpha weights stats - min: {alpha.min().item()}, max: {alpha.max().item()}, mean: {alpha.mean().item()}")
        
        cam = (weights * acts).sum(dim=1, keepdim=True)
        print(f"CAM stats - min: {cam.min().item()}, max: {cam.max().item()}, mean: {cam.mean().item()}")

        if cam.max() < 0.01 and cam.max() - cam.min() > 1e-8:
            print(f"Amplifying very small gradients by 10,000x")
            scale_factor = 10000.0
            cam = cam * scale_factor
            print(f"After amplification - min: {cam.min().item()}, max: {cam.max().item()}, mean: {cam.mean().item()}")
            
        if cam.max() - cam.min() > 1e-7:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            
            if use_relu:
                cam = F.relu(cam)
                if cam.max() > 0:
                    cam = cam / cam.max()
        else:
            print("WARNING: Flat activation map detected!")
        
        cam_np = cam.detach().cpu().numpy()[0, 0]
        
        if smooth_factor > 0:
            print(f"Applying Gaussian smoothing with sigma={smooth_factor}")
            cam_np = gaussian_filter(cam_np, sigma=smooth_factor)
            
        cam = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0)

        if was_training:
            self.model.train()
        
        self.cleanup_hooks()
        
        return cam


def find_good_layer(model, img_tensor):
    possible_layers = []
    
    if hasattr(model, 'features'):
        for i, layer in enumerate(model.features):
            if isinstance(layer, torch.nn.Conv2d):
                possible_layers.append((f"features[{i}]", layer))
    
    if hasattr(model, 'backbone'):
        for i, layer in enumerate(model.backbone):
            if isinstance(layer, torch.nn.Conv2d):
                possible_layers.append((f"backbone[{i}]", layer))
    
    if not possible_layers:
        raise ValueError("No suitable convolutional layers found in model")
    
    chosen_layer = None
    best_activation = -1
    
    with torch.no_grad():
        model.eval()
        
        for layer_name, layer in possible_layers:
            activations = []
            def hook_fn(module, input, output):
                activations.append(output.detach())
            
            hook = layer.register_forward_hook(hook_fn)
            
            _ = model(img_tensor)
            
            if activations:
                layer_strength = torch.abs(activations[0]).mean().item()
                print(f"Layer {layer_name} activation strength: {layer_strength:.6f}")
                
                if layer_strength > best_activation:
                    best_activation = layer_strength
                    chosen_layer = layer
            
            hook.remove()
    
    print(f"Selected best layer: {chosen_layer} with activation strength: {best_activation:.6f}")
    return chosen_layer


def show_grad_cam(img_pil, model, target_class=None, focus_layer=None, 
                  use_relu=True, smooth_factor=1.0, alpha=0.5):
    img = np.array(img_pil)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (256, 256))
    img_norm = img.astype('float32') / 255.0
    img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0)

    print(f"Input image shape: {img.shape}, tensor shape: {img_tensor.shape}")
    
    if focus_layer is None:
        if hasattr(model, 'backbone'):
            for i, layer in enumerate(model.backbone):
                if isinstance(layer, torch.nn.Conv2d):
                    if i == 0:
                        first_conv_layer = layer
                    if i == 2 or i == 4:
                        focus_layer = layer
                        print(f"Auto-selected Conv2d layer index {i}: {layer}")
                        break
            
            if focus_layer is None and 'first_conv_layer' in locals():
                focus_layer = first_conv_layer
                print(f"Using first Conv2d layer: {focus_layer}")
            
        if focus_layer is None:
            raise ValueError("Could not auto-select a valid Conv2d focus layer")

    layers_to_try = [focus_layer]
    if hasattr(model, 'backbone'):
        for i, layer in enumerate(model.backbone):
            if isinstance(layer, torch.nn.Conv2d) and layer != focus_layer:
                layers_to_try.append(layer)
                if len(layers_to_try) >= 3:
                    break
    
    for i, layer in enumerate(layers_to_try):
        try:
            print(f"Creating GradCAM with focus layer {i+1}: {layer}")
            gradcam = GradCAM(model, focus_layer=layer)
            cam = gradcam(img_tensor, target_class=target_class, 
                        use_relu=use_relu, smooth_factor=smooth_factor)
            break
        except Exception as e:
            print(f"ERROR with layer {i+1}: {e}")
            if i == len(layers_to_try) - 1:
                raise ValueError(f"Failed to generate Grad-CAM with any layer: {e}")

    cam_np = cam.detach().cpu().numpy()[0, 0]
    print(f"CAM numpy shape before resize: {cam_np.shape}")
    cam_np = cv2.resize(cam_np, (256, 256))
    print(f"CAM numpy shape after resize: {cam_np.shape}")
    
    if cam_np.max() - cam_np.min() > 1e-6:
        print(f"Normalizing with range: {cam_np.max() - cam_np.min()}")
        
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        
        mean_val = np.mean(cam_np)
        print(f"Mean activation: {mean_val:.4f}")
        
        threshold = min(0.6, max(0.3, mean_val * 2.5))
        print(f"Using adaptive threshold: {threshold:.4f}")
        
        cam_np[cam_np < threshold] = 0
        
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
        
        cam_np = np.power(cam_np, 1.5)
        
        if cam_np.max() > 0:
            cam_uint8 = (cam_np * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cam_enhanced = clahe.apply(cam_uint8)
            cam_np = cam_enhanced.astype(np.float32) / 255.0
            
        print(f"After processing - non-zero values: {np.count_nonzero(cam_np)} out of {cam_np.size}")
    
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    
    if np.count_nonzero(cam_np) < 100:
        print("WARNING: Almost blank heatmap generated. Try a different layer.")
    
    overlay = cv2.addWeighted(img_color, 1-alpha, heatmap, alpha, 0)
    
    return img_color, overlay
