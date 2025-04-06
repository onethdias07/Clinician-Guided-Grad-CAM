import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class HeatMapper:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.fwd_hook = target_layer.register_forward_hook(self.save_activations)
        self.bwd_hook = target_layer.register_full_backward_hook(self.save_grads)

    # this will save all the activations form the forward pass
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    # this will save all the gradients from the backward pass
    def save_grads(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def cleanup_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def __call__(self, x, target_class=None, use_relu=True, smooth_factor=0):
        was_training = self.model.training
        self.model.eval()
        
        x = x.clone()
        x.requires_grad_(True)
        
        model_output = self.model(x)
        
        if isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
        
        if target_class is None:
            target = output[0]
        else:
            target = output[0, target_class]
        
        self.model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            self.model.zero_grad()
            if len(output.shape) > 1 and output.shape[1] > 1:
                one_hot = torch.zeros_like(output)
                one_hot[0, target_class if target_class is not None else 0] = 1
                output.backward(gradient=one_hot, retain_graph=True)
            else:
                output.backward(torch.ones_like(output), retain_graph=True)
            
            if self.gradients is None or self.activations is None:
                raise ValueError("Couldn't grab gradients. Try another layer or check the model.")

        grads = self.gradients
        acts = self.activations
        b, c, h, w = grads.size()

        abs_grads = torch.abs(grads)
        weights = abs_grads.view(b, c, -1).mean(2)
        channel_weights = weights.view(b, c, 1, 1)
        
        heat_map = (channel_weights * acts).sum(dim=1, keepdim=True)

        if heat_map.max() < 0.01 and heat_map.max() - heat_map.min() > 1e-8:
            boost = 10000.0
            heat_map = heat_map * boost
            
        if heat_map.max() - heat_map.min() > 1e-7:
            heat_map = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min())
            
            if use_relu:
                heat_map = F.relu(heat_map)
                if heat_map.max() > 0:
                    heat_map = heat_map / heat_map.max()
        
        heat_np = heat_map.detach().cpu().numpy()[0, 0]
        
        if smooth_factor > 0:
            heat_np = gaussian_filter(heat_np, sigma=smooth_factor)
            
        heat_map = torch.from_numpy(heat_np).unsqueeze(0).unsqueeze(0)

        if was_training:
            self.model.train()
        
        self.cleanup_hooks()
        
        return heat_map


def find_good_layer(model, img_tensor):
    candidate_layers = []
    
    if hasattr(model, 'features'):
        for i, layer in enumerate(model.features):
            if isinstance(layer, torch.nn.Conv2d):
                candidate_layers.append((f"features[{i}]", layer))
    
    if hasattr(model, 'backbone'):
        for i, layer in enumerate(model.backbone):
            if isinstance(layer, torch.nn.Conv2d):
                candidate_layers.append((f"backbone[{i}]", layer))
    
    if not candidate_layers:
        raise ValueError("No good conv layers found in model")
    
    best_layer = None
    best_score = -1
    
    with torch.no_grad():
        model.eval()
        
        for layer_name, layer in candidate_layers:
            layer_outputs = []
            def hook_fn(module, input, output):
                layer_outputs.append(output.detach())
            
            hook = layer.register_forward_hook(hook_fn)
            
            _ = model(img_tensor)
            
            if layer_outputs:
                layer_score = torch.abs(layer_outputs[0]).mean().item()
                
                if layer_score > best_score:
                    best_score = layer_score
                    best_layer = layer
            
            hook.remove()
    
    return best_layer


def show_grad_cam(img_pil, model, target_class=None, target_layer=None, 
                  use_relu=True, smooth_factor=1.0, alpha=0.5):
    img = np.array(img_pil)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (256, 256))
    img_norm = img.astype('float32') / 255.0
    img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0)
    
    if target_layer is None:
        if hasattr(model, 'backbone'):
            for i, layer in enumerate(model.backbone):
                if isinstance(layer, torch.nn.Conv2d):
                    if i == 0:
                        first_conv_layer = layer
                    if i == 2 or i == 4:
                        target_layer = layer
                        break
            
            if target_layer is None and 'first_conv_layer' in locals():
                target_layer = first_conv_layer
            
        if target_layer is None:
            raise ValueError("Couldn't auto-pick a valid Conv2d target layer")

    layers_to_try = [target_layer]
    if hasattr(model, 'backbone'):
        for i, layer in enumerate(model.backbone):
            if isinstance(layer, torch.nn.Conv2d) and layer != target_layer:
                layers_to_try.append(layer)
                if len(layers_to_try) >= 3:
                    break
    
    for i, layer in enumerate(layers_to_try):
        try:
            heat_mapper = HeatMapper(model, target_layer=layer)
            heat_map = heat_mapper(img_tensor, target_class=target_class, 
                        use_relu=use_relu, smooth_factor=smooth_factor)
            break
        except Exception as e:
            if i == len(layers_to_try) - 1:
                raise ValueError(f"All layers failed for heatmap: {e}")

    heat_np = heat_map.detach().cpu().numpy()[0, 0]
    heat_np = cv2.resize(heat_np, (256, 256))
    
    if heat_np.max() - heat_np.min() > 1e-6:
        heat_np = (heat_np - heat_np.min()) / (heat_np.max() - heat_np.min())
        
        mean_val = np.mean(heat_np)
        threshold = min(0.6, max(0.3, mean_val * 2.5))
        
        heat_np[heat_np < threshold] = 0
        
        if heat_np.max() > 0:
            heat_np = heat_np / heat_np.max()
        
        heat_np = np.power(heat_np, 1.5)
        
        if heat_np.max() > 0:
            heat_uint8 = (heat_np * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            heat_enhanced = clahe.apply(heat_uint8)
            heat_np = heat_enhanced.astype(np.float32) / 255.0
    
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color_map = cv2.applyColorMap(np.uint8(255 * heat_np), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_color, 1-alpha, color_map, alpha, 0)
    
    return img_color, overlay
