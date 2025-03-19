import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register forward hook to capture activations
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        
        # Use register_full_backward_hook instead of register_backward_hook
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)
        
        print(f"DEBUG: GradCAM initialized with target layer: {target_layer}")

    def save_activation(self, module, input, output):
        self.activations = output.detach()
        print(f"DEBUG: Saved activation with shape: {output.shape}")
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        if self.gradients is not None:
            print(f"DEBUG: Saved gradient with shape: {self.gradients.shape}")
            print(f"DEBUG: Gradient stats - min: {self.gradients.min().item():.6f}, max: {self.gradients.max().item():.6f}, mean: {self.gradients.mean().item():.6f}")
        else:
            print("DEBUG: Received None gradient")
    
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
        print("DEBUG: Removed hooks")

    def __call__(self, x, target_class=None, use_relu=True, smooth_factor=0):
        """
        Generate a Grad-CAM heatmap
        
        Args:
            x: Input tensor
            target_class: Index of target class to visualize (None for binary models)
            use_relu: Whether to apply ReLU to the final heatmap
            smooth_factor: Sigma for Gaussian smoothing (0 for no smoothing)
            
        Returns:
            cam: Grad-CAM heatmap
        """
        # Check model training state and input shape
        print(f"DEBUG: Model in training mode: {self.model.training}")
        print(f"DEBUG: Input shape: {x.shape}")
        
        # Store training state and set model to eval mode
        was_training = self.model.training
        self.model.eval()
        
        # Make sure input requires grad
        x = x.clone()
        x.requires_grad_(True)
        
        # Forward pass - handle both single and dual-output models
        model_output = self.model(x)
        
        # Handle SimpleAttentionCNN's dual output (output, attention_map)
        if isinstance(model_output, tuple):
            print("DEBUG: Model output is a tuple, taking first element")
            output = model_output[0]  # We want the class prediction, not the attention map
        else:
            print("DEBUG: Model output is not a tuple")
            output = model_output
        
        # DEBUG: Print output shape and value
        print(f"DEBUG: Output shape: {output.shape}, value: {output.detach().cpu().numpy()}")
        
        # Handle target class selection 
        if target_class is None:
            # For binary classification, use first output
            target = output[0]
            print(f"DEBUG: Using binary target: {target.item()}")
        else:
            # For multi-class, use specified target class
            target = output[0, target_class]
            print(f"DEBUG: Using target class {target_class}: {target.item()}")
        
        # Zero gradients before backprop
        self.model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        
        # Backprop to get gradients w.r.t. target class
        target.backward(retain_graph=True)

        # Check if we have gradients and activations
        if self.gradients is None or self.activations is None:
            print("WARNING: No gradients or activations captured! Trying one more approach...")
            
            # Try a different approach for backprop
            self.model.zero_grad()
            if len(output.shape) > 1 and output.shape[1] > 1:  # Multi-class
                one_hot = torch.zeros_like(output)
                one_hot[0, target_class if target_class is not None else 0] = 1
                output.backward(gradient=one_hot, retain_graph=True)
            else:  # Binary
                output.backward(torch.ones_like(output), retain_graph=True)
            
            # If still no gradients, raise error
            if self.gradients is None or self.activations is None:
                raise ValueError("Could not capture gradients. Try a different layer or model.")

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        b, c, h, w = gradients.size()

        # MODIFIED: Use absolute gradients for better visualization
        # This ensures we capture both positive and negative influences
        abs_gradients = torch.abs(gradients)
        
        # Global average pooling of absolute gradients
        alpha = abs_gradients.view(b, c, -1).mean(2)
        weights = alpha.view(b, c, 1, 1)
        print(f"DEBUG: Alpha weights stats - min: {alpha.min().item()}, max: {alpha.max().item()}, mean: {alpha.mean().item()}")
        
        # Weight activations by gradients
        cam = (weights * activations).sum(dim=1, keepdim=True)
        print(f"DEBUG: CAM stats - min: {cam.min().item()}, max: {cam.max().item()}, mean: {cam.mean().item()}")

        # MODIFIED: Amplify signal for very small values
        # This is crucial when gradients are extremely small (e.g., 1e-6 range)
        if cam.max() < 0.01 and cam.max() - cam.min() > 1e-8:
            print(f"DEBUG: Amplifying very small gradients by 10,000x")
            scale_factor = 10000.0
            cam = cam * scale_factor
            print(f"DEBUG: After amplification - min: {cam.min().item()}, max: {cam.max().item()}, mean: {cam.mean().item()}")
            
        # Handle normalization
        if cam.max() - cam.min() > 1e-7:
            # Normalize to 0-1 range
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            
            # Apply ReLU if specified (after normalization)
            if use_relu:
                cam = F.relu(cam)
                # Re-normalize after ReLU if needed
                if cam.max() > 0:
                    cam = cam / cam.max()
        else:
            print("WARNING: Flat activation map detected!")
        
        # Convert to numpy for potential smoothing
        cam_np = cam.detach().cpu().numpy()[0, 0]
        
        # Apply smoothing if requested
        if smooth_factor > 0:
            print(f"DEBUG: Applying Gaussian smoothing with sigma={smooth_factor}")
            cam_np = gaussian_filter(cam_np, sigma=smooth_factor)
            
        # Restore tensor format
        cam = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0)

        # Restore original model state
        if was_training:
            self.model.train()
        
        # Remove hooks
        self.remove_hooks()
        
        return cam


def find_best_target_layer(model, img_tensor):
    """
    Automatically find the best target layer for Grad-CAM by testing
    multiple layers and selecting the one with the strongest activations.
    
    Args:
        model: The neural network model
        img_tensor: Input image tensor (already preprocessed)
        
    Returns:
        best_layer: The best layer for Grad-CAM visualization
    """
    candidate_layers = []
    
    # Look for convolutional layers in common model architectures
    # For SimpleCNN or similar architectures
    if hasattr(model, 'features'):
        for i, layer in enumerate(model.features):
            if isinstance(layer, torch.nn.Conv2d):
                candidate_layers.append((f"features[{i}]", layer))
    
    # For SimpleAttentionCNN
    if hasattr(model, 'feature_extractor'):
        for i, layer in enumerate(model.feature_extractor):
            if isinstance(layer, torch.nn.Conv2d):
                candidate_layers.append((f"feature_extractor[{i}]", layer))
    
    # If no candidates found, raise error
    if not candidate_layers:
        raise ValueError("No suitable convolutional layers found in model")
    
    # Setup for layer testing
    best_layer = None
    best_activation_strength = -1
    
    with torch.no_grad():
        model.eval()
        
        # Test each candidate layer
        for layer_name, layer in candidate_layers:
            # Register a temporary forward hook
            activations = []
            def hook_fn(module, input, output):
                activations.append(output.detach())
            
            hook = layer.register_forward_hook(hook_fn)
            
            # Forward pass
            _ = model(img_tensor)
            
            # Calculate activation strength (use absolute mean as metric)
            if activations:
                activation_strength = torch.abs(activations[0]).mean().item()
                print(f"Layer {layer_name} activation strength: {activation_strength:.6f}")
                
                if activation_strength > best_activation_strength:
                    best_activation_strength = activation_strength
                    best_layer = layer
            
            # Remove the hook
            hook.remove()
    
    print(f"Selected best layer: {best_layer} with activation strength: {best_activation_strength:.6f}")
    return best_layer


def show_grad_cam(img_pil, model, target_class=None, target_layer=None, 
                  use_relu=True, smooth_factor=1.0, alpha=0.5):
    """
    Generate and blend a Grad-CAM visualization with the original image
    
    Args:
        img_pil: PIL image or numpy array
        model: The neural network model
        target_class: Class index to visualize (None for binary models)
        target_layer: Target layer for Grad-CAM (default: automatically selected)
        use_relu: Whether to apply ReLU to final heatmap
        smooth_factor: Sigma for Gaussian smoothing
        alpha: Blending factor for overlay (0-1)
        
    Returns:
        (original_img, overlay): Tuple with original image and Grad-CAM overlay
    """
    # Handle inputs that are already numpy arrays
    img = np.array(img_pil)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize and normalize input
    img = cv2.resize(img, (256, 256))
    img_norm = img.astype('float32') / 255.0
    img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0)

    print(f"DEBUG: Input image shape: {img.shape}, tensor shape: {img_tensor.shape}")
    
    # MODIFIED: Target only Conv2d layers, not activation layers like ReLU
    if target_layer is None:
        if hasattr(model, 'feature_extractor'):
            for i, layer in enumerate(model.feature_extractor):
                if isinstance(layer, torch.nn.Conv2d):
                    # Find the first Conv2d layer
                    if i == 0:
                        first_conv_layer = layer
                    # Find a middle conv layer if available (better feature visualization)
                    if i == 2 or i == 4:
                        target_layer = layer
                        print(f"DEBUG: Auto-selected Conv2d layer index {i}: {layer}")
                        break
            
            # If no middle conv layer was found, use the first one
            if target_layer is None and 'first_conv_layer' in locals():
                target_layer = first_conv_layer
                print(f"DEBUG: Using first Conv2d layer: {target_layer}")
            
        # If still no target layer, raise error
        if target_layer is None:
            raise ValueError("Could not auto-select a valid Conv2d target layer")

    # Define multiple layers to try if the first choice fails
    layers_to_try = [target_layer]
    if hasattr(model, 'feature_extractor'):
        # Add backup conv layers in case the first one fails
        for i, layer in enumerate(model.feature_extractor):
            if isinstance(layer, torch.nn.Conv2d) and layer != target_layer:
                layers_to_try.append(layer)
                if len(layers_to_try) >= 3:  # Limit to 3 backup layers
                    break
    
    # Generate Grad-CAM - try different layers if needed
    for i, layer in enumerate(layers_to_try):
        try:
            print(f"DEBUG: Creating GradCAM with target layer {i+1}: {layer}")
            gradcam = GradCAM(model, target_layer=layer)
            cam = gradcam(img_tensor, target_class=target_class, 
                        use_relu=use_relu, smooth_factor=smooth_factor)
            break  # Exit the loop if successful
        except Exception as e:
            print(f"ERROR with layer {i+1}: {e}")
            if i == len(layers_to_try) - 1:  # If this was the last layer to try
                raise ValueError(f"Failed to generate Grad-CAM with any layer: {e}")
            # Otherwise continue to the next layer

    # Convert to numpy and resize
    cam_np = cam.detach().cpu().numpy()[0, 0]
    print(f"DEBUG: CAM numpy shape before resize: {cam_np.shape}")
    cam_np = cv2.resize(cam_np, (256, 256))
    print(f"DEBUG: CAM numpy shape after resize: {cam_np.shape}")
    
    # MODIFIED: Enhanced normalization with adaptive thresholding
    if cam_np.max() - cam_np.min() > 1e-6:
        print(f"DEBUG: Normalizing with range: {cam_np.max() - cam_np.min()}")
        
        # Normalize to 0-1 range
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        
        # Calculate adaptive threshold based on mean value
        mean_val = np.mean(cam_np)
        print(f"DEBUG: Mean activation: {mean_val:.4f}")
        
        # Use lower threshold when mean is small
        threshold = min(0.6, max(0.3, mean_val * 2.5))
        print(f"DEBUG: Using adaptive threshold: {threshold:.4f}")
        
        cam_np[cam_np < threshold] = 0
        
        # Re-normalize after thresholding if there are non-zero values
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
        
        # Apply stronger gamma correction to enhance contrast
        cam_np = np.power(cam_np, 1.5)
        
        # Add contrast enhancement with CLAHE
        if cam_np.max() > 0:
            # Convert to uint8
            cam_uint8 = (cam_np * 255).astype(np.uint8)
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cam_enhanced = clahe.apply(cam_uint8)
            # Convert back to float [0,1]
            cam_np = cam_enhanced.astype(np.float32) / 255.0
            
        print(f"DEBUG: After processing - non-zero values: {np.count_nonzero(cam_np)} out of {cam_np.size}")
    
    # Create heatmap and overlay
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # MODIFIED: Use a more vivid colormap for better visibility
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    
    # Check if heatmap is blank and notify user
    if np.count_nonzero(cam_np) < 100:
        print("WARNING: Almost blank heatmap generated. Try a different layer.")
    
    overlay = cv2.addWeighted(img_color, 1-alpha, heatmap, alpha, 0)
    
    return img_color, overlay


def debug_grad_cam(img_path, model, save_path=None):
    """
    Debug function to run Grad-CAM with multiple configurations and save visualizations
    
    Args:
        img_path: Path to input image
        model: Model to use for Grad-CAM
        save_path: Path to save debug output (None for no saving)
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load image
    img = Image.open(img_path).convert('L')
    
    # Try different target layers
    target_layers = []
    
    # For SimpleCNN - try different layers
    if hasattr(model, 'features'):
        for i in [0, 2, 4, 6]:  # Include first conv layer
            if i < len(model.features):
                target_layers.append((f"features[{i}]", model.features[i]))
    
    # For SimpleAttentionCNN - try different layers
    if hasattr(model, 'feature_extractor'):
        for i in [0, 2, 4, 6]:  # Include first conv layer
            if i < len(model.feature_extractor):
                target_layers.append((f"feature_extractor[{i}]", model.feature_extractor[i]))
    
    # Setup figure for plotting
    n_rows = len(target_layers)
    n_cols = 5  # Original + 4 variations
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    
    if n_rows == 1:
        axs = axs.reshape(1, -1)
    
    # Display original image
    img_np = np.array(img)
    img_color = cv2.cvtColor(cv2.resize(img_np, (256, 256)), cv2.COLOR_GRAY2BGR)
    
    # For each target layer, try different parameters
    for i, (layer_name, target_layer) in enumerate(target_layers):
        print(f"\nTesting layer: {layer_name}")
        
        # Show original image in first column
        axs[i, 0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        axs[i, 0].set_title("Original Image")
        
        # 1. Default settings
        _, cam1 = show_grad_cam(img, model, target_layer=target_layer, 
                               use_relu=True, smooth_factor=1.0)
        
        # 2. No ReLU
        _, cam2 = show_grad_cam(img, model, target_layer=target_layer, 
                               use_relu=False, smooth_factor=1.0)
        
        # 3. More smoothing
        _, cam3 = show_grad_cam(img, model, target_layer=target_layer, 
                               use_relu=True, smooth_factor=2.0)
        
        # 4. Different colormap (INFERNO)
        _, cam4 = show_grad_cam(img, model, target_layer=target_layer,
                               use_relu=False, smooth_factor=1.5)
        # Apply different colormap manually
        cam4_gray = cv2.cvtColor(cam4, cv2.COLOR_BGR2GRAY)
        cam4_inferno = cv2.applyColorMap(cam4_gray, cv2.COLORMAP_INFERNO)
        
        # Display results
        axs[i, 1].imshow(cv2.cvtColor(cam1, cv2.COLOR_BGR2RGB))
        axs[i, 1].set_title(f"{layer_name}\nDefault")
        
        axs[i, 2].imshow(cv2.cvtColor(cam2, cv2.COLOR_BGR2RGB))
        axs[i, 2].set_title(f"{layer_name}\nNo ReLU")
        
        axs[i, 3].imshow(cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB))
        axs[i, 3].set_title(f"{layer_name}\nMore Smoothing")
        
        axs[i, 4].imshow(cv2.cvtColor(cam4_inferno, cv2.COLOR_BGR2RGB))
        axs[i, 4].set_title(f"{layer_name}\nINFERNO Colormap")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved debug visualization to {save_path}")
    else:
        plt.show()


def visualize_layer_outputs(model, img_path, save_path=None):
    """
    Visualize all feature maps from all convolutional layers in the model
    to help identify the most informative layers for Grad-CAM.
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load and preprocess image
    img = Image.open(img_path).convert('L')
    img_np = np.array(img)
    img = cv2.resize(img_np, (256, 256))
    img_tensor = torch.tensor(img.astype('float32') / 255.0).unsqueeze(0).unsqueeze(0)
    
    # Get all conv layers
    conv_layers = []
    
    # For SimpleCNN
    if hasattr(model, 'features'):
        for i, layer in enumerate(model.features):
            if isinstance(layer, torch.nn.Conv2d):
                conv_layers.append((f"features[{i}]", layer))
    
    # For SimpleAttentionCNN
    if hasattr(model, 'feature_extractor'):
        for i, layer in enumerate(model.feature_extractor):
            if isinstance(layer, torch.nn.Conv2d):
                conv_layers.append((f"feature_extractor[{i}]", layer))
    
    # Set up hooks to capture outputs
    outputs = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            outputs[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for name, layer in conv_layers:
        hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        model(img_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate grid size
    n_layers = len(conv_layers)
    
    if n_layers == 0:
        print("No convolutional layers found in model")
        return
    
    # Plot first 64 filters from each layer or fewer if layer has fewer filters
    fig = plt.figure(figsize=(20, 4 * n_layers))
    
    for i, (name, _) in enumerate(conv_layers):
        output = outputs[name]
        n_filters = min(output.shape[1], 8)  # Show up to 8 filters per layer
        
        # For each channel in output
        for j in range(n_filters):
            ax = fig.add_subplot(n_layers, 8, i*8 + j + 1)
            feature_map = output[0, j].cpu().numpy()
            ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f"{name}\nFilter {j}")
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved feature map visualization to {save_path}")
    else:
        plt.show()
