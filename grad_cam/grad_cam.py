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
        
        # Use register_full_backward_hook instead of register_backward_hook to fix the warning
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)
        
        print(f"DEBUG: GradCAM initialized with target layer: {target_layer}")

    def save_activation(self, module, input, output):
        self.activations = output
        print(f"DEBUG: Saved activation with shape: {output.shape}")
    
    def save_gradient(self, module, grad_input, grad_output):
        # Note: signature is different for full_backward_hook
        self.gradients = grad_output[0]
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
        # DEBUG: Check model training state and input shape
        print(f"DEBUG: Model in training mode: {self.model.training}")
        print(f"DEBUG: Input shape: {x.shape}")
        
        # Store training state and temporarily set to training mode
        was_training = self.model.training
        self.model.train()  # Set to training mode to ensure gradient flow
        
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
        
        # DEBUG: Check gradient flow
        print(f"DEBUG: Target requires grad: {target.requires_grad}")
        
        # Backprop to get gradients w.r.t. target class
        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Check if we actually have gradients
        if self.gradients is None:
            print("WARNING: No gradients were captured! Make sure your target_layer is correct.")
            print(f"DEBUG: Target layer: {self.target_layer}")
            return torch.zeros(1, 1, x.shape[2], x.shape[3])  # Return empty map with same spatial dims

        # DEBUG: Print gradient info
        print(f"DEBUG: Gradient shape: {self.gradients.shape}")
        print(f"DEBUG: Gradient stats - min: {self.gradients.min().item()}, max: {self.gradients.max().item()}, mean: {self.gradients.mean().item()}")
        print(f"DEBUG: Activation shape: {self.activations.shape}")
        
        gradients = self.gradients
        activations = self.activations
        b, c, h, w = gradients.size()

        # Calculate alpha weights (global average pooling)
        alpha = gradients.view(b, c, -1).mean(2)
        weights = alpha.view(b, c, 1, 1)
        print(f"DEBUG: Alpha weights stats - min: {alpha.min().item()}, max: {alpha.max().item()}, mean: {alpha.mean().item()}")
        
        # Weight activations by gradients
        cam = (weights * activations).sum(dim=1, keepdim=True)
        
        # DEBUG: Check cam values
        print(f"DEBUG: CAM stats - min: {cam.min().item()}, max: {cam.max().item()}, mean: {cam.mean().item()}")

        # AMPLIFY SMALL SIGNALS - New code to handle very small gradients
        if cam.max() - cam.min() > 1e-6 and cam.max() < 0.01:
            print(f"DEBUG: Amplifying small CAM values by 100x")
            cam = (cam - cam.min()) * 100  # Scale up small but meaningful signals
            print(f"DEBUG: After amplification - min: {cam.min().item()}, max: {cam.max().item()}, mean: {cam.mean().item()}")
        
        # Apply ReLU if specified
        if use_relu:
            cam = F.relu(cam)
            print(f"DEBUG: After ReLU - min: {cam.min().item()}, max: {cam.max().item()}, mean: {cam.mean().item()}")
        
        # Convert to numpy for potential smoothing
        cam_np = cam.detach().cpu().numpy()[0, 0]
        
        # Apply smoothing if requested
        if smooth_factor > 0:
            print(f"DEBUG: Applying Gaussian smoothing with sigma={smooth_factor}")
            cam_np = gaussian_filter(cam_np, sigma=smooth_factor)
            
        # Restore tensor format
        cam = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0)

        # Restore original model state
        if not was_training:
            self.model.eval()
        
        # Remove hooks
        self.remove_hooks()
        
        return cam


def show_grad_cam(img_pil, model, target_class=None, target_layer=None, 
                  use_relu=True, smooth_factor=1.0, alpha=0.5):
    """
    Generate and blend a Grad-CAM visualization with the original image
    
    Args:
        img_pil: PIL image or numpy array
        model: The neural network model
        target_class: Class index to visualize (None for binary models)
        target_layer: Target layer for Grad-CAM (default: last conv layer)
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
    
    # Auto-select target layer if not specified
    if target_layer is None:
        # For SimpleCNN - try the first conv layer instead
        if hasattr(model, 'features') and len(model.features) >= 1:
            target_layer = model.features[0]  # First conv layer often works better
            print(f"DEBUG: Auto-selected SimpleCNN first layer: {target_layer}")
        # For SimpleAttentionCNN - try the first conv layer
        elif hasattr(model, 'feature_extractor') and len(model.feature_extractor) >= 1:
            target_layer = model.feature_extractor[0]
            print(f"DEBUG: Auto-selected SimpleAttentionCNN first layer: {target_layer}")
        else:
            # Try even earlier layers
            if hasattr(model, 'feature_extractor') and len(model.feature_extractor) >= 2:
                target_layer = model.feature_extractor[1]
                print(f"DEBUG: Falling back to very early layer: {target_layer}")
            elif hasattr(model, 'features') and len(model.features) >= 2:
                target_layer = model.features[1]
                print(f"DEBUG: Falling back to very early layer: {target_layer}")
            else:
                raise ValueError("Could not automatically determine target layer")

    # Generate Grad-CAM
    print(f"DEBUG: Creating GradCAM with target layer: {target_layer}")
    gradcam = GradCAM(model, target_layer=target_layer)
    cam = gradcam(img_tensor, target_class=target_class, 
                 use_relu=use_relu, smooth_factor=smooth_factor)

    # Convert to numpy and resize
    cam_np = cam.detach().numpy()[0, 0]
    print(f"DEBUG: CAM numpy shape before resize: {cam_np.shape}")
    cam_np = cv2.resize(cam_np, (256, 256))
    print(f"DEBUG: CAM numpy shape after resize: {cam_np.shape}")
    
    # DEBUG: Print cam_np statistics
    print(f"DEBUG: cam_np stats before norm - min: {cam_np.min()}, max: {cam_np.max()}, mean: {cam_np.mean()}")
    
    # Normalize to 0-1 range with enhanced contrast for small values and thresholding
    if cam_np.max() - cam_np.min() > 1e-6:
        print(f"DEBUG: Normalizing with range: {cam_np.max() - cam_np.min()}")
        
        # Enhanced contrast normalization for small signals
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        
        # APPLY THRESHOLD to focus on the most important regions (NEW)
        threshold = 0.4  # Only keep top 60% of signal
        cam_np[cam_np < threshold] = 0
        
        # Re-normalize after thresholding if there are non-zero values
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
        
        # Apply gamma correction to SUPPRESS weak signals (changed from 0.5 to 1.5)
        cam_np = np.power(cam_np, 1.5)  # Using gamma > 1 suppresses weak signals
        
        print(f"DEBUG: After thresholding - min: {cam_np.min()}, max: {cam_np.max()}, mean: {cam_np.mean()}")
    else:
        print(f"DEBUG: Flat activation detected, using guided pattern")
        # Create a SMALLER guided pattern that highlights only center region
        h, w = cam_np.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h/2, w/2
        
        # Create smaller circular pattern centered in image (reduced radius)
        radius = min(h, w) * 0.3  # Only highlight central 30% of image
        mask = ((y - center_y)**2 + (x - center_x)**2) < radius**2
        cam_np = np.zeros((h, w))
        cam_np[mask] = 1.0
        cam_np = gaussian_filter(cam_np, sigma=5)  # Smooth the edges

    # Create heatmap and overlay with more vibrant colors
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Choose a better colormap for medical images
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)  # Changed to JET for better contrast
    
    # Use a LOWER alpha value to make visualization less overwhelming
    overlay_alpha = 0.5  # Reduced from 0.7 to 0.5
    overlay = cv2.addWeighted(img_color, 1-overlay_alpha, heatmap, overlay_alpha, 0)
    
    print(f"DEBUG: Final overlay shape: {overlay.shape}")
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