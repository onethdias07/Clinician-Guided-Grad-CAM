import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    @staticmethod
    def generate_cam(model, input_tensor, target_label):
        conv_outputs = []
        gradients = []
        target_layer = model.features.denseblock4
        def forward_hook(module, inp, out):
            conv_outputs.append(out)
        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_backward_hook(backward_hook)
        logits = model(input_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        score = logits[0, target_label]
        model.zero_grad()
        score.backward(retain_graph=True)
        feature_maps = conv_outputs[0]
        grads = gradients[0]
        alpha = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = (feature_maps * alpha).sum(dim=1).squeeze(0)
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam_4d = cam.unsqueeze(0).unsqueeze(0)
        upsampled_cam = F.interpolate(
            cam_4d,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        upsampled_cam = F.relu(upsampled_cam)
        if upsampled_cam.max() > 0:
            upsampled_cam /= upsampled_cam.max()
        fwd_handle.remove()
        bwd_handle.remove()
        return upsampled_cam.detach().cpu().numpy()

    @staticmethod
    def generate_cam_smooth(model, input_tensor, target_label, num_runs=5, threshold=0.05):
        cams = []
        for _ in range(num_runs):
            cam_np = GradCAM.generate_cam(model, input_tensor, target_label)
            cams.append(cam_np)
        avg_cam = np.mean(cams, axis=0)
        avg_cam = np.clip(avg_cam, 0, 1)
        if avg_cam.max() > 0:
            avg_cam /= avg_cam.max()
        if threshold > 0:
            avg_cam[avg_cam < threshold] = 0.0
        if avg_cam.max() > 0:
            avg_cam /= avg_cam.max()
        return avg_cam

def upsample_cam(cam_np, target_size=(224, 224)):
    cam_resized = cv2.resize(cam_np, target_size, interpolation=cv2.INTER_LINEAR)
    cam_resized = np.clip(cam_resized, 0, 1)
    if cam_resized.max() > 0:
        cam_resized /= cam_resized.max()
    return cam_resized

def overlay_cam_on_image(image_np, cam_np, alpha=0.5):
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    if image_np.dtype != np.uint8:
        image_np = (255 * np.clip(image_np, 0, 1)).astype(np.uint8)
    heatmap = cv2.applyColorMap((cam_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, alpha, heatmap, 1 - alpha, 0)
    return overlay
