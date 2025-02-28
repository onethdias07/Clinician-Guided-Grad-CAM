import torch
import torch.nn.functional as F
import numpy as np
from .grad_cam import GradCAM

def compute_feedback_loss(gradcam_map, expert_map):
    gradcam_torch = torch.from_numpy(gradcam_map).float().unsqueeze(0).unsqueeze(0)
    expert_torch = torch.from_numpy(expert_map).float().unsqueeze(0).unsqueeze(0)
    if torch.cuda.is_available():
        gradcam_torch = gradcam_torch.cuda()
        expert_torch = expert_torch.cuda()
    gradcam_torch = torch.clamp(gradcam_torch, 1e-7, 1 - 1e-7)
    gradcam_logit = torch.logit(gradcam_torch)
    loss = F.binary_cross_entropy_with_logits(gradcam_logit, expert_torch, reduction='mean')
    return loss

def do_finetuning_step(model, input_tensor, real_label, user_mask, target_label, optimizer, lambda_feedback=5.0):
    model.train()
    optimizer.zero_grad()
    logits = model(input_tensor)
    if isinstance(logits, tuple):
        logits = logits[0]
    prob_for_label = logits[:, target_label]
    real_label_torch = torch.tensor([float(real_label)], device=prob_for_label.device)
    classification_loss = F.binary_cross_entropy_with_logits(prob_for_label, real_label_torch)
    gradcam_map = GradCAM.generate_cam(model, input_tensor, target_label)
    feedback_loss = compute_feedback_loss(gradcam_map, user_mask)
    total_loss = classification_loss + lambda_feedback * feedback_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item()
