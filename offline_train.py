import os
import yaml
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose
from PIL import Image
import torchxrayvision as xrv
import torchxrayvision.utils as xrv_utils
from torch.utils.tensorboard import SummaryWriter

from utils.grad_cam import GradCAM
from utils.feedback_utils import compute_feedback_loss

def patched_features2(self, x):
    x = xrv_utils.fix_resolution(x, 224, self)
    xrv_utils.warn_normalization(x)
    features = self.features(x)
    out = F.relu(features, inplace=False)
    out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
    return out

def disable_inplace_relu(model):
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

def load_xrv_densenet(weights="densenet121-res224-all"):
    xrv.models.DenseNet.features2 = patched_features2
    model = xrv.models.DenseNet(weights=weights)
    disable_inplace_relu(model)
    return model

class XRayDataset(Dataset):
    """
    Loads CSV rows with columns:
      xray_path, mask_path, pathology_idx, real_label
    Applies TorchXRayVision transforms. 
    mask is stored as a NumPy array in [H,W].
    """
    def __init__(self, csv_file, transform=None, invert_threshold=0.7):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.invert_threshold = invert_threshold

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xray_path = row["xray_path"]
        mask_path = row["mask_path"]
        pathology_idx = int(row["pathology_idx"])
        real_label = int(row["real_label"])

        # Load X-ray
        pil_img = Image.open(xray_path).convert("L")
        img_np = np.array(pil_img, dtype=np.float32)
        img_np = xrv.datasets.normalize(img_np, 255)
        if img_np.mean() > self.invert_threshold:
            img_np = 1.0 - img_np
        img_np = img_np[None, ...]
        if self.transform:
            img_np = self.transform(img_np)
        xray_tensor = torch.from_numpy(img_np)

        # Load mask
        pil_mask = Image.open(mask_path).convert("L")
        mask_np = np.array(pil_mask, dtype=np.float32) / 255.0

        return {
            "xray": xray_tensor,        # shape (1,224,224), Torch tensor
            "mask": mask_np,            # shape (H,W), NumPy array
            "pathology_idx": pathology_idx,
            "real_label": real_label,
            "xray_path": xray_path,
            "mask_path": mask_path
        }

def xrv_transform_pipeline():
    # Standard TorchXRayVision transforms
    return Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])

def compute_batch_loss(model, batch, device, lambda_feedback=5.0):
    """
    We do a forward pass, then for each sample we call Grad-CAM => backward.
    That means we must have gradient tracking on. So do NOT wrap this in no_grad.
    """
    xray_tensor = batch["xray"].to(device, dtype=torch.float32)
    pathology_idxs = batch["pathology_idx"].to(device, dtype=torch.long)
    real_labels = batch["real_label"].to(device, dtype=torch.float32)

    # Convert 'mask' from NumPy to torch. But we want it as np again for 'compute_feedback_loss'.
    # We'll do that below for each sample.
    masks_batch_np = batch["mask"]  # this is a list or a collated Torch Tensor if your DataLoader's default collate

    logits = model(xray_tensor)
    if isinstance(logits, tuple):
        logits = logits[0]

    chosen_logits = []
    for i in range(logits.size(0)):
        chosen_logits.append(logits[i, pathology_idxs[i]])
    chosen_logits = torch.stack(chosen_logits, dim=0)

    classification_loss = F.binary_cross_entropy_with_logits(chosen_logits, real_labels)

    feedback_losses = []
    for i in range(xray_tensor.size(0)):
        single_input = xray_tensor[i].unsqueeze(0)
        single_label_idx = pathology_idxs[i].item()

        # 'masks_batch_np[i]' might be a NumPy array or a Torch tensor, depending on your DataLoader collate.
        # If it's a Torch tensor, convert it back to np:
        single_mask = masks_batch_np[i]
        if isinstance(single_mask, torch.Tensor):
            single_mask = single_mask.cpu().numpy()

        cam_map = GradCAM.generate_cam(model, single_input, single_label_idx)
        fl = compute_feedback_loss(cam_map, single_mask)
        feedback_losses.append(fl.item())

    feedback_loss = np.mean(feedback_losses)
    total_loss = classification_loss + lambda_feedback * feedback_loss
    return total_loss, classification_loss.item(), feedback_loss

def train_one_epoch(model, dataloader, optimizer, device, lambda_feedback=5.0):
    """
    Training loop (with gradients).
    """
    model.train()
    total_loss, total_cls_loss, total_fb_loss = 0.0, 0.0, 0.0
    count = 0

    for batch in dataloader:
        optimizer.zero_grad()
        loss, cls_loss_val, fb_loss_val = compute_batch_loss(model, batch, device, lambda_feedback)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_cls_loss += cls_loss_val
        total_fb_loss += fb_loss_val
        count += 1

    if count == 0:
        return 0.0, 0.0, 0.0
    return (total_loss / count, total_cls_loss / count, total_fb_loss / count)

def validate_one_epoch(model, dataloader, device, lambda_feedback=5.0):
    """
    Validation loop. We STILL need gradient tracking so Grad-CAM can do backward.
    So we DO NOT wrap in torch.no_grad().
    We'll only call model.eval() so BN/Dropout layers act in inference mode,
    but we won't disable gradient creation because Grad-CAM needs it.
    """
    model.eval()
    total_loss, total_cls_loss, total_fb_loss = 0.0, 0.0, 0.0
    count = 0

    for batch in dataloader:
        # No optimizer step, but we STILL need grad for Grad-CAM
        loss, cls_loss_val, fb_loss_val = compute_batch_loss(model, batch, device, lambda_feedback)
        total_loss += loss.item()
        total_cls_loss += cls_loss_val
        total_fb_loss += fb_loss_val
        count += 1

    if count == 0:
        return 0.0, 0.0, 0.0
    return (total_loss / count, total_cls_loss / count, total_fb_loss / count)

def main(config_path="config.yaml"):
    import argparse
    from torch.utils.tensorboard import SummaryWriter

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    lr = float(config["training"]["learning_rate"])
    lambda_feedback = float(config["training"]["lambda_feedback"])
    output_dir = config["training"]["output_dir"]
    train_split = float(config["data"]["train_split"])
    csv_file = config["data"]["annotations_csv"]

    weights_name = config["model"]["weights"]
    log_dir = config["logging"].get("log_dir", "runs/offline_train")
    use_tensorboard = config["logging"].get("use_tensorboard", True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading dataset from: {csv_file}")
    dataset = XRayDataset(csv_file=csv_file, transform=xrv_transform_pipeline())

    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"[INFO] Loading DenseNet weights: {weights_name}")
    model = load_xrv_densenet(weights=weights_name)
    model.to(device)
    model.train()  # We'll set eval() only inside the validate function

    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir) if use_tensorboard else None

    best_val_loss = math.inf
    best_model_path = os.path.join(output_dir, "model_finetuned.pth")

    for epoch in range(1, epochs + 1):
        train_loss, train_cls, train_fb = train_one_epoch(
            model, train_loader, optimizer, device, lambda_feedback
        )
        val_loss, val_cls, val_fb = validate_one_epoch(
            model, val_loader, device, lambda_feedback
        )

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss={train_loss:.4f} (cls={train_cls:.4f}, fb={train_fb:.4f}) | "
              f"Val Loss={val_loss:.4f} (cls={val_cls:.4f}, fb={val_fb:.4f})")

        if writer:
            writer.add_scalar("Train/Total_Loss", train_loss, epoch)
            writer.add_scalar("Train/Classification_Loss", train_cls, epoch)
            writer.add_scalar("Train/Feedback_Loss", train_fb, epoch)

            writer.add_scalar("Val/Total_Loss", val_loss, epoch)
            writer.add_scalar("Val/Classification_Loss", val_cls, epoch)
            writer.add_scalar("Val/Feedback_Loss", val_fb, epoch)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  [*] Best Val Loss improved to {val_loss:.4f}, "
                  f"model saved => {best_model_path}")

    if writer:
        writer.close()

    print(f"Finished training. Best model at {best_model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline Training for Clinician-Guided Grad-CAM")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    main(config_path=args.config)
