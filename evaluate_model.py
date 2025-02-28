import os
import csv
import argparse
import math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import torchvision.transforms as T
import torchxrayvision as xrv
import torchxrayvision.utils as xrv_utils

# Import Grad-CAM (if you want to visualize as well) and feedback loss if needed
# from utils.grad_cam import GradCAM
# from utils.feedback_utils import compute_feedback_loss

# ---------------------------
# Define a Dataset Class
# ---------------------------
class XRayMaskDataset(Dataset):
    """
    A simple dataset that reads a CSV file with columns:
    xray_path, mask_path, pathology_idx, real_label
    and applies the TorchXrayvision transforms.
    """
    def __init__(self, csv_path):
        self.records = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(row)
                
        self.transform = T.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        xray_path = rec['xray_path']
        mask_path = rec['mask_path']
        pathology_idx = int(rec['pathology_idx'])
        real_label = float(rec['real_label'])

        # Load and preprocess X-ray image
        pil_img = Image.open(xray_path).convert('L')
        img_np = np.array(pil_img, dtype=np.float32)
        # Normalize to [0,1]
        img_np = xrv.datasets.normalize(img_np, 255)
        if img_np.mean() > 0.7:
            img_np = 1.0 - img_np
        img_np = img_np[None, ...]  # shape (1, H, W)
        img_np = self.transform(img_np)
        xray_tensor = torch.from_numpy(img_np)  # shape (1,224,224)

        # Load mask image (assumed to be black & white, 224x224)
        mask_pil = Image.open(mask_path).convert('L')
        mask_np = np.array(mask_pil, dtype=np.float32) / 255.0  # [0,1]

        return xray_tensor, mask_np, pathology_idx, real_label

# ---------------------------
# Evaluation Functionality
# ---------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    # For each sample, we compute the prediction for the targeted pathology.
    with torch.no_grad():
        for batch in dataloader:
            xray_tensor, mask_np, pathology_idx, real_label = batch
            # Move to device
            xray_tensor = xray_tensor.to(device, dtype=torch.float32)
            real_label = real_label.to(device, dtype=torch.float32)
            pathology_idx = pathology_idx.to(device, dtype=torch.long)
            
            logits = model(xray_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            # For each sample, extract the logit corresponding to its pathology index
            for i in range(logits.size(0)):
                idx = pathology_idx[i].item()
                logit = logits[i, idx]
                prob = torch.sigmoid(logit).item()
                pred_label = 1 if prob >= 0.5 else 0
                y_true.append(int(real_label[i].item()))
                y_pred.append(pred_label)
    
    # Compute confusion matrix and accuracy
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)
    
    return cm, acc, report

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned Model")
    parser.add_argument("--csv", type=str, default="data/annotations.csv",
                        help="Path to annotations CSV for evaluation")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_finetuned.pth",
                        help="Path to fine-tuned model checkpoint")
    args = parser.parse_args()
    
    # Create dataset and DataLoader
    dataset = XRayMaskDataset(args.csv)
    print(f"Loaded dataset with {len(dataset)} samples from {args.csv}")
    # You can also split into a test set here if needed
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Load model
    print("Loading model...")
    xrv.models.DenseNet.features2 = lambda self, x: xrv_utils.fix_resolution(x, 224, self)  # dummy patch
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    disable_inplace_relu(model)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load fine-tuned weights if they exist
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded fine-tuned model from {args.checkpoint}")
    else:
        print("Checkpoint not found. Evaluating base model.")
    
    # Evaluate model
    evaluate_model(model, dataloader, device)

if __name__ == "__main__":
    main()
