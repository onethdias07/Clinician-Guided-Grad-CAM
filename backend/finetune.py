import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from attention_model import SimpleAttentionCNN

###############################################################################
# 1. LOGGING SETUP
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Finetune")

###############################################################################
# 2. ARGUMENT PARSING
###############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Offline finetuning script for TB X-ray classification using clinician feedback with masks."
    )
    parser.add_argument("--old-model-path", type=str, default="model/tb_chest_xray_attention_best.pt",
                        help="Path to the current model weights (.pt).")
    parser.add_argument("--new-model-path", type=str, default="finetuning/tb_chest_xray_refined.pt",
                        help="Path to save the finetuned model.")
    parser.add_argument("--feedback-log", type=str, default="feedback/feedback_log.csv",
                        help="Path to feedback_log.csv containing user-corrected data.")
    parser.add_argument("--feedback-images-dir", type=str, default="feedback/images",
                        help="Directory containing feedback images.")
    parser.add_argument("--feedback-masks-dir", type=str, default="feedback/masks",
                        help="Directory containing clinician-drawn masks.")
    parser.add_argument("--original-train-dir", type=str, default="model/tuberculosis-dataset",
                        help="Directory with the original dataset (Normal/ and Tuberculosis/).")
    parser.add_argument("--include-original-data", action="store_true", 
                        help="Whether to include original training data in fine-tuning.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for finetuning.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for finetuning.")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate for optimizer.")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Fraction of data to hold out as validation.")
    parser.add_argument("--mask-loss-weight", type=float, default=0.2,
                        help="Weight for the mask-based attention loss (0-1).")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()

###############################################################################
# 3. DATASET CLASS FOR IMAGES AND MASKS
###############################################################################
class TBXrayFeedbackDataset(Dataset):
    """Dataset that loads X-ray images and clinician-drawn masks."""
    def __init__(self, filepaths, labels, mask_paths=None, transform=None, mask_transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.mask_paths = mask_paths if mask_paths is not None else [None] * len(filepaths)
        self.transform = transform
        self.mask_transform = mask_transform or T.Compose([
            T.Resize((32, 32)),  # Must match model's attention map size (was 30x30)
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image_path = self.filepaths[idx]
        label_val = self.labels[idx]
        mask_path = self.mask_paths[idx]

        # Read grayscale image
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Convert to PIL for transforms
        pil_img = Image.fromarray(img_gray, mode='L')
        
        # Apply transforms or ensure 256x256 size
        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            # Resize to 256x256 to match model's expected input size
            pil_img = pil_img.resize((256, 256), Image.BICUBIC)
            img_array = np.array(pil_img, dtype=np.float32) / 255.0
            img_tensor = torch.tensor(img_array).unsqueeze(0)  # [1,H,W]
            
        # Process mask if available
        mask_tensor = None
        if mask_path and os.path.isfile(mask_path):
            try:
                # Load mask as binary image
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    # Convert to binary (>128 = foreground, <=128 = background)
                    mask_binary = (mask_img > 128).astype(np.uint8) * 255
                    mask_pil = Image.fromarray(mask_binary, mode='L')
                    mask_tensor = self.mask_transform(mask_pil)
            except Exception as e:
                logger.warning(f"Error loading mask {mask_path}: {e}")
                mask_tensor = None

        label_tensor = torch.tensor(label_val, dtype=torch.float32)
        
        # Return image, label, and mask (None if not available)
        return img_tensor, label_tensor, mask_tensor

###############################################################################
# 4. MASK-GUIDED ATTENTION LOSS
###############################################################################
def attention_alignment_loss(attention_maps, expert_masks):
    """
    Calculate loss between model's attention and expert-drawn masks.
    
    Args:
        attention_maps: Model's attention maps (B, 1, H, W)
        expert_masks: Binary masks drawn by experts (B, 1, H, W)
        
    Returns:
        Tensor: Mean squared error between attention and normalized masks
    """
    losses = []
    batch_size = attention_maps.size(0)
    
    for i in range(batch_size):
        if expert_masks[i] is None:
            continue
            
        # Get current attention map and mask
        attn = attention_maps[i]  # (1, H, W)
        mask = expert_masks[i]    # (1, H, W)
        
        # Check if sizes match and resize if needed
        if attn.shape != mask.shape:
            # Log the mismatch
            print(f"Shape mismatch: attention {attn.shape}, mask {mask.shape}")
            
            # Resize mask to match attention map
            if hasattr(torch.nn.functional, 'interpolate'):
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0), 
                    size=attn.shape[1:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
        
        # Normalize mask to sum to 1 (like a probability distribution)
        mask_sum = mask.sum()
        if mask_sum > 0:
            mask = mask / mask_sum
            
            # MSE loss between normalized attention and mask
            loss = F.mse_loss(attn, mask)
            losses.append(loss)
    
    # Return mean loss if we have any masks, otherwise 0
    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=attention_maps.device)

###############################################################################
# 5. DATA LOADING FUNCTIONS
###############################################################################
def load_original_dataset(base_dir):
    """
    Loads data from folders:
      base_dir/Normal/   (label=0)
      base_dir/Tuberculosis/  (label=1)
    Returns:
      filepaths (list[str]), labels (list[int])
    """
    normal_dir = os.path.join(base_dir, "Normal")
    tb_dir = os.path.join(base_dir, "Tuberculosis")

    filepaths = []
    labels = []

    # Normal -> label=0
    if os.path.isdir(normal_dir):
        for fn in os.listdir(normal_dir):
            path = os.path.join(normal_dir, fn)
            if os.path.isfile(path):
                filepaths.append(path)
                labels.append(0)

    # TB -> label=1
    if os.path.isdir(tb_dir):
        for fn in os.listdir(tb_dir):
            path = os.path.join(tb_dir, fn)
            if os.path.isfile(path):
                filepaths.append(path)
                labels.append(1)

    return filepaths, labels

def load_feedback_data(feedback_csv, images_dir, masks_dir):
    """
    Loads feedback data from CSV file with columns:
      - image_filename: Image filename in images_dir
      - mask_filename: Mask filename in masks_dir
      - label: "TB" or "Normal"
      
    Returns:
      filepaths (list[str]), labels (list[int]), mask_paths (list[str or None])
    """
    # Add debug information
    logger.info(f"Debug: Looking for feedback CSV at: {feedback_csv}")
    logger.info(f"Debug: File exists? {os.path.exists(feedback_csv)}")
    logger.info(f"Debug: Images directory: {images_dir} (exists: {os.path.isdir(images_dir)})")
    logger.info(f"Debug: Masks directory: {masks_dir} (exists: {os.path.isdir(masks_dir)})")
    
    # Try alternative path resolution if file not found
    if not os.path.isfile(feedback_csv):
        try:
            from pathlib import Path
            base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            alt_feedback_csv = base_dir / 'feedback' / 'feedback_log.csv'
            logger.info(f"Debug: Trying alternative path: {alt_feedback_csv}")
            logger.info(f"Debug: Alternative path exists? {os.path.exists(alt_feedback_csv)}")
            
            if os.path.exists(alt_feedback_csv):
                feedback_csv = str(alt_feedback_csv)
                # Also update images and masks directories
                images_dir = str(base_dir / 'feedback' / 'images')
                masks_dir = str(base_dir / 'feedback' / 'masks')
                logger.info(f"Debug: Using alternative paths for feedback data")
        except Exception as e:
            logger.error(f"Debug: Error resolving alternative path: {e}")

    if not os.path.isfile(feedback_csv):
        logger.error(f"Feedback log CSV not found: {feedback_csv}")
        return [], [], []

    try:
        # Read the file contents to handle the comment line
        with open(feedback_csv, 'r') as f:
            lines = f.readlines()
            
        # Skip comment line if it starts with //
        if lines and lines[0].strip().startswith('//'):
            logger.info(f"Debug: Skipping comment line: {lines[0].strip()}")
            clean_lines = lines[1:]
        else:
            clean_lines = lines
            
        # Check if we need to add a header row
        if clean_lines and not any(line.lower().startswith('image_filename') for line in clean_lines):
            logger.info(f"Debug: Adding header row to CSV data")
            header = "image_filename,mask_filename,label,timestamp\n"
            clean_lines.insert(0, header)
        
        # Create a temporary file with clean data for pandas
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp:
            temp_path = temp.name
            temp.writelines(clean_lines)
            
        # Read the cleaned CSV
        logger.info(f"Debug: Reading cleaned CSV from: {temp_path}")
        df = pd.read_csv(temp_path)
        os.unlink(temp_path)  # Clean up temp file
        
        # Log dataframe information
        logger.info(f"Debug: CSV columns: {df.columns.tolist()}")
        logger.info(f"Debug: CSV has {len(df)} rows")
        if len(df) > 0:
            logger.info(f"Debug: First row: {df.iloc[0].to_dict()}")
        
    except Exception as e:
        logger.error(f"Failed to read feedback CSV: {e}")
        return [], [], []
    
    # Check for required columns
    img_col = None
    mask_col = None
    label_col = None
    
    # Try different possible column names
    if "image_filename" in df.columns:
        img_col = "image_filename"
    elif "image_path" in df.columns:
        img_col = "image_path"
        
    if "mask_filename" in df.columns:
        mask_col = "mask_filename"
    elif "mask_path" in df.columns:
        mask_col = "mask_path"
        
    if "label" in df.columns:
        label_col = "label"
    elif "correct_label" in df.columns:
        label_col = "correct_label"
    
    # Check if required columns exist
    if img_col is None or label_col is None:
        logger.error("Missing required columns in feedback CSV. Available columns: %s", 
                     df.columns.tolist())
        return [], [], []

    filepaths = []
    labels = []
    mask_paths = []

    for _, row in df.iterrows():
        img_rel = row[img_col]
        mask_rel = row.get(mask_col) if mask_col else None
        string_label = row[label_col]

        # Convert TB/Normal string labels to integers (1/0)
        if isinstance(string_label, str):
            if string_label.lower() == "tb":
                int_label = 1
            elif string_label.lower() == "normal":
                int_label = 0
            else:
                try:
                    int_label = int(string_label)
                except ValueError:
                    logger.warning(f"Invalid label '{string_label}', skipping")
                    continue
        else:
            try:
                int_label = int(string_label)
            except ValueError:
                logger.warning(f"Invalid label {string_label}, skipping")
                continue

        # Resolve image path
        img_abs = os.path.join(images_dir, img_rel)
        if not os.path.isfile(img_abs):
            # Try just the basename
            img_abs = os.path.join(images_dir, os.path.basename(img_rel))
            if not os.path.isfile(img_abs):
                logger.warning(f"Image not found: {img_rel}, skipping")
                continue
            else:
                logger.info(f"Debug: Found image using basename: {os.path.basename(img_rel)}")

        # Resolve mask path
        mask_abs = None
        if mask_rel:
            mask_abs = os.path.join(masks_dir, mask_rel)
            if not os.path.isfile(mask_abs):
                # Try just the basename
                mask_abs = os.path.join(masks_dir, os.path.basename(mask_rel))
                if not os.path.isfile(mask_abs):
                    logger.warning(f"Mask not found: {mask_rel}, proceeding without mask")
                    mask_abs = None
                else:
                    logger.info(f"Debug: Found mask using basename: {os.path.basename(mask_rel)}")

        filepaths.append(img_abs)
        labels.append(int_label)
        mask_paths.append(mask_abs)  # Will be None if no mask

    logger.info(f"Loaded {len(filepaths)} valid feedback images")
    logger.info(f"Found {sum(1 for m in mask_paths if m is not None)} valid masks")
    
    return filepaths, labels, mask_paths

def create_train_val_split(paths, labels, masks=None, test_split=0.2, random_seed=42):
    """
    Creates train/val split from given data.
    """
    if masks is None:
        masks = [None] * len(paths)
    
    # Convert to numpy arrays for easier indexing
    paths_array = np.array(paths)
    labels_array = np.array(labels)
    masks_array = np.array(masks, dtype=object)  # object dtype for None values

    # Shuffle with fixed seed
    np.random.seed(random_seed)
    indices = np.arange(len(paths))
    np.random.shuffle(indices)

    # Split into train/val
    split_idx = int(len(indices) * (1 - test_split))
    train_idx = indices[:split_idx]
    val_idx   = indices[split_idx:]

    # Extract train/val sets
    train_paths = paths_array[train_idx].tolist()
    train_labels = labels_array[train_idx].tolist()
    train_masks = masks_array[train_idx].tolist()
    
    val_paths = paths_array[val_idx].tolist()
    val_labels = labels_array[val_idx].tolist()
    val_masks = masks_array[val_idx].tolist()

    return train_paths, train_labels, train_masks, val_paths, val_labels, val_masks

###############################################################################
# 6. TRAINING FUNCTIONS
###############################################################################
def train_one_epoch(model, dataloader, optimizer, criterion, device, mask_loss_weight):
    """Train for one epoch with mask-guided attention loss."""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_attn_loss = 0.0
    correct = 0
    total_samples = 0
    valid_mask_samples = 0

    for batch in dataloader:
        # Unpack batch
        imgs, labels, masks = batch
        
        # Move to device
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)  # shape: [B,1]
        
        # Filter out None masks
        valid_masks = []
        for mask in masks:
            if mask is not None:
                valid_masks.append(mask.to(device))
            else:
                valid_masks.append(None)
        
        # Forward pass
        optimizer.zero_grad()
        outputs, attention_maps = model(imgs)
        
        # Classification loss
        cls_loss = criterion(outputs, labels)
        
        # Attention alignment loss with masks
        attn_loss = torch.tensor(0.0, device=device)
        if mask_loss_weight > 0 and any(mask is not None for mask in valid_masks):
            attn_loss = attention_alignment_loss(attention_maps, valid_masks)
            valid_mask_samples += sum(1 for mask in valid_masks if mask is not None)
        
        # Combined loss
        loss = cls_loss + mask_loss_weight * attn_loss
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        # Track metrics
        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        if attn_loss.item() > 0:
            total_attn_loss += attn_loss.item() * batch_size
        
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total_samples += batch_size

    # Calculate epoch metrics
    return {
        'loss': total_loss / total_samples,
        'cls_loss': total_cls_loss / total_samples, 
        'attn_loss': total_attn_loss / total_samples if total_attn_loss > 0 else 0,
        'accuracy': correct / total_samples,
        'mask_samples': valid_mask_samples
    }

def evaluate_model(model, dataloader, criterion, device, mask_loss_weight):
    """Evaluate model with mask-guided attention loss."""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_attn_loss = 0.0
    correct = 0
    total_samples = 0
    valid_mask_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch
            imgs, labels, masks = batch
            
            # Move to device
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # Filter out None masks
            valid_masks = []
            for mask in masks:
                if mask is not None:
                    valid_masks.append(mask.to(device))
                else:
                    valid_masks.append(None)
            
            # Forward pass
            outputs, attention_maps = model(imgs)
            
            # Classification loss
            cls_loss = criterion(outputs, labels)
            
            # Attention alignment loss
            attn_loss = torch.tensor(0.0, device=device)
            if mask_loss_weight > 0 and any(mask is not None for mask in valid_masks):
                attn_loss = attention_alignment_loss(attention_maps, valid_masks)
                valid_mask_samples += sum(1 for mask in valid_masks if mask is not None)
            
            # Combined loss
            loss = cls_loss + mask_loss_weight * attn_loss

            # Track metrics
            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            total_cls_loss += cls_loss.item() * batch_size
            if attn_loss.item() > 0:
                total_attn_loss += attn_loss.item() * batch_size
            
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total_samples += batch_size

    # Calculate metrics
    return {
        'loss': total_loss / total_samples,
        'cls_loss': total_cls_loss / total_samples,
        'attn_loss': total_attn_loss / total_samples if total_attn_loss > 0 else 0,
        'accuracy': correct / total_samples,
        'mask_samples': valid_mask_samples
    }
    
    

def custom_collate(batch):
    """
    Custom collate function that handles None values for masks.
    """
    # Separate elements by type
    images = []
    labels = []
    masks = []
    
    for item in batch:
        images.append(item[0])
        labels.append(item[1])
        masks.append(item[2])  # Can be None
    
    # Stack tensors
    images = torch.stack(images)
    labels = torch.stack(labels)
    
    # Don't try to stack masks - leave as list since they may contain None
    
    return images, labels, masks    

###############################################################################
# 7. MAIN FUNCTION
###############################################################################
def main():
    # Get arguments
    args = parse_arguments()
    logger.info("Finetuning with mask-guided attention using these arguments:")
    logger.info(args)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory if needed
    output_dir = os.path.dirname(args.new_model_path)
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Debug output directory status
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Directory exists: {os.path.isdir(output_dir)}")
    logger.info(f"Directory is writable: {os.access(output_dir, os.W_OK)}")
    
    # Convert to absolute path
    args.new_model_path = os.path.abspath(args.new_model_path)
    logger.info(f"Absolute path for new model: {args.new_model_path}")
    
    # Load existing model
    if not os.path.isfile(args.old_model_path):
        logger.error(f"Model not found: {args.old_model_path}")
        return
    
    model = SimpleAttentionCNN().to(device)
    model.load_state_dict(torch.load(args.old_model_path, map_location=device))
    logger.info(f"Loaded model from {args.old_model_path}")

    # Load feedback data with masks
    fb_paths, fb_labels, fb_masks = load_feedback_data(
        args.feedback_log, 
        args.feedback_images_dir, 
        args.feedback_masks_dir
    )
    
    if len(fb_paths) == 0:
        logger.error("No feedback data found. Please check the feedback CSV and directories.")
        return

    # Optionally load original dataset 
    if args.include_original_data:
        orig_paths, orig_labels = load_original_dataset(args.original_train_dir)
        logger.info(f"Including {len(orig_paths)} images from original dataset")
        
        # Combine feedback data with original data
        all_paths = orig_paths + fb_paths
        all_labels = orig_labels + fb_labels
        all_masks = [None] * len(orig_paths) + fb_masks
    else:
        # Use only feedback data
        logger.info("Using only feedback data for fine-tuning")
        all_paths = fb_paths
        all_labels = fb_labels
        all_masks = fb_masks

    # Create train/val split
    train_paths, train_labels, train_masks, val_paths, val_labels, val_masks = create_train_val_split(
        all_paths, all_labels, all_masks, 
        test_split=args.test_split,
        random_seed=args.random_seed
    )
    
    logger.info(f"Training set: {len(train_paths)} images, {sum(1 for m in train_masks if m is not None)} masks")
    logger.info(f"Validation set: {len(val_paths)} images, {sum(1 for m in val_masks if m is not None)} masks")

    # Data transformations
    train_transforms = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomAutocontrast(p=0.2),
        T.ToTensor()
    ])

    val_transforms = T.Compose([
        T.Resize((256, 256)),  # Match expected input size for the model
        T.ToTensor()
    ])

    # Create datasets
    train_dataset = TBXrayFeedbackDataset(
        train_paths, train_labels, train_masks, transform=train_transforms
    )
    
    val_dataset = TBXrayFeedbackDataset(
        val_paths, val_labels, val_masks, transform=val_transforms
    )

    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=custom_collate
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=custom_collate
    )

    # Set up training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=2, min_lr=1e-6, verbose=True
    )

    # Evaluate initial performance
    logger.info("Evaluating initial model performance...")
    val_metrics = evaluate_model(model, val_loader, criterion, device, args.mask_loss_weight)
    best_val_acc = val_metrics['accuracy']
    
    logger.info(f"Initial validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
    if val_metrics['mask_samples'] > 0:
        logger.info(f"Initial attention loss: {val_metrics['attn_loss']:.6f} on {val_metrics['mask_samples']} masks")

    # Fine-tuning loop
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.mask_loss_weight
        )
        
        # Evaluate
        val_metrics = evaluate_model(
            model, val_loader, criterion, device, args.mask_loss_weight
        )
        
        # Update learning rate
        scheduler.step(val_metrics['accuracy'])

        # Log metrics
        logger.info(f"[Epoch {epoch}/{args.epochs}] " +
                   f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | " +
                   f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Log attention loss details
        if train_metrics['mask_samples'] > 0:
            logger.info(f"    Train: Class Loss: {train_metrics['cls_loss']:.4f}, " +
                       f"Attn Loss: {train_metrics['attn_loss']:.6f} ({train_metrics['mask_samples']} masks)")
        
        if val_metrics['mask_samples'] > 0:
            logger.info(f"    Val: Class Loss: {val_metrics['cls_loss']:.4f}, " +
                      f"Attn Loss: {val_metrics['attn_loss']:.6f} ({val_metrics['mask_samples']} masks)")

        # Save if validation improved
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            try:
                # Ensure the directory exists before saving
                save_dir = os.path.dirname(args.new_model_path)
                os.makedirs(save_dir, exist_ok=True)
                
                # Save the model
                torch.save(model.state_dict(), args.new_model_path)
                
                # Verify the file was created
                if os.path.exists(args.new_model_path):
                    logger.info(f"Validation accuracy improved to {best_val_acc:.4f}. Model saved to {args.new_model_path}")
                else:
                    logger.error(f"Failed to save model: File not created at {args.new_model_path}")
            except Exception as e:
                logger.error(f"Error saving model: {e}")

    # Save final model regardless of validation improvement
    try:
        torch.save(model.state_dict(), args.new_model_path)
        logger.info(f"Final model saved to {args.new_model_path}")
    except Exception as e:
        logger.error(f"Error saving final model: {e}")

    logger.info(f"Fine-tuning complete. Best validation accuracy: {best_val_acc:.4f}")
    
    # Final verification
    if os.path.exists(args.new_model_path):
        file_size = os.path.getsize(args.new_model_path) / 1024 / 1024  # Size in MB
        logger.info(f"Verified: Model file exists at {args.new_model_path} (Size: {file_size:.2f} MB)")
    else:
        logger.error(f"ERROR: Model file not found at {args.new_model_path} after training")

if __name__ == "__main__":
    main()