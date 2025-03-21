import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import cv2
import traceback
import platform
import psutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as T

from attention_model import SimpleAttentionCNN

###############################################################################
# 1. LOGGING SETUP
###############################################################################
# Create a unique log file for each run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'finetune_{timestamp}.log')

# Configure logging to write to both console and file
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger("Finetune")

# Log system information
def log_system_info():
    logger.info("=" * 40)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 40)
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"OS: {platform.platform()}")
    logger.info(f"CPU: {platform.processor()}")
    mem = psutil.virtual_memory()
    logger.info(f"Memory: Total={mem.total/1e9:.1f}GB, Available={mem.available/1e9:.1f}GB")
    logger.info("=" * 40)

# Log memory usage
def log_memory_usage(step_name):
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        gpu_mem = "N/A"
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.memory_allocated()/1e9:.2f}GB"
        
        logger.info(f"MEMORY [{step_name}] - RAM: {mem_info.rss/1e9:.2f}GB, GPU: {gpu_mem}")
    except Exception as e:
        logger.error(f"Failed to log memory usage: {e}")

###############################################################################
# 2. ARGUMENT PARSING AND VALIDATION
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
    parser.add_argument("--balance-datasets", action="store_true", 
                        help="Whether to balance the original and feedback datasets.")
    parser.add_argument("--freeze-layers", type=int, default=9,
                        help="Number of early layers to freeze (max 13).")
    parser.add_argument("--gradual-unfreeze", action="store_true",
                        help="Whether to gradually unfreeze layers during training.")
    parser.add_argument("--initial-lr", type=float, default=1e-5,
                        help="Initial learning rate for optimizer (used with gradual unfreezing).")
    parser.add_argument("--final-lr", type=float, default=1e-4,
                        help="Final learning rate for optimizer (used with gradual unfreezing).")
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.isfile(args.old_model_path):
        logger.error(f"Model file not found: {args.old_model_path}")
        raise FileNotFoundError(f"Model file not found: {args.old_model_path}")
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.new_model_path)), exist_ok=True)
    
    # Validate freeze layers is within range
    if args.freeze_layers < 0 or args.freeze_layers > 13:
        logger.warning(f"Invalid freeze_layers value: {args.freeze_layers}. Using default of 9.")
        args.freeze_layers = 9
    
    # Log all arguments for debugging
    logger.info("Command line arguments:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    return args

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
# 5. DATA LOADING FUNCTIONS WITH ENHANCED ERROR HANDLING
###############################################################################
def load_original_dataset(base_dir):
    """
    Loads data from folders:
      base_dir/Normal/   (label=0)
      base_dir/Tuberculosis/  (label=1)
    Returns:
      filepaths (list[str]), labels (list[int])
    """
    # Path validation
    if not os.path.isdir(base_dir):
        logger.error(f"Original dataset directory not found: {base_dir}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Absolute path: {os.path.abspath(base_dir)}")
        raise FileNotFoundError(f"Original dataset directory not found: {base_dir}")
    
    normal_dir = os.path.join(base_dir, "Normal")
    tb_dir = os.path.join(base_dir, "Tuberculosis")
    
    # Subdirectories validation
    if not os.path.isdir(normal_dir):
        logger.warning(f"Normal directory not found: {normal_dir}")
    if not os.path.isdir(tb_dir):
        logger.warning(f"Tuberculosis directory not found: {tb_dir}")
    
    filepaths = []
    labels = []
    normal_count = 0
    tb_count = 0

    # Normal -> label=0
    if os.path.isdir(normal_dir):
        for fn in os.listdir(normal_dir):
            path = os.path.join(normal_dir, fn)
            if os.path.isfile(path):
                filepaths.append(path)
                labels.append(0)
                normal_count += 1

    # TB -> label=1
    if os.path.isdir(tb_dir):
        for fn in os.listdir(tb_dir):
            path = os.path.join(tb_dir, fn)
            if os.path.isfile(path):
                filepaths.append(path)
                labels.append(1)
                tb_count += 1
    
    logger.info(f"Loaded original dataset: {normal_count} Normal, {tb_count} TB images")
    
    # Validate data
    if len(filepaths) == 0:
        logger.warning("No original dataset images found!")
    
    return filepaths, labels

def load_feedback_data(feedback_csv, images_dir, masks_dir):
    """
    Loads feedback data with enhanced error handling and debug output
    """
    try:
        # Extensive path debugging
        logger.debug(f"Feedback CSV path: {feedback_csv}")
        logger.debug(f"Absolute path: {os.path.abspath(feedback_csv)}")
        logger.debug(f"CSV exists: {os.path.exists(feedback_csv)}")
        logger.debug(f"Images dir: {images_dir} (exists: {os.path.isdir(images_dir)})")
        logger.debug(f"Masks dir: {masks_dir} (exists: {os.path.isdir(masks_dir)})")
        
        # Current directory for reference
        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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
            raise FileNotFoundError(f"Feedback log CSV not found: {feedback_csv}")

        # Read and debug the CSV content
        with open(feedback_csv, 'r') as f:
            csv_content = f.read()
            logger.debug(f"CSV file size: {len(csv_content)} bytes")
            lines = csv_content.splitlines()
            logger.debug(f"CSV has {len(lines)} lines")
            if lines:
                logger.debug(f"First line: {lines[0]}")
                if len(lines) > 1:
                    logger.debug(f"Second line: {lines[1]}")
            
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
            
        # Initialize empty lists to store data
        filepaths = []
        labels = []
        mask_paths = []
        
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

        # Process each row in the dataframe
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
        
        # Validate that we found some data
        if len(filepaths) == 0:
            logger.error("No valid feedback data found after processing CSV")
            if len(df) > 0:
                # Log some example rows to help debug the issue
                logger.debug("Example rows from CSV that failed processing:")
                for i in range(min(3, len(df))):
                    logger.debug(f"Row {i}: {df.iloc[i].to_dict()}")
        else:
            logger.info(f"Successfully loaded {len(filepaths)} feedback images")
            logger.info(f"Found {sum(1 for m in mask_paths if m is not None)} valid masks")
        
        return filepaths, labels, mask_paths
        
    except Exception as e:
        logger.error(f"Error loading feedback data: {e}")
        logger.error(traceback.format_exc())
        # Return empty lists instead of failing completely
        return [], [], []

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

def balance_datasets(orig_paths, orig_labels, fb_paths, fb_labels, fb_masks, random_seed=42):
    """
    Create a balanced mix of original and feedback datasets by either:
    1. Upsampling the smaller dataset to match the larger one, or
    2. Downsampling the larger dataset to match the smaller one
    
    Returns combined and balanced paths, labels, and masks
    """
    logger.info(f"Balancing datasets - Original: {len(orig_paths)}, Feedback: {len(fb_paths)}")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # If one dataset is significantly larger, downsample it
    # Otherwise, prefer to use all feedback data and sample from original data
    if len(orig_paths) > len(fb_paths) * 20:  # Original dataset is much larger
        # Sample from original dataset to match feedback size
        orig_indices = np.random.choice(
            len(orig_paths), 
            size=len(fb_paths) * 2,  # Use 2x feedback samples from original data
            replace=False
        )
        orig_paths_balanced = [orig_paths[i] for i in orig_indices]
        orig_labels_balanced = [orig_labels[i] for i in orig_indices]
        
        logger.info(f"Downsampled original dataset from {len(orig_paths)} to {len(orig_paths_balanced)}")
        
        # Combine datasets
        all_paths = orig_paths_balanced + fb_paths
        all_labels = orig_labels_balanced + fb_labels
        all_masks = [None] * len(orig_paths_balanced) + fb_masks
        
    else:  # Feedback dataset is small or comparable in size
        # Just use all data
        all_paths = orig_paths + fb_paths
        all_labels = orig_labels + fb_labels
        all_masks = [None] * len(orig_paths) + fb_masks
    
    # Calculate class distribution for logging
    orig_positive = sum(orig_labels) / len(orig_labels) if orig_labels else 0
    fb_positive = sum(fb_labels) / len(fb_labels) if fb_labels else 0
    combined_positive = sum(all_labels) / len(all_labels) if all_labels else 0
    
    logger.info(f"Class distribution (TB positive) - Original: {orig_positive:.2f}, " +
               f"Feedback: {fb_positive:.2f}, Combined: {combined_positive:.2f}")
    
    return all_paths, all_labels, all_masks

def freeze_model_layers(model, num_layers_to_freeze):
    """
    Freeze early layers of the model:
    - feature_extractor (13 layers total)
    - spatial_attention
    - classifier
    
    Higher num_layers_to_freeze means more layers are frozen.
    """
    # Limit to maximum number of layers
    num_layers_to_freeze = min(num_layers_to_freeze, 13)
    
    if num_layers_to_freeze > 0:
        logger.info(f"Freezing first {num_layers_to_freeze} layers of feature_extractor")
        
        # Freeze early layers of feature_extractor
        for i in range(num_layers_to_freeze):
            for param in model.feature_extractor[i].parameters():
                param.requires_grad = False
    
    # Log trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}/{total_params:,} " +
              f"({100 * trainable_params / total_params:.2f}%)")
    
    return model

def unfreeze_one_layer(model, current_unfrozen):
    """
    Unfreeze the next layer in feature_extractor
    Returns the new number of unfrozen layers
    """
    if current_unfrozen < 13:  # Only feature_extractor has 13 layers
        # Unfreeze the next layer
        layer_to_unfreeze = 13 - current_unfrozen - 1
        
        for param in model.feature_extractor[layer_to_unfreeze].parameters():
            param.requires_grad = True
            
        logger.info(f"Unfroze layer {layer_to_unfreeze} of feature_extractor")
        return current_unfrozen + 1
    
    return current_unfrozen

###############################################################################
# 7. MAIN FUNCTION WITH ENHANCED ERROR HANDLING
###############################################################################
def main():
    try:
        # Log system information
        log_system_info()
        log_memory_usage("startup")
        
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
        
        # Load existing model with error handling
        if not os.path.isfile(args.old_model_path):
            logger.error(f"Model not found: {args.old_model_path}")
            return
        
        try:
            model = SimpleAttentionCNN().to(device)
            model.load_state_dict(torch.load(args.old_model_path, map_location=device))
            logger.info(f"Loaded model from {args.old_model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            return
        
        log_memory_usage("model_loaded")

        # Load feedback data with masks - wrapped in try/except
        try:
            fb_paths, fb_labels, fb_masks = load_feedback_data(
                args.feedback_log, 
                args.feedback_images_dir, 
                args.feedback_masks_dir
            )
            
            if len(fb_paths) == 0:
                logger.error("No feedback data found. Please check the feedback CSV and directories.")
                return
                
            log_memory_usage("feedback_data_loaded")
        except Exception as e:
            logger.error(f"Failed to load feedback data: {e}")
            logger.error(traceback.format_exc())
            return

        # Load original dataset, with error handling
        orig_paths, orig_labels = [], []
        if args.include_original_data:
            try:
                orig_paths, orig_labels = load_original_dataset(args.original_train_dir)
                logger.info(f"Loaded {len(orig_paths)} images from original dataset")
                log_memory_usage("original_data_loaded")
            except Exception as e:
                logger.error(f"Failed to load original dataset: {e}")
                logger.error(traceback.format_exc())
                logger.warning("Proceeding with only feedback data")
        
        # Dataset preparation with validation at each step
        try:
            # Verify we have some data to work with
            if len(fb_paths) == 0 and len(orig_paths) == 0:
                logger.error("No data available for training")
                return
                
            # Apply dataset balancing if requested
            if args.include_original_data and args.balance_datasets:
                all_paths, all_labels, all_masks = balance_datasets(
                    orig_paths, orig_labels, fb_paths, fb_labels, fb_masks, 
                    random_seed=args.random_seed
                )
            else:
                # Use datasets based on what's available
                if args.include_original_data and len(orig_paths) > 0:
                    logger.info(f"Using all available data without balancing")
                    all_paths = orig_paths + fb_paths
                    all_labels = orig_labels + fb_labels
                    all_masks = [None] * len(orig_paths) + fb_masks
                else:
                    # Use only feedback data
                    logger.info("Using only feedback data for fine-tuning")
                    all_paths = fb_paths
                    all_labels = fb_labels
                    all_masks = fb_masks
                    
            # Verify we have data after dataset preparation
            if len(all_paths) == 0:
                logger.error("No data available after dataset preparation")
                return
                
            logger.info(f"Final dataset size: {len(all_paths)} images")
            logger.debug(f"Sample paths: {all_paths[:min(3, len(all_paths))]}")
        except Exception as e:
            logger.error(f"Failed during dataset preparation: {e}")
            logger.error(traceback.format_exc())
            return

        # Create train/val split
        try:
            train_paths, train_labels, train_masks, val_paths, val_labels, val_masks = create_train_val_split(
                all_paths, all_labels, all_masks, 
                test_split=args.test_split,
                random_seed=args.random_seed
            )
            
            logger.info(f"Training set: {len(train_paths)} images, {sum(1 for m in train_masks if m is not None)} masks")
            logger.info(f"Validation set: {len(val_paths)} images, {sum(1 for m in val_masks if m is not None)} masks")
            
            # Validate paths exist
            for i, path in enumerate(train_paths[:min(5, len(train_paths))]):
                if not os.path.exists(path):
                    logger.warning(f"Training path {i} does not exist: {path}")
            
            log_memory_usage("data_split_complete")
        except Exception as e:
            logger.error(f"Failed to create train/val split: {e}")
            logger.error(traceback.format_exc())
            return

        # Create DataLoader with error handling
        try:
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
            
            # Validate dataloaders by fetching one batch
            try:
                logger.debug("Testing train dataloader with one batch")
                sample_batch = next(iter(train_loader))
                logger.debug(f"Train batch shapes: {sample_batch[0].shape}, {sample_batch[1].shape}")
                
                logger.debug("Testing val dataloader with one batch")
                sample_batch = next(iter(val_loader))
                logger.debug(f"Val batch shapes: {sample_batch[0].shape}, {sample_batch[1].shape}")
            except Exception as e:
                logger.error(f"Failed to fetch a sample batch: {e}")
                logger.error(traceback.format_exc())
                
            log_memory_usage("dataloaders_created")
        except Exception as e:
            logger.error(f"Failed to create datasets/dataloaders: {e}")
            logger.error(traceback.format_exc())
            return

        # Model preparation
        try:
            # Freeze early layers of the model
            model = freeze_model_layers(model, args.freeze_layers)
            
            # Set up training with lower initial learning rate for gradual unfreezing
            criterion = nn.BCELoss()
            
            if args.gradual_unfreeze:
                # Start with a lower learning rate for gradual unfreezing
                optimizer = optim.Adam(
                    [p for p in model.parameters() if p.requires_grad], 
                    lr=args.initial_lr
                )
            else:
                optimizer = optim.Adam(
                    [p for p in model.parameters() if p.requires_grad], 
                    lr=args.learning_rate
                )
                
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.1, patience=2, min_lr=1e-6, verbose=True
            )
        except Exception as e:
            logger.error(f"Failed during model preparation: {e}")
            logger.error(traceback.format_exc())
            return

        # Initial evaluation
        try:
            logger.info("Evaluating initial model performance...")
            val_metrics = evaluate_model(model, val_loader, criterion, device, args.mask_loss_weight)
            best_val_acc = val_metrics['accuracy']
            
            logger.info(f"Initial validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
            if val_metrics['mask_samples'] > 0:
                logger.info(f"Initial attention loss: {val_metrics['attn_loss']:.6f} on {val_metrics['mask_samples']} masks")
        except Exception as e:
            logger.error(f"Failed during initial evaluation: {e}")
            logger.error(traceback.format_exc())
            return
            
        log_memory_usage("before_training")

        # Tracking for gradual unfreezing
        unfrozen_layers = 13 - args.freeze_layers  # Start with initial unfrozen count
        epochs_per_unfreeze = max(1, args.epochs // args.freeze_layers) if args.gradual_unfreeze else 0
        
        # Fine-tuning loop with error handling for each epoch
        for epoch in range(1, args.epochs + 1):
            try:
                logger.info(f"Starting epoch {epoch}/{args.epochs}")
                
                # Gradual unfreezing
                if args.gradual_unfreeze and epoch % epochs_per_unfreeze == 0 and unfrozen_layers < 13:
                    unfrozen_layers = unfreeze_one_layer(model, unfrozen_layers)
                    
                    # Adjust learning rate when unfreezing layers
                    if unfrozen_layers > 13 - args.freeze_layers:  # If we've unfrozen at least one layer
                        lr_increment = (args.final_lr - args.initial_lr) / args.freeze_layers
                        new_lr = args.initial_lr + lr_increment * (unfrozen_layers - (13 - args.freeze_layers))
                        
                        # Update learning rate
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        logger.info(f"Adjusting learning rate to {new_lr:.6f}")
                
                # Train
                try:
                    train_metrics = train_one_epoch(
                        model, train_loader, optimizer, criterion, device, args.mask_loss_weight
                    )
                except Exception as e:
                    logger.error(f"Error during training epoch {epoch}: {e}")
                    logger.error(traceback.format_exc())
                    break
                
                # Evaluate
                try:
                    val_metrics = evaluate_model(
                        model, val_loader, criterion, device, args.mask_loss_weight
                    )
                except Exception as e:
                    logger.error(f"Error during evaluation for epoch {epoch}: {e}")
                    logger.error(traceback.format_exc())
                    break
                
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
                        logger.error(traceback.format_exc())
                        
                # Log memory after each epoch
                log_memory_usage(f"epoch_{epoch}")
                
            except Exception as e:
                logger.error(f"Unexpected error during epoch {epoch}: {e}")
                logger.error(traceback.format_exc())
                break

        # Save final model regardless of validation improvement
        try:
            torch.save(model.state_dict(), args.new_model_path)
            logger.info(f"Final model saved to {args.new_model_path}")
        except Exception as e:
            logger.error(f"Error saving final model: {e}")
            logger.error(traceback.format_exc())

        logger.info(f"Fine-tuning complete. Best validation accuracy: {best_val_acc:.4f}")
        
        # Final verification
        if os.path.exists(args.new_model_path):
            file_size = os.path.getsize(args.new_model_path) / 1024 / 1024  # Size in MB
            logger.info(f"Verified: Model file exists at {args.new_model_path} (Size: {file_size:.2f} MB)")
        else:
            logger.error(f"ERROR: Model file not found at {args.new_model_path} after training")
            
        log_memory_usage("training_complete")
        
    except Exception as e:
        logger.error(f"Uncaught exception in main function: {e}")
        logger.error(traceback.format_exc())
        raise  # Re-raise to ensure non-zero exit code

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)  # Exit with error code