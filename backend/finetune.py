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

# importing the forward pass for the CNN
from attention_model import SimpleAttentionCNN


# Configure logging with immediate flush
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.__stdout__)
    ]
)
logger = logging.getLogger("Finetune")

# Force immediate flushing of stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
else:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)


# this is the function which will parse the arguments from the command line or in this case app.py
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Offline finetuning script for TB X-ray classification using clinician feedback with masks."
    )
    parser.add_argument(
        "--old-model-path", 
        type=str, 
        default="model/tb_chest_xray_attention_best.pt",
        help="Path to the current model weights (.pt)."
    )
    parser.add_argument(
        "--new-model-path", 
        type=str, 
        default="finetuning/tb_chest_xray_refined.pt",
        help="Path to save the finetuned model."
    )
    parser.add_argument(
        "--feedback-log", 
        type=str, 
        default="feedback/feedback_log.csv",
        help="Path to feedback_log.csv containing user-corrected data."
    )
    parser.add_argument(
        "--feedback-images-dir", 
        type=str, 
        default="feedback/images",
        help="Directory containing feedback images."
    )
    parser.add_argument(
        "--feedback-masks-dir", 
        type=str, 
        default="feedback/masks",
        help="Directory containing clinician-drawn masks."
    )
    parser.add_argument(
        "--original-train-dir", 
        type=str, 
        default="model/tuberculosis-dataset",
        help="Directory with the original dataset (Normal/ and Tuberculosis/)."
    )
    parser.add_argument(
        "--include-original-data", 
        action="store_true", 
        help="Whether to include original training data in fine-tuning."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Number of epochs for finetuning."
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16,
        help="Batch size for finetuning."
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=1e-4,
        help="Learning rate for optimizer."
    )
    parser.add_argument(
        "--test-split", 
        type=float, 
        default=0.2,
        help="Fraction of data to hold out as validation."
    )
    parser.add_argument(
        "--mask-loss-weight", 
        type=float, 
        default=0.2,
        help="Weight for the mask-based attention loss (0-1)."
    )
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()

# this is the dataset class which will load the images and masks
class TBXrayFeedbackDataset(Dataset):
    def __init__(self, filepaths, labels, mask_paths=None, transform=None, mask_transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.mask_paths = mask_paths if mask_paths is not None else [None] * len(filepaths)
        self.transform = transform
        self.mask_transform = mask_transform or T.Compose([
            T.Resize((32, 32)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image_path = self.filepaths[idx]
        label_val = self.labels[idx]
        mask_path = self.mask_paths[idx]

        # Load the X-ray image
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        pil_img = Image.fromarray(img_gray, mode='L')
        
        # Apply transformations if specified
        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            pil_img = pil_img.resize((256, 256), Image.BICUBIC)
            img_array = np.array(pil_img, dtype=np.float32) / 255.0
            img_tensor = torch.tensor(img_array).unsqueeze(0)
        
        # Load clinician annotation mask if available
        mask_tensor = None
        if mask_path and os.path.isfile(mask_path):
            try:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    # Convert to binary mask (white regions are areas of interest)
                    mask_binary = (mask_img > 128).astype(np.uint8) * 255
                    mask_pil = Image.fromarray(mask_binary, mode='L')
                    mask_tensor = self.mask_transform(mask_pil)
            except Exception as e:
                logger.warning(f"Error loading mask {mask_path}: {e}")
                mask_tensor = None

        label_tensor = torch.tensor(label_val, dtype=torch.float32)
        
        return img_tensor, label_tensor, mask_tensor

# this is the attention loss function which will be used to calculate the loss between the attention maps and the expert masks
def attention_alignment_loss(attention_maps, expert_masks):
    losses = []
    batch_size = attention_maps.size(0)
    
    for i in range(batch_size):
        if expert_masks[i] is None:
            continue
            
        attn = attention_maps[i]
        mask = expert_masks[i]
        
        # this Handles size mismatches by resizing mask to match attention map -> crucial for smooth transmision from masks to attention maps
        if attn.shape != mask.shape:
            print(f"Shape mismatch: attention {attn.shape}, mask {mask.shape}")
            
            if hasattr(torch.nn.functional, 'interpolate'):
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0), 
                    size=attn.shape[1:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
        
        # Normalize mask to sum to 1 if it contains any non-zero values
        mask_sum = mask.sum()
        if mask_sum > 0:
            mask = mask / mask_sum
            
            loss = F.mse_loss(attn, mask)
            losses.append(loss)
    
    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=attention_maps.device)

# this function loads the original dataset al the way from model/tuberculosis-dataset
def load_original_dataset(base_dir):
    normal_dir = os.path.join(base_dir, "Normal")
    tb_dir = os.path.join(base_dir, "Tuberculosis")

    filepaths = []
    labels = []

    if os.path.isdir(normal_dir):
        for fn in os.listdir(normal_dir):
            path = os.path.join(normal_dir, fn)
            if os.path.isfile(path):
                filepaths.append(path)
                labels.append(0)

    if os.path.isdir(tb_dir):
        for fn in os.listdir(tb_dir):
            path = os.path.join(tb_dir, fn)
            if os.path.isfile(path):
                filepaths.append(path)
                labels.append(1)

    return filepaths, labels

# this function will load the feedbaclk data gathered earlier
def load_feedback_data(feedback_csv, images_dir, masks_dir):
    if not os.path.isfile(feedback_csv):
        try:
            from pathlib import Path
            base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            alt_feedback_csv = base_dir / 'feedback' / 'feedback_log.csv'
            
            if os.path.exists(alt_feedback_csv):
                feedback_csv = str(alt_feedback_csv)
                images_dir = str(base_dir / 'feedback' / 'images')
                masks_dir = str(base_dir / 'feedback' / 'masks')
        except Exception:
            return [], [], []

    if not os.path.isfile(feedback_csv):
        return [], [], []

    try:
        df = prepare_feedback_csv(feedback_csv)
        if df.empty:
            return [], [], []
        
        # Map column names to expected columns
        column_mappings = {
            'image_filename': ['image_filename', 'image_path'],
            'mask_filename': ['mask_filename', 'mask_path'],
            'label': ['label', 'correct_label']
        }
        
        # Find the actual column names in the dataframe
        cols = {}
        for target, possible_names in column_mappings.items():
            cols[target] = next((col for col in possible_names if col in df.columns), None)
        
        # Check if required columns exist
        if cols['image_filename'] is None or cols['label'] is None:
            return [], [], []

        # Process each row of labels, mask paths and image paths
        filepaths, labels, mask_paths = [], [], []
        for _, row in df.iterrows():
            # this is processing image paths
            img_rel = row[cols['image_filename']]
            img_abs = resolve_path(img_rel, images_dir)
            if not img_abs:
                continue
                
            # this is processing label values
            label_val = row[cols['label']]
            int_label = convert_label_to_int(label_val)
            if int_label is None:
                continue
                
            # this is process mask path if available (if combining original dataset then there is no masks)
            mask_abs = None
            if cols['mask_filename']:
                mask_rel = row[cols['mask_filename']]
                if mask_rel:
                    mask_abs = resolve_path(mask_rel, masks_dir)
                    if not mask_abs:
                        continue

            # Add to dataset
            filepaths.append(img_abs)
            labels.append(int_label)
            mask_paths.append(mask_abs)

        return filepaths, labels, mask_paths
        
    except Exception as e:
        return [], [], []
# this cleans the existing CSV for feedback
def prepare_feedback_csv(csv_path):
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            
        # Skip comment lines
        clean_lines = lines[1:] if lines and lines[0].strip().startswith('//') else lines
            
        # Add header if missing
        if clean_lines and not any(line.lower().startswith('image_filename') for line in clean_lines):
            header = "image_filename,mask_filename,label,timestamp\n"
            clean_lines.insert(0, header)
        
        # Write cleaned CSV to temp file and read with pandas
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp:
            temp_path = temp.name
            temp.writelines(clean_lines)
            
        df = pd.read_csv(temp_path)
        os.unlink(temp_path)
        
        return df
        
    except Exception as e:
        return pd.DataFrame()

# to ensure all images, masks and lables are loaded
def resolve_path(rel_path, base_dir):
    # Try direct path
    abs_path = os.path.join(base_dir, rel_path)
    if os.path.isfile(abs_path):
        return abs_path
        
    # Try using just the basename
    basename_path = os.path.join(base_dir, os.path.basename(rel_path))
    if os.path.isfile(basename_path):
        return basename_path
        
    return None

# this function will convert tb or not to 1 or 0
def convert_label_to_int(label_val):
    if isinstance(label_val, str):
        if label_val.lower() == "tb":
            return 1
        elif label_val.lower() == "normal":
            return 0
        else:
            try:
                return int(label_val)
            except ValueError:
                return None
    else:
        try:
            return int(label_val)
        except (ValueError, TypeError):
            return None


# data is split just like the original dataset (80% to train and 20% to test)
def create_train_val_split(paths, labels, masks=None, test_split=0.2, random_seed=42):
    if masks is None:
        masks = [None] * len(paths)
    
    paths_array = np.array(paths)
    labels_array = np.array(labels)
    masks_array = np.array(masks, dtype=object)

    np.random.seed(random_seed)
    indices = np.arange(len(paths))
    np.random.shuffle(indices)

    split_idx = int(len(indices) * (1 - test_split))
    train_idx = indices[:split_idx]
    val_idx   = indices[split_idx:]

    train_paths = paths_array[train_idx].tolist()
    train_labels = labels_array[train_idx].tolist()
    train_masks = masks_array[train_idx].tolist()
    
    val_paths = paths_array[val_idx].tolist()
    val_labels = labels_array[val_idx].tolist()
    val_masks = masks_array[val_idx].tolist()

    return train_paths, train_labels, train_masks, val_paths, val_labels, val_masks

# main function which will run the epcohs and training
def train_one_epoch(model, dataloader, optimizer, criterion, device, mask_loss_weight, epoch=None, total_epochs=None):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_attn_loss = 0.0
    correct = 0
    total_samples = 0
    valid_mask_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        imgs, labels, masks = batch
        
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        valid_masks = []
        for mask in masks:
            if mask is not None:
                valid_masks.append(mask.to(device))
            else:
                valid_masks.append(None)
        
        optimizer.zero_grad()
        outputs, attention_maps = model(imgs)
        # this is the classification loss 
        cls_loss = criterion(outputs, labels)
        
        # this is the attention loss (expert masks)
        attn_loss = torch.tensor(0.0, device=device)
        if mask_loss_weight > 0 and any(mask is not None for mask in valid_masks):
            attn_loss = attention_alignment_loss(attention_maps, valid_masks)
            valid_mask_samples += sum(1 for mask in valid_masks if mask is not None)
        
        # this is the total loss (classification + attention) -> experiemnt with it
        loss = cls_loss + mask_loss_weight * attn_loss
        
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        if attn_loss.item() > 0:
            total_attn_loss += attn_loss.item() * batch_size
        
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total_samples += batch_size

        # Log progress every few batches
        if batch_idx % 5 == 0:
            progress = (batch_idx + 1) / len(dataloader) * 100
            logger.info(f"[Batch Progress] {progress:.1f}%")
            sys.stdout.flush()

    metrics = {
        'loss': total_loss / total_samples,
        'cls_loss': total_cls_loss / total_samples, 
        'attn_loss': total_attn_loss / total_samples if total_attn_loss > 0 else 0,
        'accuracy': correct / total_samples,
        'mask_samples': valid_mask_samples
    }
    
    # Log epoch metrics in a standardized format
    if epoch is not None and total_epochs is not None:
        logger.info(f"[Epoch {epoch}/{total_epochs}] Train Loss: {metrics['loss']:.4f} Acc: {metrics['accuracy']:.4f}")
    sys.stdout.flush()
    
    return metrics

# this is be used later to evaluate the model when in traning
def evaluate_model(model, dataloader, criterion, device, mask_loss_weight):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_attn_loss = 0.0
    correct = 0
    total_samples = 0
    valid_mask_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            imgs, labels, masks = batch
            
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            valid_masks = []
            for mask in masks:
                if mask is not None:
                    valid_masks.append(mask.to(device))
                else:
                    valid_masks.append(None)
            
            outputs, attention_maps = model(imgs)
            
            cls_loss = criterion(outputs, labels)
            
            attn_loss = torch.tensor(0.0, device=device)
            if mask_loss_weight > 0 and any(mask is not None for mask in valid_masks):
                attn_loss = attention_alignment_loss(attention_maps, valid_masks)
                valid_mask_samples += sum(1 for mask in valid_masks if mask is not None)
            
            loss = cls_loss + mask_loss_weight * attn_loss

            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            total_cls_loss += cls_loss.item() * batch_size
            if attn_loss.item() > 0:
                total_attn_loss += attn_loss.item() * batch_size
            
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total_samples += batch_size

    return {
        'loss': total_loss / total_samples,
        'cls_loss': total_cls_loss / total_samples,
        'attn_loss': total_attn_loss / total_samples if total_attn_loss > 0 else 0,
        'accuracy': correct / total_samples,
        'mask_samples': valid_mask_samples
    }
    
# this is when combining the original dataset with the feedback data 
# cus the original data does not have expert drawn masks
def custom_collate(batch):
    images = []
    labels = []
    masks = []
    
    for item in batch:
        images.append(item[0])
        labels.append(item[1])
        masks.append(item[2])
    
    images = torch.stack(images)
    labels = torch.stack(labels)
    
    return images, labels, masks    

# main function that will be run in app.py
def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = os.path.dirname(args.new_model_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    args.new_model_path = os.path.abspath(args.new_model_path)
    
    if not os.path.isfile(args.old_model_path):
        return
    
    model = SimpleAttentionCNN().to(device)
    model.load_state_dict(torch.load(args.old_model_path, map_location=device))

    fb_paths, fb_labels, fb_masks = load_feedback_data(
        args.feedback_log, 
        args.feedback_images_dir, 
        args.feedback_masks_dir
    )
    
    if len(fb_paths) == 0:
        return

    if args.include_original_data:
        orig_paths, orig_labels = load_original_dataset(args.original_train_dir)
        
        all_paths = orig_paths + fb_paths
        all_labels = orig_labels + fb_labels
        all_masks = [None] * len(orig_paths) + fb_masks
    else:
        all_paths = fb_paths
        all_labels = fb_labels
        all_masks = fb_masks

    train_paths, train_labels, train_masks, val_paths, val_labels, val_masks = create_train_val_split(
        all_paths, all_labels, all_masks, 
        test_split=args.test_split,
        random_seed=args.random_seed
    )

    # this is the data augmentation and transformation for the training and validation datasets
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
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    train_dataset = TBXrayFeedbackDataset(
        train_paths, train_labels, train_masks, transform=train_transforms
    )
    
    val_dataset = TBXrayFeedbackDataset(
        val_paths, val_labels, val_masks, transform=val_transforms
    )

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

    # this is the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # this is the learning rate scheduler which will reduce the learning rate if the accuracy does not improve
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=2, min_lr=1e-6, verbose=True
    )

    val_metrics = evaluate_model(model, val_loader, criterion, device, args.mask_loss_weight)
    best_val_acc = val_metrics['accuracy']

    for epoch in range(1, args.epochs + 1):
        logger.info(f"[Epoch {epoch}/{args.epochs}] Starting...")
        sys.stdout.flush()
        
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.mask_loss_weight,
            epoch=epoch, total_epochs=args.epochs  # Pass epoch info
        )
        
        val_metrics = evaluate_model(
            model, val_loader, criterion, device, args.mask_loss_weight
        )
        
        logger.info(f"[Epoch {epoch}/{args.epochs}] Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f}")
        sys.stdout.flush()
        
        scheduler.step(val_metrics['accuracy'])

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            logger.info(f"[Epoch {epoch}/{args.epochs}] New best model saved!")
            sys.stdout.flush()
            try:
                save_dir = os.path.dirname(args.new_model_path)
                os.makedirs(save_dir, exist_ok=True)
                
                torch.save(model.state_dict(), args.new_model_path)
                
            except Exception as e:
                pass

    try:
        # this saves the model
        torch.save(model.state_dict(), args.new_model_path)
    except Exception as e:
        pass


if __name__ == "__main__":
    main()