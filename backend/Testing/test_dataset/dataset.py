import os
import shutil
import random
import pandas as pd
import csv
import logging
import time
import cv2
import numpy as np
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_image_mask_pairs(image_dir, mask_dir):
    pairs = []
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        try:
            base_name = os.path.splitext(img_file)[0]
            if '_' in base_name:
                img_id = base_name.split('_')[0]
            else:
                img_id = base_name
                
            img_id = str(int(img_id))
            
            potential_mask_names = [
                f"{img_id}_mask.png",
                f"{img_id}_mask.jpg",
                f"mask_{img_id}.png",
                f"mask_{img_id}.jpg",
                f"{img_id}.png",
                f"{img_id}.jpg"
            ]
            
            mask_file = None
            for mask_name in potential_mask_names:
                if os.path.exists(os.path.join(mask_dir, mask_name)):
                    mask_file = mask_name
                    break
                    
            if mask_file:
                img_path = os.path.join(image_dir, img_file)
                mask_path = os.path.join(mask_dir, mask_file)
                
                if validate_image_mask_pair(img_path, mask_path):
                    pairs.append((img_file, mask_file, img_id))
        except (ValueError, IndexError) as e:
            logger.debug(f"Skipping {img_file}: {e}")
            continue
    
    return pairs

def validate_image_mask_pair(img_path, mask_path):
    """
    Validate that an image and mask pair are valid.
    
    Args:
        img_path (str): Path to the image file
        mask_path (str): Path to the mask file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            return False
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            return False
            
        img_h, img_w = img.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        
        return (img_h == mask_h and img_w == mask_w) or \
               (img_h % mask_h == 0 and img_w % mask_w == 0) or \
               (mask_h % img_h == 0 and mask_w % img_w == 0)
    except Exception as e:
        logger.error(f"Error validating image-mask pair: {e}")
        return False

def transfer_to_feedback(num_samples=5, balance_classes=True, random_seed=42):
    """
    Transfer a specified number of image-mask pairs from test dataset to feedback folders.
    
    Args:
        num_samples (int): Number of samples to transfer
        balance_classes (bool): Whether to balance TB positive and negative samples
        random_seed (int): Random seed for reproducibility
        
    Returns:
        int: Number of successfully transferred samples
    """
    random.seed(random_seed)
    
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    
    test_data_dir = base_dir / 'Testing' / 'test_dataset'
    image_source_dir = test_data_dir / 'Chest-X-Ray' / 'image'
    mask_source_dir = test_data_dir / 'Chest-X-Ray' / 'mask'
    metadata_path = test_data_dir / 'MetaData.csv'
    
    feedback_dir = base_dir / 'feedback'
    image_dest_dir = feedback_dir / 'images'
    mask_dest_dir = feedback_dir / 'masks'
    feedback_log_path = feedback_dir / 'feedback_log.csv'
    
    os.makedirs(image_dest_dir, exist_ok=True)
    os.makedirs(mask_dest_dir, exist_ok=True)
    
    try:
        metadata_df = pd.read_csv(metadata_path)
        logger.info(f"Found {len(metadata_df)} entries in metadata file")
    except Exception as e:
        logger.error(f"Failed to read metadata: {e}")
        return 0
    
    if not os.path.isdir(image_source_dir):
        logger.error(f"Image source directory not found: {image_source_dir}")
        return 0
    
    if not os.path.isdir(mask_source_dir):
        logger.error(f"Mask source directory not found: {mask_source_dir}")
        return 0
    
    logger.info("Finding valid image-mask pairs...")
    pairs = find_image_mask_pairs(image_source_dir, mask_source_dir)
    logger.info(f"Found {len(pairs)} valid image-mask pairs")
    
    if len(pairs) == 0:
        logger.error("No valid image-mask pairs found")
        return 0
    
    valid_pairs = []
    for img_file, mask_file, img_id in pairs:
        try:
            meta_row = metadata_df[metadata_df['id'] == int(img_id)]
            if len(meta_row) > 0:
                tb_status = meta_row['ptb'].values[0]
                valid_pairs.append((img_file, mask_file, img_id, tb_status))
        except Exception as e:
            logger.warning(f"Error finding metadata for ID {img_id}: {e}")
            continue
    
    logger.info(f"Found {len(valid_pairs)} pairs with valid metadata")
    
    if len(valid_pairs) == 0:
        logger.error("No pairs with valid metadata found")
        return 0
    
    selected_pairs = []
    if balance_classes and num_samples > 1:
        positive_cases = [p for p in valid_pairs if p[3] == 1]
        negative_cases = [p for p in valid_pairs if p[3] == 0]
        
        logger.info(f"Found {len(positive_cases)} TB positive cases and {len(negative_cases)} TB negative cases")
        
        pos_samples = num_samples // 2
        neg_samples = num_samples - pos_samples
        
        if len(positive_cases) < pos_samples:
            pos_samples = len(positive_cases)
            neg_samples = min(num_samples - pos_samples, len(negative_cases))
        elif len(negative_cases) < neg_samples:
            neg_samples = len(negative_cases)
            pos_samples = min(num_samples - neg_samples, len(positive_cases))
            
        selected_pos = random.sample(positive_cases, pos_samples) if pos_samples > 0 else []
        selected_neg = random.sample(negative_cases, neg_samples) if neg_samples > 0 else []
        
        selected_pairs = selected_pos + selected_neg
        random.shuffle(selected_pairs)
    else:
        selected_count = min(num_samples, len(valid_pairs))
        selected_pairs = random.sample(valid_pairs, selected_count)
    
    logger.info(f"Selected {len(selected_pairs)} pairs for transfer")
    
    if not os.path.exists(feedback_log_path):
        with open(feedback_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_filename", "mask_filename", "label", "timestamp"])
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    successful_transfers = 0
    
    for img_file, mask_file, img_id, tb_status in selected_pairs:
        try:
            unique_id = uuid.uuid4().hex[:6]
            
            new_img_filename = f"{timestamp}_{unique_id}_image{os.path.splitext(img_file)[1]}"
            new_mask_filename = f"{timestamp}_{unique_id}_mask{os.path.splitext(mask_file)[1]}"
            
            img_src = os.path.join(image_source_dir, img_file)
            mask_src = os.path.join(mask_source_dir, mask_file)
            img_dst = os.path.join(image_dest_dir, new_img_filename)
            mask_dst = os.path.join(mask_dest_dir, new_mask_filename)
            
            shutil.copy2(img_src, img_dst)
            shutil.copy2(mask_src, mask_dst)
            
            label = "TB" if tb_status == 1 else "Normal"
            
            with open(feedback_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    new_img_filename,
                    new_mask_filename,
                    label,
                    timestamp
                ])
            
            successful_transfers += 1
            logger.info(f"Transferred pair {successful_transfers}/{len(selected_pairs)}: {new_img_filename} ({label})")
            
        except Exception as e:
            logger.error(f"Error transferring pair with ID {img_id}: {e}")
            continue
    
    logger.info(f"Successfully transferred {successful_transfers} image-mask pairs to feedback folders")
    return successful_transfers

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Transfer image-mask pairs from test dataset to feedback folders')
    parser.add_argument('--num-samples', type=int, default=300, help='Number of samples to transfer')
    parser.add_argument('--balance', action='store_true', help='Balance TB positive and negative cases')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    transfer_to_feedback(num_samples=args.num_samples, balance_classes=args.balance, random_seed=args.seed)
