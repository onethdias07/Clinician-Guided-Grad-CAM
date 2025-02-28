import os
import csv
import torch
import numpy as np
from PIL import Image
import torchxrayvision as xrv
import torchxrayvision.datasets as xrv_datasets
import torchvision.transforms as T

class XRayMaskDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        super().__init__()
        self.transform = transform
        self.samples = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pathology_idx = int(row['pathology_idx'])
                real_label = float(row['real_label'])
                self.samples.append({
                    'xray_path': row['xray_path'],
                    'mask_path': row['mask_path'],
                    'pathology_idx': pathology_idx,
                    'real_label': real_label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]
        pil_img = Image.open(record['xray_path']).convert('L')
        img_np = np.array(pil_img, dtype=np.float32)
        img_np = xrv.datasets.normalize(img_np, 255)
        if img_np.mean() > 0.7:
            img_np = 1.0 - img_np
        img_np = img_np[None, ...]
        if self.transform:
            img_np = self.transform(img_np)
        xray_tensor = torch.from_numpy(img_np)
        mask_pil = Image.open(record['mask_path']).convert('L')
        mask_np = np.array(mask_pil, dtype=np.float32) / 255.0
        pathology_idx = record['pathology_idx']
        real_label = record['real_label']
        return xray_tensor, mask_np, pathology_idx, real_label
