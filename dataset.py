import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from pathlib import Path


class HandXrayDataset(Dataset):
    """Dataset for Hand X-ray segmentation"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path to dataset directory
            split: 'train' or 'val'
            transform: Albumentations transform pipeline
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Get image and mask paths
        self.image_dir = self.root_dir / split / 'images'
        self.mask_dir = self.root_dir / split / 'masks'
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        
        # Verify masks exist
        self.samples = []
        for img_path in self.image_files:
            # Extract base name (e.g., 1405 from 1405.png)
            base_name = img_path.stem
            mask_path = self.mask_dir / f"{base_name}_mask.png"
            
            if mask_path.exists():
                self.samples.append({
                    'image': str(img_path),
                    'mask': str(mask_path)
                })
            else:
                print(f"Warning: Mask not found for {img_path.name}")
        
        print(f"{split} dataset: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image (grayscale)
        image = cv2.imread(sample['image'], cv2.IMREAD_GRAYSCALE)
        
        # Load mask (binary)
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Binarize mask (0 or 1)
        mask = (mask > 127).astype(np.float32)
        
        # Apply augmentations
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Add channel dimension and convert to torch
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)    # (1, H, W)
        
        return {
            'image': image,
            'mask': mask,
            'filename': Path(sample['image']).name
        }


class TestDataset(Dataset):
    """Dataset for inference on test images"""
    
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir: Path to directory containing test images
            transform: Albumentations transform pipeline
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        print(f"Test dataset: {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image (grayscale)
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        original_size = image.shape  # (H, W)
        
        # Normalize
        image_norm = image.astype(np.float32) / 255.0
        
        # Apply transform if provided
        if self.transform is not None:
            transformed = self.transform(image=image_norm)
            image_norm = transformed['image']
        
        # Add channel dimension
        image_tensor = torch.from_numpy(image_norm).unsqueeze(0)  # (1, H, W)
        
        return {
            'image': image_tensor,
            'original_size': original_size,
            'filename': img_path.name
        }