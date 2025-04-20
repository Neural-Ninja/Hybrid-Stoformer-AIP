#!/usr/bin/env python3
import os
import argparse
import random
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import glob

"""
Script to preprocess the GoPro dataset for deblurring task.
Splits the dataset into train, validation, and test sets,
and provides dataloaders for training and evaluation.

Dataset structure:
Go-Pro-Deblur-Dataset/
├── blur/
│   └── images/
│       ├── 002801.png
│       ├── ...
└── sharp/
    └── images/
        ├── 002801.png
        ├── ...
"""

def is_image_file(filename):
    """Check if a file is an image"""
    return any(filename.endswith(extension) for extension 
              in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class RandomCrop:
    """Crop randomly the image in a sample"""
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        clean, blur = sample['clean'], sample['blur']
        
        # Get random crop coordinates
        i, j, h, w = transforms.RandomCrop.get_params(
            clean, output_size=self.output_size)
        
        # Apply same crop to both images
        clean = TF.crop(clean, i, j, h, w)
        blur = TF.crop(blur, i, j, h, w)
        
        return {'clean': clean, 'blur': blur}

class RandomHorizontalFlip:
    """Horizontally flip the given image randomly with a given probability"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, sample):
        clean, blur = sample['clean'], sample['blur']
        
        if random.random() < self.p:
            clean = TF.hflip(clean)
            blur = TF.hflip(blur)
            
        return {'clean': clean, 'blur': blur}

class RandomVerticalFlip:
    """Vertically flip the given image randomly with a given probability"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, sample):
        clean, blur = sample['clean'], sample['blur']
        
        if random.random() < self.p:
            clean = TF.vflip(clean)
            blur = TF.vflip(blur)
            
        return {'clean': clean, 'blur': blur}

class RandomRotation:
    """Rotate the image by angle"""
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles
        
    def __call__(self, sample):
        clean, blur = sample['clean'], sample['blur']
        
        angle = random.choice(self.angles)
        if angle > 0:
            clean = TF.rotate(clean, angle)
            blur = TF.rotate(blur, angle)
            
        return {'clean': clean, 'blur': blur}

class ToTensor:
    """Convert images in sample to tensors"""
    def __call__(self, sample):
        clean, blur = sample['clean'], sample['blur']
        
        # Convert PIL Images to tensors
        clean = TF.to_tensor(clean)
        blur = TF.to_tensor(blur)
        
        return {'clean': clean, 'blur': blur}

class GoProDataset(Dataset):
    def __init__(self, root_dir, split='train', patch_size=256, transform=None, augment=True):
        """
        Args:
            root_dir (str): Directory with GoPro dataset
            split (str): 'train', 'val', or 'test'
            patch_size (int): Size of the patches to extract
            transform (callable, optional): Optional transform to be applied on a sample
            augment (bool): Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform
        self.augment = augment
        self.split = split
        
        # Following the structure of the GoPro dataset directory
        # The dataset has blur/images/ and sharp/images/ folders
        self.blur_dir = os.path.join(root_dir, 'blur', 'images')
        self.sharp_dir = os.path.join(root_dir, 'sharp', 'images')
        
        # Check if directories exist
        if not os.path.exists(self.blur_dir):
            raise ValueError(f"Blur directory not found: {self.blur_dir}")
        if not os.path.exists(self.sharp_dir):
            raise ValueError(f"Sharp directory not found: {self.sharp_dir}")
        
        # Get image paths
        self.blur_paths = sorted(glob.glob(os.path.join(self.blur_dir, '*.png')))
        self.sharp_paths = sorted(glob.glob(os.path.join(self.sharp_dir, '*.png')))
        
        # Verify images found
        if len(self.blur_paths) == 0 or len(self.sharp_paths) == 0:
            raise ValueError(f"No images found in {self.blur_dir} or {self.sharp_dir}")
        
        # Check if counts match
        if len(self.blur_paths) != len(self.sharp_paths):
            raise ValueError("Number of blurry and sharp images don't match!")
        
        # For train/val/test split, we create pseudo-random splits based on the image index
        # This ensures a reproducible split while keeping related frames together
        total_images = len(self.blur_paths)
        
        # Split according to the official Stoformer paper: 
        # Train on ~80% of images, validate/test on ~10% each
        if split == 'train':
            start_idx = 0
            end_idx = int(total_images * 0.8)
        elif split == 'val':
            start_idx = int(total_images * 0.8)
            end_idx = int(total_images * 0.9)
        else:  # test
            start_idx = int(total_images * 0.9)
            end_idx = total_images
        
        self.blur_paths = self.blur_paths[start_idx:end_idx]
        self.sharp_paths = self.sharp_paths[start_idx:end_idx]
    
    def __len__(self):
        return len(self.blur_paths)
    
    def __getitem__(self, idx):
        # Read images
        blur_img = Image.open(self.blur_paths[idx]).convert('RGB')
        sharp_img = Image.open(self.sharp_paths[idx]).convert('RGB')
        
        # Convert to tensors
        to_tensor = transforms.ToTensor()
        blur_tensor = to_tensor(blur_img)
        sharp_tensor = to_tensor(sharp_img)
        
        # Random crop during training
        if self.split == 'train':
            h, w = blur_tensor.shape[1], blur_tensor.shape[2]
            
            # Get random crop
            i = random.randint(0, h - self.patch_size)
            j = random.randint(0, w - self.patch_size)
            
            blur_tensor = blur_tensor[:, i:i+self.patch_size, j:j+self.patch_size]
            sharp_tensor = sharp_tensor[:, i:i+self.patch_size, j:j+self.patch_size]
            
            # Data augmentation
            if self.augment:
                # Random horizontal flip
                if random.random() > 0.5:
                    blur_tensor = torch.flip(blur_tensor, dims=[2])
                    sharp_tensor = torch.flip(sharp_tensor, dims=[2])
                
                # Random vertical flip
                if random.random() > 0.5:
                    blur_tensor = torch.flip(blur_tensor, dims=[1])
                    sharp_tensor = torch.flip(sharp_tensor, dims=[1])
                
                # Random 90-degree rotation
                if random.random() > 0.5:
                    blur_tensor = torch.rot90(blur_tensor, dims=[1, 2])
                    sharp_tensor = torch.rot90(sharp_tensor, dims=[1, 2])
        
        # Apply additional transforms if specified
        if self.transform:
            blur_tensor = self.transform(blur_tensor)
            sharp_tensor = self.transform(sharp_tensor)
        
        return {
            'blur': blur_tensor,
            'clean': sharp_tensor,
            'blur_path': self.blur_paths[idx],
            'clean_path': self.sharp_paths[idx]
        }

def split_gopro_dataset(gopro_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split GoPro dataset into train, validation, and test sets
    
    Args:
        gopro_dir: Directory containing GoPro dataset
        train_ratio: Ratio of images for training set (default: 0.8)
        val_ratio: Ratio of images for validation set (default: 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing file paths for each split
    """
    random.seed(seed)
    np.random.seed(seed)
    
    blur_dir = os.path.join(gopro_dir, 'blur', 'images')
    sharp_dir = os.path.join(gopro_dir, 'sharp', 'images')
    
    # Verify that directories exist
    if not os.path.exists(blur_dir):
        raise ValueError(f"Blur images directory not found: {blur_dir}")
    if not os.path.exists(sharp_dir):
        raise ValueError(f"Sharp images directory not found: {sharp_dir}")
    
    # Get list of image files
    blur_files = sorted([f for f in os.listdir(blur_dir) if is_image_file(f)])
    sharp_files = sorted([f for f in os.listdir(sharp_dir) if is_image_file(f)])
    
    # Verify that blur and sharp images match
    if len(blur_files) != len(sharp_files):
        raise ValueError(f"Number of blur images ({len(blur_files)}) doesn't match number of sharp images ({len(sharp_files)})")
    
    # Create pairs of blur and sharp image paths
    paired_files = list(zip(
        [os.path.join(blur_dir, f) for f in blur_files],
        [os.path.join(sharp_dir, f) for f in sharp_files]
    ))
    
    # Shuffle pairs
    random.shuffle(paired_files)
    
    # Split into train, validation, and test sets
    n_total = len(paired_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_pairs = paired_files[:n_train]
    val_pairs = paired_files[n_train:n_train+n_val]
    test_pairs = paired_files[n_train+n_val:]
    
    # Unzip pairs into separate blur and sharp lists
    train_blur, train_sharp = zip(*train_pairs) if train_pairs else ([], [])
    val_blur, val_sharp = zip(*val_pairs) if val_pairs else ([], [])
    test_blur, test_sharp = zip(*test_pairs) if test_pairs else ([], [])
    
    return {
        'train': {'blur': list(train_blur), 'sharp': list(train_sharp)},
        'val': {'blur': list(val_blur), 'sharp': list(val_sharp)},
        'test': {'blur': list(test_blur), 'sharp': list(test_sharp)}
    }

def get_gopro_dataloaders(gopro_dir, patch_size=256, batch_size=4, num_workers=4):
    """
    Create dataloaders for GoPro dataset
    
    Args:
        gopro_dir: Directory containing GoPro dataset
        patch_size: Size of image patches for training
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary containing dataloaders and datasets
    """
    # Create datasets
    train_dataset = GoProDataset(gopro_dir, split='train', patch_size=patch_size)
    val_dataset = GoProDataset(gopro_dir, split='val', patch_size=patch_size)
    test_dataset = GoProDataset(gopro_dir, split='test', patch_size=patch_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,  # Test one image at a time
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }

def main():
    parser = argparse.ArgumentParser(description='Process GoPro dataset and create dataloaders')
    parser.add_argument('--gopro_dir', type=str, default='./Go-Pro-Deblur-Dataset',
                        help='Directory containing GoPro dataset')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Size of patches to extract (default: 256)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading (default: 4)')
    args = parser.parse_args()
    
    # Get dataloaders
    dataloaders = get_gopro_dataloaders(
        gopro_dir=args.gopro_dir,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Print dataset information
    print(f"Training dataset size: {len(dataloaders['train_dataset'])}")
    print(f"Validation dataset size: {len(dataloaders['val_dataset'])}")
    print(f"Test dataset size: {len(dataloaders['test_dataset'])}")
    
    # Print dataloader information
    print(f"Number of training batches: {len(dataloaders['train_loader'])}")
    print(f"Number of validation batches: {len(dataloaders['val_loader'])}")
    print(f"Number of test batches: {len(dataloaders['test_loader'])}")
    
    print("Dataloaders created successfully!")

if __name__ == '__main__':
    main() 