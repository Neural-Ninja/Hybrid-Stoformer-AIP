import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import cv2

def is_image_file(filename):
    """Check if a file is an image based on its extension"""
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp'])

class RandomCrop(object):
    """Crop randomly the image in a sample"""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        clean, noisy = sample['clean'], sample['noisy']
        h, w = clean.shape[-2:]
        new_h, new_w = self.output_size

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        clean = clean[:, top: top + new_h, left: left + new_w]
        noisy = noisy[:, top: top + new_h, left: left + new_w]

        return {'clean': clean, 'noisy': noisy}

class RandomFlip(object):
    """Randomly flip an image horizontally and/or vertically"""
    def __call__(self, sample):
        clean, noisy = sample['clean'], sample['noisy']
        
        # Horizontal flip
        if random.random() < 0.5:
            clean = torch.flip(clean, [2])
            noisy = torch.flip(noisy, [2])
            
        # Vertical flip
        if random.random() < 0.5:
            clean = torch.flip(clean, [1])
            noisy = torch.flip(noisy, [1])
            
        return {'clean': clean, 'noisy': noisy}

class RandomRotation(object):
    """Randomly rotate an image by 90, 180, or 270 degrees"""
    def __call__(self, sample):
        clean, noisy = sample['clean'], sample['noisy']
        
        k = random.choice([1, 2, 3])  # Rotate by k * 90 degrees
        clean = torch.rot90(clean, k, [1, 2])
        noisy = torch.rot90(noisy, k, [1, 2])
        
        return {'clean': clean, 'noisy': noisy}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""
    def __call__(self, sample):
        clean, noisy = sample['clean'], sample['noisy']
        
        # Handle different input types (could be PIL Image, numpy array, etc.)
        if isinstance(clean, np.ndarray):
            # Convert numpy array: H x W x C -> C x H x W
            clean = clean.transpose((2, 0, 1))
            noisy = noisy.transpose((2, 0, 1))
            
            # Convert to torch tensors
            clean = torch.from_numpy(clean).float().div(255)
            noisy = torch.from_numpy(noisy).float().div(255)
        else:  # Assume PIL Image
            transform = transforms.ToTensor()
            clean = transform(clean)
            noisy = transform(noisy)
            
        return {'clean': clean, 'noisy': noisy}

class DenoisingDataset(Dataset):
    """Dataset for image denoising"""
    def __init__(self, clean_dir, transform=None, patch_size=256, sigma=None):
        """
        Args:
            clean_dir (string): Directory with clean images
            transform (callable, optional): Optional transform to be applied on a sample
            patch_size (int): Size of the patches to extract
            sigma (list, optional): List of noise levels for training. If None, random values in [0, 55] will be used
        """
        self.clean_paths = sorted([os.path.join(clean_dir, x) for x in os.listdir(clean_dir) if is_image_file(x)])
        self.transform = transform
        self.patch_size = patch_size
        self.sigma = sigma if sigma is not None else [15, 25, 50]  # Common noise levels
        
    def __len__(self):
        return len(self.clean_paths)
        
    def __getitem__(self, idx):
        # Load clean image
        clean_img = Image.open(self.clean_paths[idx]).convert('RGB')
        
        # Convert to numpy array for adding noise
        clean_np = np.array(clean_img)
        
        # Random noise level
        noise_level = random.choice(self.sigma)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level/255.0, clean_np.shape).astype(np.float32)
        noisy_np = clean_np / 255.0 + noise
        noisy_np = np.clip(noisy_np, 0, 1) * 255
        noisy_np = noisy_np.astype(np.uint8)
        
        # Convert back to PIL Images
        clean_img = Image.fromarray(clean_np)
        noisy_img = Image.fromarray(noisy_np)
        
        # Create sample dictionary
        sample = {'clean': clean_img, 'noisy': noisy_img, 'sigma': noise_level}
        
        # Apply transformations
        if self.transform:
            transformed_sample = self.transform(sample)
            transformed_sample['sigma'] = torch.tensor([noise_level/255.0])
            return transformed_sample
        
        # Convert to tensors if no transform is provided
        to_tensor = ToTensor()
        tensor_sample = to_tensor(sample)
        tensor_sample['sigma'] = torch.tensor([noise_level/255.0])
        
        return tensor_sample

class DerainDataset(Dataset):
    """Dataset for image deraining"""
    def __init__(self, rainy_dir, clean_dir, transform=None, patch_size=256):
        """
        Args:
            rainy_dir (string): Directory with rainy images
            clean_dir (string): Directory with clean images
            transform (callable, optional): Optional transform to be applied on a sample
            patch_size (int): Size of the patches to extract
        """
        self.rainy_paths = sorted([os.path.join(rainy_dir, x) for x in os.listdir(rainy_dir) if is_image_file(x)])
        self.clean_paths = sorted([os.path.join(clean_dir, x) for x in os.listdir(clean_dir) if is_image_file(x)])
        
        # Make sure we have a valid dataset
        assert len(self.rainy_paths) == len(self.clean_paths), "Number of rainy and clean images should be the same!"
        
        self.transform = transform
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.rainy_paths)
        
    def __getitem__(self, idx):
        # Load clean and rainy images
        clean_img = Image.open(self.clean_paths[idx]).convert('RGB')
        rainy_img = Image.open(self.rainy_paths[idx]).convert('RGB')
        
        # Create sample dictionary
        sample = {'clean': clean_img, 'noisy': rainy_img}
        
        # Apply transformations
        if self.transform:
            return self.transform(sample)
        
        # Convert to tensors if no transform is provided
        to_tensor = ToTensor()
        return to_tensor(sample)

class DeblurDataset(Dataset):
    """Dataset for image deblurring"""
    def __init__(self, blur_dir, sharp_dir, transform=None, patch_size=256):
        """
        Args:
            blur_dir (string): Directory with blurred images
            sharp_dir (string): Directory with sharp images
            transform (callable, optional): Optional transform to be applied on a sample
            patch_size (int): Size of the patches to extract
        """
        self.blur_paths = sorted([os.path.join(blur_dir, x) for x in os.listdir(blur_dir) if is_image_file(x)])
        self.sharp_paths = sorted([os.path.join(sharp_dir, x) for x in os.listdir(sharp_dir) if is_image_file(x)])
        
        # Make sure we have a valid dataset
        assert len(self.blur_paths) == len(self.sharp_paths), "Number of blurred and sharp images should be the same!"
        
        self.transform = transform
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.blur_paths)
        
    def __getitem__(self, idx):
        # Load sharp and blurred images
        sharp_img = Image.open(self.sharp_paths[idx]).convert('RGB')
        blur_img = Image.open(self.blur_paths[idx]).convert('RGB')
        
        # Create sample dictionary
        sample = {'clean': sharp_img, 'noisy': blur_img}
        
        # Apply transformations
        if self.transform:
            return self.transform(sample)
        
        # Convert to tensors if no transform is provided
        to_tensor = ToTensor()
        return to_tensor(sample)

def get_training_data(rgb_dir, patch_size, task='denoising', sigma=None, rainy_dir=None, blur_dir=None):
    """
    Create a training dataset and dataloader
    
    Args:
        rgb_dir (str): Directory of clean/ground truth images
        patch_size (int): Training patch size
        task (str): Task type ('denoising', 'deraining', 'deblurring')
        sigma (list): List of noise levels for denoising
        rainy_dir (str): Directory of rainy images for deraining
        blur_dir (str): Directory of blurred images for deblurring
    """
    # Define transforms
    transform = transforms.Compose([
        ToTensor(),
        RandomCrop((patch_size, patch_size)),
        RandomFlip(),
        RandomRotation()
    ])
    
    # Create dataset based on task
    if task == 'denoising':
        train_dataset = DenoisingDataset(rgb_dir, transform=transform, patch_size=patch_size, sigma=sigma)
    elif task == 'deraining':
        assert rainy_dir is not None, "rainy_dir must be provided for deraining task"
        train_dataset = DerainDataset(rainy_dir, rgb_dir, transform=transform, patch_size=patch_size)
    elif task == 'deblurring':
        assert blur_dir is not None, "blur_dir must be provided for deblurring task"
        train_dataset = DeblurDataset(blur_dir, rgb_dir, transform=transform, patch_size=patch_size)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return train_dataset

def get_validation_data(rgb_dir, task='denoising', sigma=None, rainy_dir=None, blur_dir=None):
    """
    Create a validation dataset and dataloader
    
    Args:
        rgb_dir (str): Directory of clean/ground truth images
        task (str): Task type ('denoising', 'deraining', 'deblurring')
        sigma (list): List of noise levels for denoising
        rainy_dir (str): Directory of rainy images for deraining
        blur_dir (str): Directory of blurred images for deblurring
    """
    # Define transforms - only ToTensor for validation (no augmentation)
    transform = transforms.Compose([
        ToTensor()
    ])
    
    # Create dataset based on task
    if task == 'denoising':
        val_dataset = DenoisingDataset(rgb_dir, transform=transform, sigma=sigma)
    elif task == 'deraining':
        assert rainy_dir is not None, "rainy_dir must be provided for deraining task"
        val_dataset = DerainDataset(rainy_dir, rgb_dir, transform=transform)
    elif task == 'deblurring':
        assert blur_dir is not None, "blur_dir must be provided for deblurring task"
        val_dataset = DeblurDataset(blur_dir, rgb_dir, transform=transform)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return val_dataset

def create_dataloaders(train_dataset, val_dataset, batch_size, num_workers):
    """
    Create training and validation dataloaders
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
    """
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,  # Always use batch size 1 for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 