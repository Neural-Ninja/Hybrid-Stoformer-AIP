import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from Metrics.metrics import Metrics
from Utils.dataloader import get_validation_data, create_dataloaders

class Tester:
    """
    Class to test image restoration models
    """
    def __init__(self, model, test_loader, device, save_dir=None, task='denoising'):
        """
        Initialize the tester
        
        Args:
            model: The model to test
            test_loader: Test data loader
            device: Device to use ('cuda' or 'cpu')
            save_dir: Directory to save outputs (optional)
            task: Image restoration task
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.task = task
        self.metrics = Metrics()
        
        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def test(self, save_images=False) -> Dict:
        """
        Test the model on the test dataset
        
        Args:
            save_images: Whether to save output images
            
        Returns:
            Dictionary with metrics
        """
        print(f"Testing {self.task} model...")
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
                # Get inputs
                clean = batch['clean'].to(self.device)
                noisy = batch['noisy'].to(self.device)
                
                # Forward pass
                output = self.model(noisy)
                
                # Update metrics
                sigma = batch.get('sigma', None)
                kwargs = {'sigma': sigma[0].item()} if sigma is not None else {}
                self.metrics.update(output, clean, task=self.task, **kwargs)
                
                # Save images if requested
                if save_images and self.save_dir:
                    self._save_images(batch_idx, clean, noisy, output)
        
        # Compute and return metrics
        results = self.metrics.compute()
        
        # Print metrics
        print(f"Test Results - {self.task}")
        print(f"PSNR: {results['psnr']:.2f} dB, SSIM: {results['ssim']:.4f}")
        
        # Print task-specific metrics if available
        if 'task_specific' in results:
            print("Task-Specific Results:")
            for key, value in results['task_specific'].items():
                print(f"  {key}: PSNR: {value['psnr']:.2f} dB, SSIM: {value['ssim']:.4f}")
        
        return results
    
    def _save_images(self, batch_idx: int, clean: torch.Tensor, noisy: torch.Tensor, output: torch.Tensor):
        """
        Save input, output, and ground truth images
        
        Args:
            batch_idx: Batch index
            clean: Ground truth image
            noisy: Input image
            output: Output image
        """
        # Convert tensors to numpy arrays
        clean = clean.detach().cpu().permute(0, 2, 3, 1).numpy()
        noisy = noisy.detach().cpu().permute(0, 2, 3, 1).numpy()
        output = output.detach().cpu().permute(0, 2, 3, 1).numpy()
        
        # Scale to 0-255 range
        clean = np.clip(clean * 255.0, 0, 255).astype(np.uint8)
        noisy = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        
        # Save each image in the batch
        for i in range(clean.shape[0]):
            save_path = os.path.join(self.save_dir, f"sample_{batch_idx}_{i}")
            os.makedirs(save_path, exist_ok=True)
            
            # Save images
            cv2.imwrite(os.path.join(save_path, 'clean.png'), cv2.cvtColor(clean[i], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_path, 'noisy.png'), cv2.cvtColor(noisy[i], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_path, 'output.png'), cv2.cvtColor(output[i], cv2.COLOR_RGB2BGR))

def test_model(model, clean_dir, device, save_dir=None, task='denoising', sigma=None, degraded_dir=None, batch_size=1, num_workers=4):
    """
    Test a model on a dataset
    
    Args:
        model: The model to test
        clean_dir: Directory with clean/ground truth images
        device: Device to use ('cuda' or 'cpu')
        save_dir: Directory to save outputs (optional)
        task: Image restoration task
        sigma: Noise levels for denoising
        degraded_dir: Directory with degraded images
        batch_size: Batch size for testing
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary with metrics
    """
    # Prepare test data
    if task == 'denoising':
        test_dataset = get_validation_data(
            rgb_dir=clean_dir,
            task=task,
            sigma=sigma
        )
    else:  # deraining or deblurring
        test_dataset = get_validation_data(
            rgb_dir=clean_dir,
            task=task,
            rainy_dir=degraded_dir if task == 'deraining' else None,
            blur_dir=degraded_dir if task == 'deblurring' else None
        )
    
    # Create test dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create tester and run test
    tester = Tester(model, test_loader, device, save_dir=save_dir, task=task)
    results = tester.test(save_images=(save_dir is not None))
    
    return results 