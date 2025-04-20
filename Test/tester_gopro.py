import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Import metrics from our metrics module instead of skimage
from Metrics.metrics import Metrics
from Metrics.psnr_ssim import calc_psnr, calc_ssim

class GoproTester:
    """Tester for evaluating models on GoPro dataset and saving comparison images"""
    
    def __init__(self, model, test_loader, device, save_dir):
        """
        Initialize the tester
        
        Args:
            model: Trained Stoformer model
            test_loader: Test data loader
            device: Device to use ('cuda' or 'cpu')
            save_dir: Directory to save evaluation results and comparison images
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        
        # Create save directories
        self.results_dir = os.path.join(save_dir, 'results')
        self.images_dir = os.path.join(save_dir, 'images')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = Metrics()
        
        # Set model to evaluation mode
        self.model.eval()
    
    def tensor_to_image(self, tensor):
        """Convert tensor to numpy image"""
        img = tensor.cpu().clamp(0, 1).numpy()
        img = np.transpose(img, (1, 2, 0)) * 255.0
        return img.astype(np.uint8)
    
    def save_comparison_image(self, blurry, clean, output, index):
        """Save comparison of blurry, clean, and output images side by side"""
        # Convert tensors to images
        blurry_img = self.tensor_to_image(blurry)
        clean_img = self.tensor_to_image(clean)
        output_img = self.tensor_to_image(output)
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Add blurry image
        plt.subplot(1, 3, 1)
        plt.imshow(blurry_img)
        plt.title('Blurry Input', fontsize=12)
        plt.axis('off')
        
        # Add model output
        plt.subplot(1, 3, 2)
        plt.imshow(output_img)
        psnr_val = calc_psnr(output.unsqueeze(0), clean.unsqueeze(0)).item()
        ssim_val = calc_ssim(output.unsqueeze(0), clean.unsqueeze(0)).item()
        plt.title(f'Deblurred (PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f})', fontsize=12)
        plt.axis('off')
        
        # Add clean image
        plt.subplot(1, 3, 3)
        plt.imshow(clean_img)
        plt.title('Ground Truth', fontsize=12)
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        comp_path = os.path.join(self.images_dir, f'comparison_{index}.png')
        plt.savefig(comp_path, dpi=200)
        plt.close()
        
        # Also save individual images
        Image.fromarray(blurry_img).save(os.path.join(self.images_dir, f'blurry_{index}.png'))
        Image.fromarray(clean_img).save(os.path.join(self.images_dir, f'clean_{index}.png'))
        Image.fromarray(output_img).save(os.path.join(self.images_dir, f'output_{index}.png'))
        
        # Save a side-by-side comparison of just blurry and deblurred images
        plt.figure(figsize=(10, 5))
        
        # Add blurry image
        plt.subplot(1, 2, 1)
        plt.imshow(blurry_img)
        plt.title('Blurry Input', fontsize=12)
        plt.axis('off')
        
        # Add model output
        plt.subplot(1, 2, 2)
        plt.imshow(output_img)
        plt.title(f'Deblurred', fontsize=12)
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, f'before_after_{index}.png'), dpi=200)
        plt.close()
    
    def test(self, save_images=True, max_save_images=10):
        """
        Evaluate model on test dataset
        
        Args:
            save_images: Whether to save comparison images
            max_save_images: Maximum number of images to save
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Reset metrics
        self.metrics.reset()
        
        # Create a directory for the samples
        samples_dir = os.path.join(self.save_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        
        all_metrics = []
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
                # Get inputs
                clean = batch['clean'].to(self.device)
                blur = batch['blur'].to(self.device)
                
                # Forward pass
                output = self.model(blur)
                
                # Update metrics (batch-wise)
                self.metrics.update(output, clean, task='deblurring')
                
                # Calculate metrics for this sample
                psnr_val = calc_psnr(output, clean).item()
                ssim_val = calc_ssim(output, clean).item()
                
                # Add to all metrics for sorting
                all_metrics.append({
                    'idx': idx,
                    'psnr': psnr_val,
                    'ssim': ssim_val,
                    'batch': batch,
                    'output': output
                })
                
                # Save comparison images
                if save_images:
                    for i in range(clean.size(0)):
                        if idx * clean.size(0) + i < max_save_images:
                            self.save_comparison_image(
                                blur[i], clean[i], output[i], idx * clean.size(0) + i
                            )
        
        # Sort samples by PSNR (best and worst)
        all_metrics.sort(key=lambda x: x['psnr'])
        
        # Save worst 5 and best 5 samples
        if save_images and len(all_metrics) >= 10:
            worst_samples = all_metrics[:5]
            best_samples = all_metrics[-5:]
            
            # Save worst samples
            for i, sample in enumerate(worst_samples):
                idx = sample['idx']
                batch = sample['batch']
                output = sample['output']
                
                for j in range(min(1, len(batch['clean']))):
                    self.save_comparison_image(
                        batch['blur'][j].to(self.device), 
                        batch['clean'][j].to(self.device), 
                        output[j], 
                        f"worst_{i}"
                    )
            
            # Save best samples
            for i, sample in enumerate(best_samples):
                idx = sample['idx']
                batch = sample['batch']
                output = sample['output']
                
                for j in range(min(1, len(batch['clean']))):
                    self.save_comparison_image(
                        batch['blur'][j].to(self.device), 
                        batch['clean'][j].to(self.device), 
                        output[j], 
                        f"best_{i}"
                    )
        
        # Compute and get metrics
        metrics_results = self.metrics.compute()
        avg_psnr = metrics_results['psnr']
        avg_ssim = metrics_results['ssim']
        
        # Save metrics to text file
        with open(os.path.join(self.results_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Number of test images: {len(self.metrics.psnr_values)}\n")
        
        # Print results
        print(f"Test Results - {self.metrics}")
        
        # Return metrics
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'num_images': len(self.metrics.psnr_values)
        }

def test_model(model_path, test_loader, device, save_dir):
    """
    Load a trained model and evaluate it
    
    Args:
        model_path: Path to the saved model checkpoint
        test_loader: Test data loader
        device: Device to use ('cuda' or 'cpu')
        save_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model'] if 'model' in checkpoint else checkpoint.get('model_state_dict', None)
    
    if model is None:
        raise ValueError("Could not find model in checkpoint")
    
    # Initialize tester
    tester = GoproTester(model, test_loader, device, save_dir)
    
    # Run evaluation
    return tester.test() 