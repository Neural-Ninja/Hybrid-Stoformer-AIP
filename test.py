import os
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import sys
from tqdm import tqdm
import time

# Add the current directory to the path
sys.path.append('.')
sys.path.append('/home/victorazad/AIP-Project/Stoformer')

# Import models
from Models.stoformer import build_stoformer
from Models.stoformer2 import build_stoformer2
try:
    # Try importing the fast inference mode
    from Models.stoformer2 import set_fast_inference_mode
except ImportError:
    # Fallback to the separate file
    from Models.stoformer_fast_inf import set_fast_inference_mode
    
from Utils.dataloader import get_validation_data
from Utils.gopro_preprocess import get_gopro_dataloaders
from Metrics.psnr_ssim import calc_psnr, calc_ssim

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stoformer Testing')
    
    # Dataset parameters
    parser.add_argument('--task', type=str, default='denoising', 
                       choices=['denoising', 'deraining', 'deblurring', 'gopro'],
                       help='Image restoration task (use gopro for GoPro deblurring)')
    parser.add_argument('--clean_dir', type=str, default=None,
                        help='Directory with clean/ground truth images')
    parser.add_argument('--degraded_dir', type=str, default=None,
                        help='Directory with degraded images (rainy/blurred for deraining/deblurring)')
    parser.add_argument('--gopro_dir', type=str, default='./Go-Pro-Deblur-Dataset',
                        help='Directory containing GoPro dataset (for task=gopro)')
    parser.add_argument('--sigma', type=int, nargs='+', default=[15, 25, 50],
                        help='Noise levels for denoising task')
    
    # Single image test
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to a single image for testing (if provided, will only test this image)')
    parser.add_argument('--gt_path', type=str, default=None,
                        help='Path to the ground truth image (optional for single image test)')
    parser.add_argument('--preserve_aspect_ratio', action='store_true',
                        help='Preserve aspect ratio when resizing images (with center crop)')
    parser.add_argument('--use_patches', action='store_true',
                        help='Process large images in patches to maintain original resolution')
    parser.add_argument('--patch_size', type=int, default=256, 
                        help='Size of patches for patch-based processing')
    parser.add_argument('--patch_overlap', type=int, default=32,
                        help='Overlap between patches in pixels')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default=None, choices=['stoformer', 'stoformer2'],
                        help='Type of model to use (stoformer or stoformer2)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Window size for attention')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size for non-patch-based processing')
    
    # Testing parameters
    parser.add_argument('--results_dir', type=str, default='./Results',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Testing batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_images', type=int, default=10,
                        help='Number of test images to save (0 to disable)')
    
    # Fast inference parameter
    parser.add_argument('--fast_inference', action='store_true',
                        help='Use fast inference mode (~7-10x faster with minimal accuracy impact)')
    
    return parser.parse_args()

def process_image_in_patches(model, image_tensor, patch_size=256, overlap=32, device='cuda'):
    """
    Process a large image by dividing it into overlapping patches.
    
    Args:
        model: The Stoformer model
        image_tensor: Input image tensor of shape [1, C, H, W]
        patch_size: Size of patches to process
        overlap: Overlap between patches in pixels
        device: Device to process on
        
    Returns:
        Processed image tensor of the same size as input
    """
    # Get image dimensions
    _, c, h, w = image_tensor.shape
    
    # Ensure image is on the correct device
    if image_tensor.device != device:
        image_tensor = image_tensor.to(device)
    
    # Create output tensor
    output = torch.zeros_like(image_tensor)
    
    # Create weight tensor for blending
    weights = torch.zeros((1, 1, h, w), device=device)
    
    # Create weight mask for smoother blending (higher weight in center)
    weight_mask = torch.ones((1, 1, patch_size, patch_size), device=device)
    if overlap > 0:
        # Create linear ramp for edges
        for i in range(overlap):
            factor = (i + 1) / (overlap + 1)
            weight_mask[:, :, i, :] *= factor  # Top edge
            weight_mask[:, :, patch_size-i-1, :] *= factor  # Bottom edge
            weight_mask[:, :, :, i] *= factor  # Left edge
            weight_mask[:, :, :, patch_size-i-1] *= factor  # Right edge
    
    # Calculate stride
    stride = patch_size - overlap
    
    # Process each patch
    total_patches = ((h - patch_size) // stride + 1) * ((w - patch_size) // stride + 1)
    patches_processed = 0
    
    # Process patches in rows and columns
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size]
            
            # Process patch with model
            with torch.no_grad():
                processed_patch = model(patch)
            
            # Add to output with weight
            output[:, :, y:y+patch_size, x:x+patch_size] += processed_patch * weight_mask
            weights[:, :, y:y+patch_size, x:x+patch_size] += weight_mask
            
            # Update progress
            patches_processed += 1
            if patches_processed % 10 == 0 or patches_processed == total_patches:
                print(f"Processed {patches_processed}/{total_patches} patches ({100*patches_processed/total_patches:.1f}%)")
    
    # Process right edge if needed
    if h % stride != 0:
        y = h - patch_size
        for x in range(0, w - patch_size + 1, stride):
            patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size]
            with torch.no_grad():
                processed_patch = model(patch)
            output[:, :, y:y+patch_size, x:x+patch_size] += processed_patch * weight_mask
            weights[:, :, y:y+patch_size, x:x+patch_size] += weight_mask
    
    # Process bottom edge if needed
    if w % stride != 0:
        x = w - patch_size
        for y in range(0, h - patch_size + 1, stride):
            patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size]
            with torch.no_grad():
                processed_patch = model(patch)
            output[:, :, y:y+patch_size, x:x+patch_size] += processed_patch * weight_mask
            weights[:, :, y:y+patch_size, x:x+patch_size] += weight_mask
    
    # Process bottom-right corner if needed
    if h % stride != 0 and w % stride != 0:
        y = h - patch_size
        x = w - patch_size
        patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size]
        with torch.no_grad():
            processed_patch = model(patch)
        output[:, :, y:y+patch_size, x:x+patch_size] += processed_patch * weight_mask
        weights[:, :, y:y+patch_size, x:x+patch_size] += weight_mask
    
    # Normalize by weights to blend patches
    output = output / (weights.expand_as(output) + 1e-8)
    
    return output

def tensor_to_image(tensor):
    """Convert tensor to numpy image"""
    img = tensor.cpu().clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0)) * 255.0
    return img.astype(np.uint8)

def save_comparison_image(blurry, output, clean=None, save_path=None):
    """Save comparison of blurry, clean, and output images side by side"""
    # Convert tensors to images
    blurry_img = tensor_to_image(blurry)
    output_img = tensor_to_image(output)
    
    if clean is not None:
        clean_img = tensor_to_image(clean)
        # Create figure with 3 subplots
        plt.figure(figsize=(15, 5))
        
        # Add blurry image
        plt.subplot(1, 3, 1)
        plt.imshow(blurry_img)
        plt.title('Blurry Input', fontsize=12)
        plt.axis('off')
        
        # Add model output
        plt.subplot(1, 3, 2)
        plt.imshow(output_img)
        psnr_val = calc_psnr(output.unsqueeze(0), clean.unsqueeze(0))
        ssim_val = calc_ssim(output.unsqueeze(0), clean.unsqueeze(0))
        
        # Handle both tensor and float returns
        if hasattr(psnr_val, 'item'):
            psnr_val = psnr_val.item()
        if hasattr(ssim_val, 'item'):
            ssim_val = ssim_val.item()
            
        plt.title(f'Deblurred (PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f})', fontsize=12)
        plt.axis('off')
        
        # Add clean image
        plt.subplot(1, 3, 3)
        plt.imshow(clean_img)
        plt.title('Ground Truth', fontsize=12)
        plt.axis('off')
    else:
        # Create figure with 2 subplots side by side
        plt.figure(figsize=(12, 6))
        
        # Add blurry image
        plt.subplot(1, 2, 1)
        plt.imshow(blurry_img)
        plt.title('Blurry Input', fontsize=14)
        plt.axis('off')
        
        # Add model output
        plt.subplot(1, 2, 2)
        plt.imshow(output_img)
        plt.title('Deblurred Output', fontsize=14)
        plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved comparison image to {save_path}")
        
        # Also save the individual processed image
        if save_path:
            output_path = os.path.join(os.path.dirname(save_path), 'deblurred.png')
            Image.fromarray(output_img).save(output_path)
            print(f"Saved deblurred image to {output_path}")
            
            # Also save the input blurry image for reference
            blurry_path = os.path.join(os.path.dirname(save_path), 'blurry.png')
            Image.fromarray(blurry_img).save(blurry_path)
    
    plt.close()
    
    return {
        'psnr': psnr_val if clean is not None else None,
        'ssim': ssim_val if clean is not None else None
    }

def test_single_image(model, image_path, gt_path=None, device='cuda', save_dir=None, 
                     win_size=8, img_size=256, preserve_aspect_ratio=False, 
                     fast_inference=False, use_patches=False, patch_size=256, 
                     patch_overlap=32):
    """Test model on a single image"""
    
    # Enable fast inference if requested
    if fast_inference:
        set_fast_inference_mode(model, True)
    
    # Create save directory if not provided
    if save_dir is None:
        save_dir = os.path.join('Results', 'single_image_test')
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Load blurry image
        blurry_img = Image.open(image_path).convert('RGB')
        original_size = blurry_img.size
        print(f"Original image size: {original_size[0]}x{original_size[1]}")
        
        # Load ground truth image if provided
        clean_tensor = None
        if gt_path:
            clean_img = Image.open(gt_path).convert('RGB')
        
        # Process using patches or standard resizing
        if use_patches and min(original_size) >= patch_size:
            print(f"Processing image in patches (size: {patch_size}, overlap: {patch_overlap})")
            
            # Convert to tensor without resizing
            to_tensor = transforms.ToTensor()
            blurry_tensor = to_tensor(blurry_img).unsqueeze(0).to(device)
            
            # Process image in patches
            processing_start = time.time()
            output = process_image_in_patches(
                model=model,
                image_tensor=blurry_tensor,
                patch_size=patch_size,
                overlap=patch_overlap,
                device=device
            )
            processing_time = time.time() - processing_start
            print(f"Patch processing completed in {processing_time:.2f} seconds")
            
            # Process ground truth if available
            if gt_path:
                clean_tensor = to_tensor(clean_img).unsqueeze(0)
                
                # Ensure clean tensor has same size as output
                if clean_tensor.shape[2:] != output.shape[2:]:
                    clean_tensor = torch.nn.functional.interpolate(
                        clean_tensor, 
                        size=(output.shape[2], output.shape[3]), 
                        mode='bilinear', 
                        align_corners=False
                    )
                clean_tensor = clean_tensor.to(device)
        else:
            # Standard processing with resizing
            print(f"Processing with standard resize to {img_size}x{img_size}")
            
            # Create transforms based on whether to preserve aspect ratio
            if preserve_aspect_ratio:
                # Preserve aspect ratio with padding
                transform = transforms.Compose([
                    transforms.Resize(img_size, max_size=img_size*2),  # Resize smaller edge to img_size
                    transforms.CenterCrop((img_size, img_size)),      # Crop to square
                    transforms.ToTensor()
                ])
                print(f"Resizing to {img_size}x{img_size} (preserving aspect ratio with center crop)")
            else:
                # Direct resize to target size
                transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor()
                ])
                print(f"Resizing to {img_size}x{img_size} (changing aspect ratio)")
            
            # Apply transforms
            blurry_tensor = transform(blurry_img).unsqueeze(0).to(device)
            
            # Process ground truth if available
            if gt_path:
                clean_tensor = transform(clean_img).unsqueeze(0).to(device)
            
            # Forward pass through model
            processing_start = time.time()
            with torch.no_grad():
                output = model(blurry_tensor)
            processing_time = time.time() - processing_start
            print(f"Processing completed in {processing_time:.2f} seconds")
        
        # Create save directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'result.png')
            
            # Also save the processed image directly (not just the comparison)
            processed_img = tensor_to_image(output[0])
            Image.fromarray(processed_img).save(os.path.join(save_dir, 'processed.png'))
        else:
            save_path = None
        
        # Save/show comparison image
        metrics = save_comparison_image(
            blurry=blurry_tensor[0], 
            output=output[0], 
            clean=clean_tensor[0] if clean_tensor is not None else None,
            save_path=save_path
        )
        
        # Print metrics if ground truth is available
        if clean_tensor is not None:
            print(f"PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
            
        print(f"Processing complete! Results saved to {save_dir}")
    
    finally:
        # Always disable fast inference if it was enabled
        if fast_inference:
            set_fast_inference_mode(model, False)
    
    return metrics

def batch_test(model, test_loader, device, save_dir=None, save_images=True, max_save_images=10, fast_inference=False):
    """Test model on a batch of images and compute metrics"""
    # Initialize metrics
    psnr_values = []
    ssim_values = []
    
    # Create save directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        images_dir = os.path.join(save_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
    
    # Enable fast inference if requested
    if fast_inference:
        set_fast_inference_mode(model, True)
        print("Using fast inference mode (7-10x faster with minimal accuracy impact)")
    
    try:
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
                # Get inputs (handle different dataset formats)
                if 'clean' in batch and 'blur' in batch:
                    clean = batch['clean'].to(device)
                    degraded = batch['blur'].to(device)
                elif 'clean' in batch and 'noisy' in batch:
                    clean = batch['clean'].to(device)
                    degraded = batch['noisy'].to(device)
                else:
                    raise ValueError("Unsupported dataset format")
                
                # Forward pass
                output = model(degraded)
                
                # Calculate metrics
                batch_psnr = calc_psnr(output, clean)
                batch_ssim = calc_ssim(output, clean)
                
                # Handle both tensor and float return types
                if hasattr(batch_psnr, 'item'):
                    batch_psnr = batch_psnr.item()
                if hasattr(batch_ssim, 'item'):
                    batch_ssim = batch_ssim.item()
                
                psnr_values.append(batch_psnr)
                ssim_values.append(batch_ssim)
                
                # Save result images
                if save_images and idx < max_save_images:
                    # Save individual images
                    for b in range(min(degraded.size(0), 4)):  # Save up to 4 images per batch
                        image_idx = idx * degraded.size(0) + b
                        if image_idx >= max_save_images:
                            break
                            
                        # Create comparison images
                        save_name = f'image_{image_idx:04d}.png'
                        save_comparison_image(
                            blurry=degraded[b],
                            output=output[b],
                            clean=clean[b],
                            save_path=os.path.join(images_dir, save_name)
                        )
    finally:
        # Always disable fast inference mode when done
        if fast_inference:
            set_fast_inference_mode(model, False)
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    # Print and save results
    print(f"\nTest Results:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    if save_dir:
        # Save metrics to a text file
        with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
            f.write(f"Test Results:\n")
            f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    
    results = {
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }
    
    return results

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    task_name = args.task
    if task_name == 'gopro':
        task_name = 'deblurring_gopro'
    
    # Try to infer model_type from checkpoint path if not specified
    model_type = args.model_type
    if model_type is None:
        if 'stoformer2' in args.checkpoint_path:
            model_type = 'stoformer2'
        else:
            model_type = 'stoformer'
        print(f"Model type inferred from checkpoint path: {model_type}")
    
    # Check if fast inference is supported
    if args.fast_inference and model_type != 'stoformer2':
        print("Warning: Fast inference is only supported for stoformer2. Disabling fast inference.")
        args.fast_inference = False
    
    # Create model based on type
    print(f"Creating {model_type} model...")
    if model_type == 'stoformer':
        model = build_stoformer(img_size=args.img_size, window_size=args.window_size)
    else:  # stoformer2
        model = build_stoformer2(img_size=args.img_size, window_size=args.window_size)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        print(f"Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Load model weights with verbose output
            if 'state_dict' in checkpoint:
                print("Loading using 'state_dict' key")
                state_dict = checkpoint['state_dict']
                
                # Check if model was trained with DataParallel (keys start with 'module.')
                if all(k.startswith('module.') for k in state_dict.keys()):
                    print("Detected DataParallel trained model, removing 'module.' prefix")
                    # Create new OrderedDict without the 'module.' prefix
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove 'module.' prefix
                        new_state_dict[name] = v
                    state_dict = new_state_dict
                
                # Load the state dictionary
                model.load_state_dict(state_dict)
            elif 'model_state_dict' in checkpoint:
                print("Loading using 'model_state_dict' key")
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("Warning: Neither state_dict nor model_state_dict found in checkpoint, trying to load directly...")
                model.load_state_dict(checkpoint)
        else:
            print("Checkpoint is not a dictionary, trying to load directly...")
            model.load_state_dict(checkpoint)
        
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Create results directories
    results_dir = os.path.join(args.results_dir, f"{model_type}_{task_name}")
    os.makedirs(results_dir, exist_ok=True)
    
    test_dir = os.path.join(results_dir, 'test_results')
    os.makedirs(test_dir, exist_ok=True)
    
    # Handle single image testing
    if args.image_path:
        print(f"Testing single image: {args.image_path}")
        test_single_image(
            model=model,
            image_path=args.image_path,
            gt_path=args.gt_path,
            device=device,
            save_dir=test_dir,
            win_size=args.window_size,
            img_size=args.img_size,
            preserve_aspect_ratio=args.preserve_aspect_ratio,
            fast_inference=args.fast_inference,
            use_patches=args.use_patches,
            patch_size=args.patch_size,
            patch_overlap=args.patch_overlap
        )
        return
    
    # Batch testing
    print("Setting up test dataset...")
    if args.task == 'gopro':
        # Use GoPro test set
        dataloaders = get_gopro_dataloaders(
            gopro_dir=args.gopro_dir,
            patch_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            only_test=True
        )
        test_loader = dataloaders['test_loader']
        print(f"Test dataset size: {len(dataloaders['test_dataset'])}")
    else:
        # Use custom test set
        if args.task == 'denoising':
            test_dataset = get_validation_data(
                rgb_dir=args.clean_dir,
                task=args.task,
                sigma=args.sigma
            )
        else:  # deraining or deblurring
            test_dataset = get_validation_data(
                rgb_dir=args.clean_dir,
                task=args.task,
                rainy_dir=args.degraded_dir if args.task == 'deraining' else None,
                blur_dir=args.degraded_dir if args.task == 'deblurring' else None
            )
        
        # Create test dataloader
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        print(f"Test dataset size: {len(test_dataset)}")
    
    # Run batch test
    test_results = batch_test(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=test_dir,
        save_images=args.save_images > 0,
        max_save_images=args.save_images,
        fast_inference=args.fast_inference
    )
    
    # Print final results
    print("\nFinal Test Results:")
    print(f"Average PSNR: {test_results['psnr']:.2f} dB")
    print(f"Average SSIM: {test_results['ssim']:.4f}")
    
    print(f"Testing completed. Results saved to {test_dir}")

if __name__ == '__main__':
    main() 