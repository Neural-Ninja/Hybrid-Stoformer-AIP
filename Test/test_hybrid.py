import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import time

# Add parent directory to path for imports
sys.path.append('.')
sys.path.append('..')

# Import models and utils
from Models.hybrid_stoformer import build_hybrid_stoformer
from Models.stoformer2 import set_fast_inference_mode
from Metrics.psnr_ssim import calc_psnr, calc_ssim

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Test Hybrid CNN-Stoformer model")
    
    # Input parameters
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to blurry input image or directory of images')
    parser.add_argument('--gt_path', type=str, default=None,
                        help='Path to ground truth image or directory (optional)')
    parser.add_argument('--is_directory', action='store_true',
                        help='Treat image_path as directory containing multiple images')
    
    # Model parameters
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--cnn_backbone', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='CNN backbone used in the model')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Window size for transformer blocks')
    
    # Processing parameters
    parser.add_argument('--use_patches', action='store_true',
                        help='Process the image in patches to maintain original resolution')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Size of patches when using patch-based processing')
    parser.add_argument('--patch_overlap', type=int, default=32,
                        help='Overlap between patches to blend results smoothly')
    parser.add_argument('--fast_inference', action='store_true',
                        help='Enable fast inference mode for 7-10x speedup with minimal quality impact')
    parser.add_argument('--preserve_aspect_ratio', action='store_true',
                        help='Preserve aspect ratio when resizing (for non-patch mode)')
    
    # Output parameters
    parser.add_argument('--results_dir', type=str, default='./Results/hybrid_test',
                        help='Directory to save results')
    parser.add_argument('--save_comparison', action='store_true', default=True,
                        help='Save side-by-side comparison of input and output')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

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
        # Create figure with 2 subplots (no clean image)
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
        
        # Also save individual images
        output_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split('.')[0] + '_deblurred.png')
        Image.fromarray(output_img).save(output_path)
        print(f"Saved deblurred image to {output_path}")
        
    plt.close()
    
    return {
        'psnr': psnr_val if clean is not None else None,
        'ssim': ssim_val if clean is not None else None
    }

def process_image_in_patches(model, image_tensor, patch_size=256, overlap=32, device='cuda'):
    """
    Process a large image by dividing it into overlapping patches.
    
    Args:
        model: The model
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

def test_single_image(model, image_path, gt_path=None, save_dir=None, device='cuda',
                      use_patches=False, patch_size=256, patch_overlap=32,
                      preserve_aspect_ratio=False, save_comparison=True):
    """Test model on a single image"""
    try:
        # Load input image
        input_img = Image.open(image_path).convert('RGB')
        original_size = input_img.size
        print(f"Input image size: {original_size[0]}x{original_size[1]}")
        
        # Load ground truth image if provided
        clean_tensor = None
        if gt_path:
            clean_img = Image.open(gt_path).convert('RGB')
            print(f"Ground truth image loaded: {clean_img.size[0]}x{clean_img.size[1]}")
        
        # Process using patches or standard resizing
        if use_patches and min(original_size) >= patch_size:
            print(f"Processing image in patches (size: {patch_size}, overlap: {patch_overlap})")
            
            # Convert to tensor without resizing
            to_tensor = transforms.ToTensor()
            input_tensor = to_tensor(input_img).unsqueeze(0).to(device)
            
            # Process image in patches
            start_time = time.time()
            output = process_image_in_patches(
                model=model,
                image_tensor=input_tensor,
                patch_size=patch_size,
                overlap=patch_overlap,
                device=device
            )
            processing_time = time.time() - start_time
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
            target_size = 256  # Default size if not using patches
            
            # Create transforms based on whether to preserve aspect ratio
            if preserve_aspect_ratio:
                transform = transforms.Compose([
                    transforms.Resize(target_size, max_size=target_size*2),
                    transforms.CenterCrop((target_size, target_size)),
                    transforms.ToTensor()
                ])
                print(f"Resizing to {target_size}x{target_size} (preserving aspect ratio with center crop)")
            else:
                transform = transforms.Compose([
                    transforms.Resize((target_size, target_size)),
                    transforms.ToTensor()
                ])
                print(f"Resizing to {target_size}x{target_size} (changing aspect ratio)")
            
            # Apply transforms
            input_tensor = transform(input_img).unsqueeze(0).to(device)
            
            # Process ground truth if available
            if gt_path:
                clean_tensor = transform(clean_img).unsqueeze(0).to(device)
            
            # Forward pass through model
            start_time = time.time()
            with torch.no_grad():
                output = model(input_tensor)
            processing_time = time.time() - start_time
            print(f"Processing completed in {processing_time:.2f} seconds")
        
        # Create save directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Create filename based on input image name
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(save_dir, f"{base_name}_result.png")
        else:
            save_path = None
        
        # Save/show comparison image
        if save_comparison:
            metrics = save_comparison_image(
                blurry=input_tensor[0], 
                output=output[0], 
                clean=clean_tensor[0] if clean_tensor is not None else None,
                save_path=save_path
            )
            
            # Print metrics if ground truth is available
            if clean_tensor is not None:
                print(f"PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
                return metrics
        
        return {"success": True}
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def test_directory(model, image_dir, gt_dir=None, save_dir=None, device='cuda',
                  use_patches=False, patch_size=256, patch_overlap=32,
                  preserve_aspect_ratio=False, save_comparison=True):
    """Test model on all images in a directory"""
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_dir) 
                  if os.path.isfile(os.path.join(image_dir, f)) and 
                  f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    psnr_values = []
    ssim_values = []
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        
        # Find corresponding ground truth if available
        gt_path = None
        if gt_dir:
            gt_file = image_file  # Assume same filename
            gt_path = os.path.join(gt_dir, gt_file)
            if not os.path.exists(gt_path):
                print(f"Warning: Ground truth not found for {image_file}")
                gt_path = None
        
        # Create specific save directory for this image
        if save_dir:
            image_save_dir = os.path.join(save_dir, os.path.splitext(image_file)[0])
            os.makedirs(image_save_dir, exist_ok=True)
        else:
            image_save_dir = None
        
        # Process image
        result = test_single_image(
            model=model,
            image_path=image_path,
            gt_path=gt_path,
            save_dir=image_save_dir,
            device=device,
            use_patches=use_patches,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            preserve_aspect_ratio=preserve_aspect_ratio,
            save_comparison=save_comparison
        )
        
        # Collect metrics if ground truth was used
        if gt_path and 'psnr' in result and result['psnr'] is not None:
            psnr_values.append(result['psnr'])
            ssim_values.append(result['ssim'])
    
    # Print average metrics if ground truth was used
    if psnr_values:
        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        
        print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        
        # Save metrics to file
        if save_dir:
            with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
                f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
                f.write(f"Average SSIM: {avg_ssim:.4f}\n")
                f.write(f"Number of images: {len(psnr_values)}\n")
        
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'num_images': len(psnr_values)
        }
    
    return {"success": True}

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
    print(f"Building hybrid model with {args.cnn_backbone} backbone...")
    model = build_hybrid_stoformer(
        img_size=args.patch_size if args.use_patches else 256,
        window_size=args.window_size,
        cnn_backbone=args.cnn_backbone,
        pretrained=False  # No need for pretrained weights when loading checkpoint
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Check for DataParallel prefix
            if all(k.startswith('module.') for k in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' prefix
                    new_state_dict[name] = v
                state_dict = new_state_dict
                
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
            
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Enable fast inference if requested
    if args.fast_inference:
        print("Enabling fast inference mode")
        set_fast_inference_mode(model, True)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Test on directory or single image
    if args.is_directory:
        print(f"Testing on all images in directory: {args.image_path}")
        result = test_directory(
            model=model,
            image_dir=args.image_path,
            gt_dir=args.gt_path,
            save_dir=args.results_dir,
            device=device,
            use_patches=args.use_patches,
            patch_size=args.patch_size,
            patch_overlap=args.patch_overlap,
            preserve_aspect_ratio=args.preserve_aspect_ratio,
            save_comparison=args.save_comparison
        )
    else:
        print(f"Testing on single image: {args.image_path}")
        result = test_single_image(
            model=model,
            image_path=args.image_path,
            gt_path=args.gt_path,
            save_dir=args.results_dir,
            device=device,
            use_patches=args.use_patches,
            patch_size=args.patch_size,
            patch_overlap=args.patch_overlap,
            preserve_aspect_ratio=args.preserve_aspect_ratio,
            save_comparison=args.save_comparison
        )
    
    # Disable fast inference if it was enabled
    if args.fast_inference:
        set_fast_inference_mode(model, False)
    
    print("Testing completed!")

if __name__ == "__main__":
    main() 