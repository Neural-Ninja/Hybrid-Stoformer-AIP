import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import random
import time

# Add parent directory to path for imports
sys.path.append('.')
sys.path.append('..')

# Import model and utils
from Models.hybrid_stoformer import build_hybrid_stoformer
from Models.stoformer2 import build_stoformer2 
from Utils.gopro_preprocess import get_gopro_dataloaders
from Metrics.psnr_ssim import calc_psnr, calc_ssim
from Test.test_hybrid import process_image_in_patches

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
    parser = argparse.ArgumentParser(description="Train Hybrid CNN-Stoformer model")
    
    # Dataset parameters
    parser.add_argument('--gopro_dir', type=str, default='./GoPro', 
                        help='Directory containing GoPro dataset')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Training patch size')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    
    # Model parameters
    parser.add_argument('--cnn_backbone', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='CNN backbone for feature extraction')
    parser.add_argument('--stoformer_checkpoint', type=str, default=None,
                        help='Path to pretrained Stoformer2 checkpoint')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Window size for transformer blocks')
    
    # Fast inference parameters
    parser.add_argument('--use_fast_inference', action='store_true',
                        help='Use fast inference mode during validation')
    parser.add_argument('--use_patches', action='store_true',
                        help='Process large images in patches during validation')
    parser.add_argument('--patch_overlap', type=int, default=32,
                        help='Overlap size between patches when using patch-based inference')
    
    # Training parameters
    parser.add_argument('--freeze_cnn', action='store_true',
                        help='Freeze CNN backbone during initial training')
    parser.add_argument('--freeze_epochs', type=int, default=50,
                        help='Number of epochs to freeze CNN backbone')
    parser.add_argument('--total_epochs', type=int, default=200,
                        help='Total number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Initial learning rate')
    parser.add_argument('--lr_cnn', type=float, default=2e-5,
                        help='Learning rate for CNN backbone (after unfreezing)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Checkpointing and logging
    parser.add_argument('--save_dir', type=str, default='./Results/hybrid_stoformer',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint frequency (epochs)')
    parser.add_argument('--val_freq', type=int, default=5,
                        help='Validation frequency (epochs)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def load_pretrained_weights(model, stoformer_checkpoint, device):
    """Load pretrained Stoformer2 weights into the transformer part of hybrid model."""
    print(f"Loading pretrained weights from {stoformer_checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(stoformer_checkpoint, map_location=device)
    
    # Extract state dict
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
    
    # Create mappings for transformer components
    transformer_mappings = {
        'conv': 'conv',
        'upsample_0': 'upsample_0',
        'decoderlayer_0': 'decoderlayer_0',
        'upsample_1': 'upsample_1',
        'decoderlayer_1': 'decoderlayer_1',
        'upsample_2': 'upsample_2',
        'decoderlayer_2': 'decoderlayer_2',
        'upsample_3': 'upsample_3',
        'decoderlayer_3': 'decoderlayer_3',
        'output_proj': 'output_proj'
    }
    
    # Create new state dict for hybrid model (transformer components only)
    hybrid_state_dict = {}
    skipped_keys = []
    transferred_keys = []
    
    for k, v in state_dict.items():
        # Check if key belongs to transformer components
        for src_key, dst_key in transformer_mappings.items():
            if k.startswith(src_key):
                new_key = k  # Same key structure
                hybrid_state_dict[new_key] = v
                transferred_keys.append(k)
                break
        else:
            skipped_keys.append(k)
    
    # Load the weights into model
    missing_keys, unexpected_keys = model.load_state_dict(hybrid_state_dict, strict=False)
    
    print(f"Transferred {len(transferred_keys)} keys from Stoformer2 checkpoint")
    print(f"Skipped {len(skipped_keys)} keys from checkpoint")
    print(f"Missing {len(missing_keys)} keys in model")
    print(f"Unexpected {len(unexpected_keys)} keys in model")
    
    return model

def get_param_groups(model, args):
    """Separate parameter groups for different learning rates."""
    # Handle DataParallel model
    if isinstance(model, nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
        
    # CNN backbone parameters (lower learning rate or frozen)
    cnn_params = list(model_module.cnn_extractor.parameters())
    
    # All other parameters (higher learning rate)
    other_params = []
    for name, param in model.named_parameters():
        if not any(name.startswith(f"module.cnn_extractor.{layer}") for layer in 
                  ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]) and \
           not any(name.startswith(f"cnn_extractor.{layer}") for layer in 
                  ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]):
            other_params.append(param)
    
    if args.freeze_cnn:
        return [{'params': other_params, 'lr': args.lr}]
    else:
        return [
            {'params': cnn_params, 'lr': args.lr_cnn},
            {'params': other_params, 'lr': args.lr}
        ]

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args):
    """Train model for one epoch."""
    model.train()
    
    # Get the actual model module if it's wrapped in DataParallel
    if isinstance(model, nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    # Freeze CNN if in freezing stage
    if args.freeze_cnn and epoch < args.freeze_epochs:
        for param in model_module.cnn_extractor.parameters():
            param.requires_grad = False
    else:
        # Unfreeze CNN if we've passed the freezing stage
        for param in model_module.cnn_extractor.parameters():
            param.requires_grad = True
        
        # Update optimizer with CNN parameters if first epoch after unfreezing
        if args.freeze_cnn and epoch == args.freeze_epochs:
            print(f"Unfreezing CNN backbone at epoch {epoch}")
            optimizer.param_groups = get_param_groups(model, args)
    
    loss_sum = 0
    psnr_sum = 0
    sample_count = 0
    
    # Initialize progress bar
    train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    
    for batch in train_pbar:
        # Get data
        if 'clean' in batch and 'blur' in batch:
            clean = batch['clean'].to(device)
            blur = batch['blur'].to(device)
        else:
            raise ValueError("Unsupported dataset format")
        
        # Forward pass
        optimizer.zero_grad()
        output = model(blur)
        
        # Calculate loss
        loss = criterion(output, clean)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate PSNR
        batch_psnr = calc_psnr(output, clean)
        if isinstance(batch_psnr, torch.Tensor):
            batch_psnr = batch_psnr.item()
        
        # Update metrics
        batch_size = clean.size(0)
        loss_sum += loss.item() * batch_size
        psnr_sum += batch_psnr * batch_size
        sample_count += batch_size
        
        # Update progress bar
        train_pbar.set_postfix({
            'loss': loss.item(),
            'psnr': batch_psnr,
            'avg_loss': loss_sum / sample_count,
            'avg_psnr': psnr_sum / sample_count
        })
    
    # Calculate averages
    avg_loss = loss_sum / sample_count
    avg_psnr = psnr_sum / sample_count
    
    return avg_loss, avg_psnr

def validate(model, val_loader, criterion, device, args):
    """Validate model on validation set."""
    model.eval()
    
    # Enable fast inference mode for validation if specified
    if args.use_fast_inference:
        # Get actual model (handle DataParallel)
        if isinstance(model, nn.DataParallel):
            model.module.set_fast_inference_mode(True)
        else:
            model.set_fast_inference_mode(True)
        print("Using fast inference mode for validation")
    
    loss_sum = 0
    psnr_sum = 0
    ssim_sum = 0
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Get data
            if 'clean' in batch and 'blur' in batch:
                clean = batch['clean'].to(device)
                blur = batch['blur'].to(device)
            else:
                raise ValueError("Unsupported dataset format")
            
            # Forward pass
            output = model(blur)

            # output = process_image_in_patches(
            # model=model,
            # image_tensor=blur,
            # patch_size=args.patch_size,
            # overlap=args.patch_overlap,
            # device=device
            # )
            
            # Calculate loss and metrics
            loss = criterion(output, clean)
            batch_psnr = calc_psnr(output, clean)
            batch_ssim = calc_ssim(output, clean)
            
            # Convert to float if tensor
            if isinstance(batch_psnr, torch.Tensor):
                batch_psnr = batch_psnr.item()
            if isinstance(batch_ssim, torch.Tensor):
                batch_ssim = batch_ssim.item()
            
            # Update metrics
            batch_size = clean.size(0)
            loss_sum += loss.item() * batch_size
            psnr_sum += batch_psnr * batch_size
            ssim_sum += batch_ssim * batch_size
            sample_count += batch_size
    
    # Disable fast inference mode after validation (back to training mode)
    if args.use_fast_inference:
        # Get actual model (handle DataParallel)
        if isinstance(model, nn.DataParallel):
            model.module.set_fast_inference_mode(False)
        else:
            model.set_fast_inference_mode(False)
    
    # Calculate averages
    avg_loss = loss_sum / sample_count
    avg_psnr = psnr_sum / sample_count
    avg_ssim = ssim_sum / sample_count
    
    return avg_loss, avg_psnr, avg_ssim

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print fast inference settings
    if args.use_fast_inference:
        print("Fast inference mode ENABLED for validation")
        if args.use_patches:
            print(f"Patch-based processing ENABLED with size {args.patch_size} and overlap {args.patch_overlap}")
    
    # Get dataloaders
    print("Loading datasets...")
    dataloaders = get_gopro_dataloaders(
        gopro_dir=args.gopro_dir,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    train_loader = dataloaders['train_loader']
    val_loader = dataloaders['val_loader']
    
    print(f"Train dataset size: {len(dataloaders['train_dataset'])}")
    print(f"Val dataset size: {len(dataloaders['val_dataset'])}")
    
    # Build model
    print(f"Building hybrid model with {args.cnn_backbone} backbone...")
    model = build_hybrid_stoformer(
        img_size=args.patch_size,
        window_size=args.window_size,
        cnn_backbone=args.cnn_backbone,
        pretrained=True
    )
    
    # Load pretrained Stoformer2 weights if provided
    if args.stoformer_checkpoint:
        model = load_pretrained_weights(model, args.stoformer_checkpoint, device)
    
    # Move model to device
    model = model.to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Loss function
    criterion = nn.L1Loss()
    
    # Optimizer with separate parameter groups
    param_groups = get_param_groups(model, args)
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_epochs, eta_min=1e-7)
    
    # Initialize best metrics
    best_psnr = 0
    best_epoch = 0
    
    # Training loop
    print(f"Starting training for {args.total_epochs} epochs...")
    for epoch in range(args.total_epochs):
        # Train one epoch
        start_time = time.time()
        train_loss, train_psnr = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args
        )
        epoch_time = time.time() - start_time
        
        # Print results
        print(f"Epoch {epoch+1}/{args.total_epochs} - "
              f"Time: {epoch_time:.2f}s - "
              f"Train Loss: {train_loss:.4f} - "
              f"Train PSNR: {train_psnr:.2f} dB")
        
        # Validate
        if (epoch + 1) % args.val_freq == 0:
            val_loss, val_psnr, val_ssim = validate(
                model, val_loader, criterion, device, args
            )
            
            print(f"Validation - "
                  f"Loss: {val_loss:.4f} - "
                  f"PSNR: {val_psnr:.2f} dB - "
                  f"SSIM: {val_ssim:.4f}")
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch + 1
                
                # Get model state dict (handle DataParallel)
                model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model_state_dict,
                    'optimizer': optimizer.state_dict(),
                    'psnr': val_psnr,
                    'ssim': val_ssim,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                
                print(f"New best model saved (PSNR: {val_psnr:.2f} dB)")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            # Get model state dict (handle DataParallel)
            model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model_state_dict,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth'))
        
        # Update scheduler
        scheduler.step()
    
    print(f"Training completed. Best PSNR: {best_psnr:.2f} dB at epoch {best_epoch}")

if __name__ == "__main__":
    main() 