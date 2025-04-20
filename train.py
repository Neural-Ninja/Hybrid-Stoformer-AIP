import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from Models.stoformer import build_stoformer
from Models.stoformer2 import build_stoformer2
from Utils.dataloader import get_training_data, get_validation_data, create_dataloaders
from Utils.gopro_preprocess import get_gopro_dataloaders
from Trainer.trainer import Trainer
from Test.tester_gopro import GoproTester
# Import for benchmarking fast validation
from Models.stoformer_fast_inf import benchmark_inference, compare_inference_accuracy

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_training_history(train_losses, val_psnrs, save_path):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot validation PSNR
    ax2.plot(epochs, val_psnrs, 'r-')
    ax2.set_title('Validation PSNR')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stoformer Training')
    
    # Dataset parameters
    parser.add_argument('--task', type=str, default='denoising', 
                        choices=['denoising', 'deraining', 'deblurring', 'gopro'],
                        help='Image restoration task (use gopro for GoPro deblurring)')
    parser.add_argument('--clean_dir', type=str, default=None,
                        help='Directory with clean/ground truth images')
    parser.add_argument('--degraded_dir', type=str, default=None,
                        help='Directory with degraded images (rainy/blurred for deraining/deblurring)')
    parser.add_argument('--val_clean_dir', type=str, default=None,
                        help='Directory with validation clean images (if different from training)')
    parser.add_argument('--val_degraded_dir', type=str, default=None,
                        help='Directory with validation degraded images (if different from training)')
    parser.add_argument('--gopro_dir', type=str, default='./Go-Pro-Deblur-Dataset',
                        help='Directory containing GoPro dataset (for task=gopro)')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Training patch size')
    parser.add_argument('--sigma', type=int, nargs='+', default=[15, 25, 50],
                        help='Noise levels for denoising task')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='stoformer', 
                        choices=['stoformer', 'stoformer2'],
                        help='Model architecture to use')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Window size for attention')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Initial learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='./Results',
                        help='Directory to save results')
    parser.add_argument('--save_frequency', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--find_lr', action='store_true',
                        help='Run learning rate finder instead of training')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run testing on a pretrained model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model for testing')
    parser.add_argument('--save_images', type=int, default=10,
                        help='Number of test images to save (0 to disable)')
    
    # Fast validation parameters
    parser.add_argument('--fast_validation', action='store_true',
                        help='Use fast validation mode (~7-10x faster with minimal accuracy impact)')
    parser.add_argument('--benchmark_fast_validation', action='store_true',
                        help='Run a benchmark to compare regular and fast validation')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    task_name = args.task
    if task_name == 'gopro':
        task_name = 'deblurring_gopro'

    model_save_dir = os.path.join(args.save_dir, f"{args.model}_{task_name}")
    results_dir = os.path.join(args.results_dir, f"{args.model}_{task_name}")
    
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Get data loaders based on task
    if args.task == 'gopro':
        print("Preparing GoPro dataset...")
        dataloaders = get_gopro_dataloaders(
            gopro_dir=args.gopro_dir,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        train_loader = dataloaders['train_loader']
        val_loader = dataloaders['val_loader']
        test_loader = dataloaders['test_loader']
        
        print(f"Training dataset size: {len(dataloaders['train_dataset'])}")
        print(f"Validation dataset size: {len(dataloaders['val_dataset'])}")
        print(f"Test dataset size: {len(dataloaders['test_dataset'])}")
    else:
        print(f"Preparing {args.task} dataset...")
        
        # Get training data
        if args.task == 'denoising':
            train_dataset = get_training_data(
                rgb_dir=args.clean_dir,
                patch_size=args.patch_size,
                task=args.task,
                sigma=args.sigma
            )
        else:  # deraining or deblurring
            train_dataset = get_training_data(
                rgb_dir=args.clean_dir,
                patch_size=args.patch_size,
                task=args.task,
                rainy_dir=args.degraded_dir if args.task == 'deraining' else None,
                blur_dir=args.degraded_dir if args.task == 'deblurring' else None
            )
        
        # Get validation data
        val_clean_dir = args.val_clean_dir if args.val_clean_dir else args.clean_dir
        val_degraded_dir = args.val_degraded_dir if args.val_degraded_dir else args.degraded_dir
        
        if args.task == 'denoising':
            val_dataset = get_validation_data(
                rgb_dir=val_clean_dir,
                task=args.task,
                sigma=args.sigma
            )
        else:  # deraining or deblurring
            val_dataset = get_validation_data(
                rgb_dir=val_clean_dir,
                task=args.task,
                rainy_dir=val_degraded_dir if args.task == 'deraining' else None,
                blur_dir=val_degraded_dir if args.task == 'deblurring' else None
            )
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        test_loader = val_loader  # Use validation set for testing too
    
    # For test-only mode
    if args.test_only:
        if not args.model_path:
            print("Error: Must provide --model_path for --test_only mode")
            return
        
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Extract model from checkpoint
        if 'model_state_dict' in checkpoint:
            # Build model 
            if args.model == 'stoformer':
                model = build_stoformer(img_size=args.img_size, window_size=args.window_size)
            elif args.model == 'stoformer2':
                model = build_stoformer2(img_size=args.img_size, window_size=args.window_size)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Error: Could not find model state dict in checkpoint")
            return
        
        model = model.to(device)
        model.eval()
        
        # Create tester directory
        test_save_dir = os.path.join(results_dir, 'test_results')
        os.makedirs(test_save_dir, exist_ok=True)
        
        # Test model
        print("Testing model...")
        tester = GoproTester(model, test_loader, device, test_save_dir)
        test_results = tester.test(save_images=args.save_images > 0, max_save_images=args.save_images)
        
        return
    
    # Build model for training
    print(f"Building {args.model} model...")
    if args.model == 'stoformer':
        model = build_stoformer(img_size=args.img_size, window_size=args.window_size)
    elif args.model == 'stoformer2':
        model = build_stoformer2(img_size=args.img_size, window_size=args.window_size)
    print(f"Model built with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Move model to device
    model = model.to(device)
    
    # Run benchmark if requested
    if args.benchmark_fast_validation and args.model == 'stoformer2':
        # Only stoformer2 has the optimized window attention
        print("\n==== Running Fast Validation Benchmark ====")
        
        # Get a sample batch for benchmarking
        sample_batch = next(iter(val_loader))
        if 'noisy' in sample_batch:
            sample_input = sample_batch['noisy'].to(device)
        elif 'blur' in sample_batch:
            sample_input = sample_batch['blur'].to(device)
        elif 'rainy' in sample_batch:
            sample_input = sample_batch['rainy'].to(device)
        
        # Benchmark inference speed
        speed_results = benchmark_inference(model, sample_input, runs=3)
        
        # Compare accuracy
        accuracy_results = compare_inference_accuracy(model, val_loader, device)
        
        # Print benchmark summary
        print("\n==== Fast Validation Benchmark Summary ====")
        print(f"Speed improvement: {speed_results['speedup']:.2f}x faster")
        print(f"Accuracy difference: {accuracy_results['psnr_diff']:.3f} dB PSNR")
        print(f"Fast validation is {speed_results['speedup']:.1f}x faster with a negligible {abs(accuracy_results['psnr_diff']):.3f} dB PSNR difference")
        print("==========================================\n")
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=model_save_dir,
        device=device,
        lr=args.lr,
        num_epochs=args.epochs,
        save_frequency=args.save_frequency,
        resume=args.resume,
        resume_path=args.resume_path,
        use_fast_validation=args.fast_validation
    )
    
    # Find learning rate if requested
    if args.find_lr:
        print("Running learning rate finder...")
        suggested_lr = trainer.find_lr()
        print(f"Suggested learning rate: {suggested_lr}")
        return
    
    # Train model
    print("Starting training...")
    train_losses, val_psnrs = trainer.train()
    
    # Plot training history
    print("Plotting training history...")
    plot_path = os.path.join(results_dir, 'training_history.png')
    plot_training_history(train_losses, val_psnrs, plot_path)
    print(f"Training history plot saved to {plot_path}")
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main() 