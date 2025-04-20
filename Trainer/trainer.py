import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import time
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

from Losses.charbonnier_loss import CharbonnierLoss
# Import the fast inference functionality
from Models.stoformer_fast_inf import set_fast_inference_mode, fast_validate

class Trainer:
    def __init__(self, model, train_loader, val_loader, save_dir, 
                 device, lr=3e-4, num_epochs=300, save_frequency=10,
                 resume=False, resume_path=None, use_fast_validation=False):
        """
        Initialize the trainer
        
        Args:
            model: The Stoformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            device: Training device ('cuda' or 'cpu')
            lr: Initial learning rate
            num_epochs: Number of training epochs
            save_frequency: How often to save checkpoints
            resume: Whether to resume training from a checkpoint
            resume_path: Path to the checkpoint to resume from
            use_fast_validation: Whether to use fast inference during validation
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.save_frequency = save_frequency
        self.use_fast_validation = use_fast_validation
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize loss function, optimizer and scheduler
        self.criterion = CharbonnierLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize training variables
        self.start_epoch = 0
        self.best_psnr = 0
        
        # Resume training if specified
        if resume and resume_path:
            self._load_checkpoint(resume_path)
            
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_psnr = checkpoint['best_psnr']
        
        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}")
    
    def _save_checkpoint(self, epoch, best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr
        }
        
        # Regular checkpoint in the save_dir
        if not best:
            save_path = os.path.join(self.save_dir, f'model_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved to {save_path}")
        else:
            # Save best model in both save_dir and Results folder
            # First, extract task name from the save_dir path
            task_name = os.path.basename(self.save_dir).replace('stoformer_', '')
            
            # Save in normal checkpoint directory
            save_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, save_path)
            
            # Also save in Results folder
            results_dir = os.path.join('./Results', task_name)
            os.makedirs(results_dir, exist_ok=True)
            results_save_path = os.path.join(results_dir, f'best_model_{task_name}.pth')
            torch.save(checkpoint, results_save_path)
            
            print(f"Best model checkpoint saved to {save_path} and {results_save_path}")
    
    def train_one_epoch(self, epoch):
        """Train the model for one epoch"""
        self.model.train()
        epoch_loss = 0
        start_time = time.time()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Get inputs based on available keys in the batch
            clean = batch['clean'].to(self.device)
            
            # Handle different dataset structures
            if 'noisy' in batch:
                degraded = batch['noisy'].to(self.device)
            elif 'blur' in batch:
                degraded = batch['blur'].to(self.device)
            elif 'rainy' in batch:
                degraded = batch['rainy'].to(self.device)
            else:
                raise KeyError("Batch doesn't contain recognized degraded image key ('noisy', 'blur', or 'rainy')")
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(degraded)
            
            # Compute Charbonnier loss (as used in the Stoformer paper)
            loss = self.criterion(output, clean)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Update learning rate
        self.scheduler.step()
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(self.train_loader)
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch} completed in {elapsed_time:.2f}s - Avg Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model"""
        # If fast validation is enabled, use the optimized version
        if self.use_fast_validation:
            return self.fast_validate(epoch)
            
        self.model.eval()
        psnr_values = []
        
        validation_start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Get inputs based on available keys in the batch
                clean = batch['clean'].to(self.device)
                
                # Handle different dataset structures
                if 'noisy' in batch:
                    degraded = batch['noisy'].to(self.device)
                elif 'blur' in batch:
                    degraded = batch['blur'].to(self.device)
                elif 'rainy' in batch:
                    degraded = batch['rainy'].to(self.device)
                else:
                    raise KeyError("Batch doesn't contain recognized degraded image key ('noisy', 'blur', or 'rainy')")
                
                # Forward pass
                output = self.model(degraded)
                
                # Calculate PSNR
                mse = F.mse_loss(output, clean).item()
                psnr = -10 * np.log10(mse)
                psnr_values.append(psnr)
        
        # Calculate average PSNR
        avg_psnr = np.mean(psnr_values)
        validation_time = time.time() - validation_start_time
        
        print(f"Validation - Epoch {epoch} - Avg PSNR: {avg_psnr:.2f} dB (completed in {validation_time:.2f}s)")
        
        return avg_psnr

    def fast_validate(self, epoch):
        """Validate the model using fast inference mode"""
        validation_start_time = time.time()
        
        # Use the fast validation function from our stoformer_fast_inf module
        avg_psnr = fast_validate(self.model, self.val_loader, self.device)
        
        validation_time = time.time() - validation_start_time
        print(f"Fast Validation - Epoch {epoch} - Avg PSNR: {avg_psnr:.2f} dB (completed in {validation_time:.2f}s)")
        
        return avg_psnr
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.num_epochs} epochs...")
        if self.use_fast_validation:
            print("Using fast validation mode - validation will be ~7-10x faster with minimal accuracy impact")
        
        # Lists to store training history
        train_losses = []
        val_psnrs = []
        
        # Create visualization directory for sample comparisons
        vis_dir = os.path.join('./Results', os.path.basename(self.save_dir).replace('stoformer_', ''), 'epoch_samples')
        os.makedirs(vis_dir, exist_ok=True)
        
        for epoch in range(self.start_epoch, self.num_epochs):
            # Train for one epoch
            train_loss = self.train_one_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validate
            val_psnr = self.validate(epoch)
            val_psnrs.append(val_psnr)
            
            # Save checkpoint if needed
            if (epoch + 1) % self.save_frequency == 0:
                self._save_checkpoint(epoch)
            
            # Save best model
            if val_psnr > self.best_psnr:
                self.best_psnr = val_psnr
                self._save_checkpoint(epoch, best=True)
                print(f"New best model saved with PSNR: {val_psnr:.2f} dB")
            
            # Save a comparison image for this epoch
            if (epoch + 1) % 5 == 0 or epoch == 0:  # Every 5 epochs or first epoch
                # Get a sample image from validation set
                with torch.no_grad():
                    for batch in self.val_loader:
                        # Get a single batch
                        clean = batch['clean'].to(self.device)
                        
                        # Handle different dataset structures
                        if 'noisy' in batch:
                            degraded = batch['noisy'].to(self.device)
                        elif 'blur' in batch:
                            degraded = batch['blur'].to(self.device)
                        elif 'rainy' in batch:
                            degraded = batch['rainy'].to(self.device)
                        else:
                            continue
                        
                        # Forward pass
                        output = self.model(degraded)
                        
                        # Save comparison image
                        self.save_comparison_image(degraded[0], clean[0], output[0], epoch+1, vis_dir)
                        break  # Only need one batch
        
        print("Training completed!")
        
        # Return training history
        return train_losses, val_psnrs
        
    def save_comparison_image(self, degraded, target, output, epoch, save_dir):
        """Save comparison of degraded/target/output images"""
        def tensor_to_image(tensor):
            """Convert tensor to numpy image"""
            img = tensor.cpu().detach().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1) * 255.0
            return img.astype(np.uint8)
        
        # Create figure with 3 subplots
        plt.figure(figsize=(15, 5))
        
        # Get image type
        if hasattr(self, 'image_type'):
            image_type = self.image_type
        else:
            # Try to infer image type from directory name
            dirname = os.path.basename(self.save_dir).lower()
            if 'denois' in dirname:
                image_type = 'noisy'
            elif 'deblur' in dirname:
                image_type = 'blurry'
            elif 'derain' in dirname:
                image_type = 'rainy'
            else:
                image_type = 'degraded'
        
        # Plot degraded image
        plt.subplot(1, 3, 1)
        plt.imshow(tensor_to_image(degraded))
        plt.title(f"{image_type.capitalize()} Input", fontsize=12)
        plt.axis('off')
        
        # Plot output image
        plt.subplot(1, 3, 2)
        plt.imshow(tensor_to_image(output))
        # Calculate PSNR for the output
        mse = F.mse_loss(output, target).item()
        psnr = -10 * np.log10(mse)
        plt.title(f"Output (PSNR: {psnr:.2f} dB)", fontsize=12)
        plt.axis('off')
        
        # Plot target image
        plt.subplot(1, 3, 3)
        plt.imshow(tensor_to_image(target))
        plt.title("Ground Truth", fontsize=12)
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def find_lr(self, start_lr=1e-7, end_lr=1e-2, num_steps=100):
        """Find the optimal learning rate using the learning rate finder technique"""
        # Save current model state 
        old_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Set up optimizer with start_lr
        optimizer = optim.Adam(self.model.parameters(), lr=start_lr)
        
        # Initialize lists to store lr and loss values
        lrs = []
        losses = []
        
        # Calculate the multiplication factor
        mult_factor = (end_lr / start_lr) ** (1 / num_steps)
        
        # Train for num_steps iterations with increasing learning rate
        batch_idx = 0
        smoothed_loss = None
        min_loss = float('inf')
        
        # Use a loop instead of iterating directly to handle updating LR during iteration
        loader_iter = iter(self.train_loader)
        
        try:
            for step in tqdm(range(num_steps), desc="Finding optimal learning rate"):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)
                    
                # Get inputs
                clean = batch['clean'].to(self.device)
                
                # Handle different dataset structures
                if 'noisy' in batch:
                    degraded = batch['noisy'].to(self.device)
                elif 'blur' in batch:
                    degraded = batch['blur'].to(self.device)
                elif 'rainy' in batch:
                    degraded = batch['rainy'].to(self.device)
                else:
                    raise KeyError("Batch doesn't contain a recognized degraded image key")
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(degraded)
                
                # Compute loss
                loss = self.criterion(output, clean)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Smooth the loss
                if smoothed_loss is None:
                    smoothed_loss = loss.item()
                else:
                    smoothed_loss = 0.9 * smoothed_loss + 0.1 * loss.item()
                
                # Record the learning rate and loss
                lrs.append(optimizer.param_groups[0]['lr'])
                losses.append(smoothed_loss)
                
                # Break if loss is exploding
                if step > 0 and losses[-1] > 4 * min_loss:
                    break
                    
                if losses[-1] < min_loss:
                    min_loss = losses[-1]
                
                # Update the learning rate for the next step
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= mult_factor
                    
                batch_idx += 1
        finally:
            # Restore model to its original state
            self.model.load_state_dict(old_state_dict)
        
        # Plot learning rate vs loss
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, losses)
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)
        save_path = os.path.join(self.save_dir, 'lr_finder.png')
        plt.savefig(save_path)
        
        print(f"Learning rate finder results saved to {save_path}")
        
        # Find the learning rate with the steepest downward slope
        min_grad_idx = None
        min_grad = float('inf')
        for i in range(1, len(lrs) - 1):
            grad = (losses[i+1] - losses[i-1]) / (np.log(lrs[i+1]) - np.log(lrs[i-1]))
            if grad < min_grad:
                min_grad = grad
                min_grad_idx = i
        
        if min_grad_idx is not None:
            suggested_lr = lrs[min_grad_idx]
            print(f"Suggested learning rate: {suggested_lr:.6f}")
            return suggested_lr
        else:
            print("Could not determine optimal learning rate.")
            return None 