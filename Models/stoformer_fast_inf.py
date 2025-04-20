import torch
import torch.nn as nn
import time
from tqdm import tqdm

def set_fast_inference_mode(model, enable=True):
    """Enable or disable fast inference mode for all transformer blocks in the model.
    
    This significantly speeds up validation by using a smart sampling strategy
    for window shifts instead of computing all possible combinations.
    
    Args:
        model: The Stoformer model
        enable: Boolean indicating whether to enable (True) or disable (False) fast mode
    """
    # Set the fast_inference flag on all StoTransformerBlock instances
    for module in model.modules():
        if hasattr(module, '__class__') and 'StoTransformerBlock' in module.__class__.__name__:
            module.fast_inference = enable
    return model

def benchmark_inference(model, input_tensor, runs=5):
    """
    Benchmark inference speed with and without fast inference optimization.
    
    Args:
        model: The Stoformer model
        input_tensor: An input tensor to use for benchmarking
        runs: Number of runs to average timing over
        
    Returns:
        Dictionary with timing results
    """
    results = {}
    
    model.eval()  # Ensure model is in evaluation mode
    
    # Run with standard inference
    set_fast_inference_mode(model, False)
    torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    
    # Warmup
    with torch.no_grad():
        _ = model(input_tensor)
    torch.cuda.synchronize()
    
    # Benchmark standard inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)
            torch.cuda.synchronize()
    standard_time = (time.time() - start_time) / runs
    results['standard_inference_time'] = standard_time
    
    # Run with fast inference
    set_fast_inference_mode(model, True)
    torch.cuda.synchronize()
    
    # Warmup
    with torch.no_grad():
        _ = model(input_tensor)
    torch.cuda.synchronize()
    
    # Benchmark fast inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)
            torch.cuda.synchronize()
    fast_time = (time.time() - start_time) / runs
    results['fast_inference_time'] = fast_time
    
    # Calculate speedup
    results['speedup'] = standard_time / fast_time
    
    # Print benchmark results
    print("\n----- Inference Speed Benchmark -----")
    print(f"Standard inference: {standard_time:.4f}s per image")
    print(f"Fast inference:     {fast_time:.4f}s per image")
    print(f"Speedup:            {results['speedup']:.2f}x")
    print("------------------------------------\n")
    
    return results

def fast_validate(model, val_loader, device, calculate_psnr=True):
    """
    Run validation with fast inference enabled.
    
    This is a drop-in replacement for the normal validation function
    that enables fast inference mode.
    
    Args:
        model: The Stoformer model
        val_loader: Validation data loader
        device: Device to run validation on
        calculate_psnr: Whether to calculate and return PSNR values
        
    Returns:
        Average PSNR value if calculate_psnr is True
    """
    model.eval()
    # Enable fast inference mode
    set_fast_inference_mode(model, True)
    
    psnr_values = []
    
    try:
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc="Fast Validation")
            for batch in val_iter:
                # Get inputs based on available keys in the batch
                clean = batch['clean'].to(device)
                
                # Handle different dataset structures
                if 'noisy' in batch:
                    degraded = batch['noisy'].to(device)
                elif 'blur' in batch:
                    degraded = batch['blur'].to(device)
                elif 'rainy' in batch:
                    degraded = batch['rainy'].to(device)
                else:
                    raise KeyError("Batch doesn't contain recognized degraded image key ('noisy', 'blur', or 'rainy')")
                
                # Forward pass with fast inference
                output = model(degraded)
                
                # Calculate PSNR if requested
                if calculate_psnr:
                    mse = torch.nn.functional.mse_loss(output, clean).item()
                    psnr = -10 * torch.log10(torch.tensor(mse)).item()
                    psnr_values.append(psnr)
    finally:
        # Always restore standard inference mode when done
        set_fast_inference_mode(model, False)
    
    # Calculate average PSNR
    if calculate_psnr and psnr_values:
        avg_psnr = sum(psnr_values) / len(psnr_values)
        return avg_psnr
    
    return None

def compare_inference_accuracy(model, val_loader, device):
    """
    Compare the accuracy between standard and fast inference modes.
    
    Args:
        model: The Stoformer model
        val_loader: Validation data loader
        device: Device to run validation on
        
    Returns:
        Dictionary with accuracy comparison results
    """
    result = {}
    
    # Standard inference
    model.eval()
    set_fast_inference_mode(model, False)
    
    standard_psnr_values = []
    
    print("Running validation with standard inference...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            # Get inputs based on available keys in the batch
            clean = batch['clean'].to(device)
            
            # Handle different dataset structures
            if 'noisy' in batch:
                degraded = batch['noisy'].to(device)
            elif 'blur' in batch:
                degraded = batch['blur'].to(device)
            elif 'rainy' in batch:
                degraded = batch['rainy'].to(device)
            else:
                continue
            
            # Forward pass
            output = model(degraded)
            
            # Calculate PSNR
            mse = torch.nn.functional.mse_loss(output, clean).item()
            psnr = -10 * torch.log10(torch.tensor(mse)).item()
            standard_psnr_values.append(psnr)
    
    # Fast inference
    set_fast_inference_mode(model, True)
    
    fast_psnr_values = []
    
    print("Running validation with fast inference...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            # Get inputs based on available keys in the batch
            clean = batch['clean'].to(device)
            
            # Handle different dataset structures
            if 'noisy' in batch:
                degraded = batch['noisy'].to(device)
            elif 'blur' in batch:
                degraded = batch['blur'].to(device)
            elif 'rainy' in batch:
                degraded = batch['rainy'].to(device)
            else:
                continue
            
            # Forward pass
            output = model(degraded)
            
            # Calculate PSNR
            mse = torch.nn.functional.mse_loss(output, clean).item()
            psnr = -10 * torch.log10(torch.tensor(mse)).item()
            fast_psnr_values.append(psnr)
    
    # Restore standard inference mode
    set_fast_inference_mode(model, False)
    
    # Calculate average PSNRs
    avg_standard_psnr = sum(standard_psnr_values) / len(standard_psnr_values)
    avg_fast_psnr = sum(fast_psnr_values) / len(fast_psnr_values)
    psnr_diff = avg_standard_psnr - avg_fast_psnr
    
    result['standard_psnr'] = avg_standard_psnr
    result['fast_psnr'] = avg_fast_psnr
    result['psnr_diff'] = psnr_diff
    
    # Print comparison
    print("\n----- Inference Accuracy Comparison -----")
    print(f"Standard inference PSNR: {avg_standard_psnr:.2f} dB")
    print(f"Fast inference PSNR:     {avg_fast_psnr:.2f} dB")
    print(f"PSNR difference:         {psnr_diff:.2f} dB")
    print("----------------------------------------\n")
    
    return result

# Usage example:
"""
Example usage:

from Models.stoformer2 import build_stoformer2
from Models.stoformer_fast_inf import set_fast_inference_mode, benchmark_inference

# Create model
model = build_stoformer2(img_size=256, window_size=8)
model = model.to(device)
model.eval()

# During validation or inference
set_fast_inference_mode(model, True)  # Enable fast inference mode
with torch.no_grad():
    output = model(input_tensor)  # Will use optimized inference

# To restore standard inference mode
set_fast_inference_mode(model, False)

# To benchmark the speedup
sample_input = torch.randn(1, 3, 256, 256).to(device)
results = benchmark_inference(model, sample_input)
print(f"Standard inference: {results['standard_inference_time']:.4f}s")
print(f"Fast inference: {results['fast_inference_time']:.4f}s")
print(f"Speedup: {results['speedup']:.2f}x")
""" 