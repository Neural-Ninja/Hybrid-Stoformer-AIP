import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Union, Tuple

def calc_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
        img1: First image (predicted, B x C x H x W)
        img2: Second image (ground truth, B x C x H x W)
        max_val: Maximum value of the images
        
    Returns:
        PSNR value
    """
    # MSE (Mean Square Error)
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 20 * math.log10(max_val / math.sqrt(mse))
    return psnr

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Create a Gaussian kernel
    
    Args:
        size: Size of the kernel
        sigma: Standard deviation
        
    Returns:
        Gaussian kernel
    """
    coords = torch.arange(size).to(torch.float32)
    coords -= (size - 1) / 2
    
    g = coords**2
    g = torch.exp(-(g / (2 * sigma**2)))
    
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)

def _create_window(window_size: int, num_channels: int) -> torch.Tensor:
    """
    Create a window for SSIM calculation
    
    Args:
        window_size: Size of the window
        num_channels: Number of channels
        
    Returns:
        Window tensor
    """
    _1D_window = _gaussian_kernel(window_size).squeeze(0)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(num_channels, 1, window_size, window_size).contiguous()
    return window

def calc_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True, full: bool = False) -> Union[float, Tuple[float, float]]:
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    
    Args:
        img1: First image (predicted, B x C x H x W)
        img2: Second image (ground truth, B x C x H x W)
        window_size: Size of the window for SSIM calculation
        size_average: Whether to average over all batches
        full: Whether to return SSIM and contrast sensitivity (CS)
        
    Returns:
        SSIM value or (SSIM, CS) if full=True
    """
    # Check input
    if not img1.shape == img2.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {img1.shape} and {img2.shape}")
    
    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Get device and channels
    device = img1.device
    num_channels = img1.size(1)
    
    # Create window
    window = _create_window(window_size, num_channels).to(device)
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=num_channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=num_channels)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate sigma squares
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=num_channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=num_channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=num_channels) - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Calculate CS (contrast sensitivity)
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    
    if size_average:
        ssim_val = ssim_map.mean().item()
        cs = cs_map.mean().item()
    else:
        ssim_val = ssim_map.mean(1).mean(1).mean(1).item()  # Average over HWC dimensions
        cs = cs_map.mean(1).mean(1).mean(1).item()
    
    if full:
        return ssim_val, cs
    
    return ssim_val 