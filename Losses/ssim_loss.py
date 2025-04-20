import torch
import torch.nn as nn
import torch.functional as F

class SSIMLoss(nn.Module):
    """
    SSIM Loss (Structural Similarity Index)
    
    The SSIM loss is designed to measure the perceived similarity between two images. It considers luminance, contrast, and structure to compare images, and is widely used in image restoration tasks.
    
    Paper: "Image Quality Assessment: From Error Visibility to Structural Similarity"
    - https://ieeexplore.ieee.org/document/1284395
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.window = self.create_window(window_size)

    def create_window(self, window_size):
        """Create a Gaussian window for SSIM calculation."""
        gauss = torch.Tensor([torch.exp(-x ** 2 / (2.0 * 1.5 ** 2)) for x in range(-window_size // 2 + 1, window_size // 2 + 1)])
        window = gauss / gauss.sum()
        return window.view(1, 1, window_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def gaussian_blur(self, x, window):
        """Apply Gaussian blur to the input tensor."""
        return F.conv2d(x, window, padding=self.window_size // 2, groups=x.size(1))

    def forward(self, x, y):
        """Compute SSIM loss between x and y."""
        mu_x = self.gaussian_blur(x, self.window)
        mu_y = self.gaussian_blur(y, self.window)
        
        sigma_x = self.gaussian_blur(x * x, self.window) - mu_x * mu_x
        sigma_y = self.gaussian_blur(y * y, self.window) - mu_y * mu_y
        sigma_xy = self.gaussian_blur(x * y, self.window) - mu_x * mu_y
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # SSIM Index
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        
        # Convert to loss: 1 - SSIM (since higher SSIM is better, but we want to minimize the loss)
        ssim_loss = 1 - torch.mean(ssim)
        
        return ssim_loss