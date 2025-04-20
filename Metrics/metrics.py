import torch
import numpy as np
from typing import Dict, List, Optional, Union

from Metrics.psnr_ssim import calc_psnr, calc_ssim

class Metrics:
    """
    Class to calculate and track image restoration metrics
    """
    def __init__(self):
        self.psnr_values = []
        self.ssim_values = []
        self.task_specific_metrics = {}
    
    def reset(self):
        """Reset all metrics"""
        self.psnr_values = []
        self.ssim_values = []
        self.task_specific_metrics = {}
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, task: Optional[str] = None, **kwargs):
        """
        Update metrics with a new batch of predictions and targets
        
        Args:
            pred: Predicted images (B x C x H x W)
            target: Ground truth images (B x C x H x W)
            task: Specific task (e.g., 'denoising', 'deraining', 'deblurring')
            **kwargs: Additional task-specific parameters
        """
        with torch.no_grad():
            # Calculate PSNR
            psnr = calc_psnr(pred, target)
            self.psnr_values.append(psnr)
            
            # Calculate SSIM
            ssim = calc_ssim(pred, target)
            self.ssim_values.append(ssim)
            
            # Task-specific metrics
            if task == 'denoising':
                # For denoising, we might want to track metrics per noise level
                if 'sigma' in kwargs:
                    sigma = kwargs['sigma']
                    if sigma not in self.task_specific_metrics:
                        self.task_specific_metrics[sigma] = {'psnr': [], 'ssim': []}
                    self.task_specific_metrics[sigma]['psnr'].append(psnr)
                    self.task_specific_metrics[sigma]['ssim'].append(ssim)
    
    def compute(self) -> Dict[str, Union[float, Dict]]:
        """
        Compute average metrics
        
        Returns:
            Dictionary with all computed metrics
        """
        results = {
            'psnr': np.mean(self.psnr_values) if self.psnr_values else 0.0,
            'ssim': np.mean(self.ssim_values) if self.ssim_values else 0.0
        }
        
        # Add task-specific metrics
        if self.task_specific_metrics:
            task_results = {}
            for key, values in self.task_specific_metrics.items():
                task_results[key] = {
                    'psnr': np.mean(values['psnr']) if values['psnr'] else 0.0,
                    'ssim': np.mean(values['ssim']) if values['ssim'] else 0.0
                }
            results['task_specific'] = task_results
        
        return results
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get the most recent metric values
        
        Returns:
            Dictionary with the most recent metric values
        """
        return {
            'psnr': self.psnr_values[-1] if self.psnr_values else 0.0,
            'ssim': self.ssim_values[-1] if self.ssim_values else 0.0
        }
    
    def __str__(self) -> str:
        """String representation of the metrics"""
        results = self.compute()
        return f"PSNR: {results['psnr']:.2f} dB, SSIM: {results['ssim']:.4f}" 