import torch
import torch.nn as nn

class HuberLoss(nn.Module):
    """
    Huber Loss
    
    The Huber loss function is a combination of L1 and L2 loss. It behaves like L2 loss when the error is small, and like L1 loss when the error is large. It is less sensitive to outliers than L2 loss.
    
    Paper: "Robust Estimation with the Huber Loss"
    """
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        
    def forward(self, x, y):
        diff = torch.abs(x - y)
        # Use the Huber loss formulation
        loss = torch.mean(torch.where(diff < self.delta, 0.5 * diff ** 2, self.delta * (diff - 0.5 * self.delta)))
        return loss