import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1)
    
    This loss is commonly used in image restoration tasks and is a differentiable
    variant of L1 loss. It's more robust to outliers compared to L2 loss, but still
    ensures smooth gradients near zero.
    
    Paper: "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution"
    - https://arxiv.org/abs/1704.03915
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        
    def forward(self, x, y):
        diff = x - y
        # Use the true Charbonnier loss formulation: sqrt(x^2 + eps^2)
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss 