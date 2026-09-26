import torch
import torch.nn as nn

class RMSNorm(nn.Module): 
  def __init__(self, hidden_size: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor: 
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + self.eps)
    return x_normed * self.weight