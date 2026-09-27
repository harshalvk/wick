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
    
def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 1000000.0):
  freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float()/head_dim))
  positions = torch.arange(max_seq_len).float()
  angles = torch.outer(positions, freqs)
  cos = torch.cos(angles)
  sin = torch.sin(angles)
  return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
  x1, x2 = x.chunk(2, dim=-1)
  return torch.cat((-x2, x1), dim=-1)
  
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  cos = torch.cat((cos, cos), dim=-1)
  sin = torch.cat((sin, sin), dim=-1)
  cos = cos.unsqueeze(0).unsqueeze(0)
  sin = sin.unsqueeze(0).unsqueeze(0)
  return (x * cos) + (rotate_half(x) * sin)
  
  