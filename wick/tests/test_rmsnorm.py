import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from wick.model.layers import RMSNorm

torch.manual_seed(0)
hidden_size = 896
x = torch.randn(1, 5, hidden_size)

ref_norm = Qwen2RMSNorm(hidden_size, eps=1e-6)
ref_out = ref_norm(x)

my_norm = RMSNorm(hidden_size, eps=1e-6)
my_norm.weight.data.copy_(ref_norm.weight.data)
my_out = my_norm(x)

assert torch.allclose(ref_out, my_out, atol=1e-5), "mismatch"
print("RMSNorm matches reference")