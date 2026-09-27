import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, apply_rotary_pos_emb
from wick.model.layers import precompute_rope_freqs, apply_rope

torch.manual_seed(0)
head_dim = 64
seq_len = 5
num_heads = 14
theta = 1000000.0

q = torch.randn(1, num_heads, seq_len, head_dim)
k = torch.randn(1, 2, seq_len, head_dim)  # 2 KV heads
position_ids = torch.arange(seq_len).unsqueeze(0)

# Reference — load the real config from your downloaded model
config = AutoConfig.from_pretrained("models/qwen2.5-0.5b-instruct")
ref_rope = Qwen2RotaryEmbedding(config=config)
ref_cos, ref_sin = ref_rope(q, position_ids)
ref_q, ref_k = apply_rotary_pos_emb(q, k, ref_cos, ref_sin)

# Yours
my_cos, my_sin = precompute_rope_freqs(head_dim, seq_len, theta)
my_q = apply_rope(q, my_cos, my_sin)
my_k = apply_rope(k, my_cos, my_sin)

assert torch.allclose(ref_q, my_q, atol=1e-4), "Q mismatch!"
assert torch.allclose(ref_k, my_k, atol=1e-4), "K mismatch!"
print("✅ RoPE matches reference")