import torch
from transformers import AutoConfig, AutoModelForCausalLM
from wick.model.layers import Attention, precompute_rope_freqs

model_path = "models/qwen2.5-0.5b-instruct"
config = AutoConfig.from_pretrained(model_path)
hf_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

hf_layer0_attn = hf_model.model.layers[0].self_attn

torch.manual_seed(0)
seq_len = 5
x = torch.randn(1, seq_len, config.hidden_size)
position_ids = torch.arange(seq_len).unsqueeze(0)

from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
rope = Qwen2RotaryEmbedding(config=config)
ref_cos, ref_sin = rope(x, position_ids)
ref_out = hf_layer0_attn(x, attention_mask=None, position_embeddings=(ref_cos, ref_sin))[0]

my_attn = Attention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads)
my_attn.q_proj.weight.data.copy_(hf_layer0_attn.q_proj.weight.data)
my_attn.q_proj.bias.data.copy_(hf_layer0_attn.q_proj.bias.data)
my_attn.k_proj.weight.data.copy_(hf_layer0_attn.k_proj.weight.data)
my_attn.k_proj.bias.data.copy_(hf_layer0_attn.k_proj.bias.data)
my_attn.v_proj.weight.data.copy_(hf_layer0_attn.v_proj.weight.data)
my_attn.v_proj.bias.data.copy_(hf_layer0_attn.v_proj.bias.data)
my_attn.o_proj.weight.data.copy_(hf_layer0_attn.o_proj.weight.data)

rope_theta = config.rope_parameters["rope_theta"]
my_cos, my_sin = precompute_rope_freqs(config.hidden_size // config.num_attention_heads, seq_len, rope_theta)
my_out = my_attn(x, my_cos, my_sin)

assert torch.allclose(ref_out, my_out, atol=1e-4), f"Mismatch! Max diff: {(ref_out - my_out).abs().max()}"
print("✅ Attention matches reference")