"""Build a Megatron-core GPTModel using the LOCAL spec (no Transformer Engine) on the
torch-nightly venv and run a forward pass. Proves Priority 1: GPTModel builds + forwards
without TE on torch 2.14 nightly / cu130.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist

# --- single-process distributed init (required by megatron parallel_state) ---
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29555")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

torch.cuda.set_device(0)  # within CUDA_VISIBLE_DEVICES this is the first allowed GPU
dist.init_process_group(backend="nccl", world_size=1, rank=0)

from megatron.core import parallel_state
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
)

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

# Seed for reproducibility of init
torch.manual_seed(0)
from megatron.core import tensor_parallel
tensor_parallel.model_parallel_cuda_manual_seed(0)

HIDDEN = 256
HEADS = 8
LAYERS = 2
VOCAB = 1024
SEQ = 64
DTYPE = torch.bfloat16

config = TransformerConfig(
    num_layers=LAYERS,
    hidden_size=HIDDEN,
    num_attention_heads=HEADS,
    ffn_hidden_size=4 * HIDDEN,
    use_cpu_initialization=False,
    pipeline_dtype=DTYPE,
    bf16=True,
    params_dtype=DTYPE,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    sequence_parallel=False,
    add_bias_linear=False,
    gated_linear_unit=True,
    normalization="RMSNorm",
)

# LOCAL spec => no transformer-engine. Uses megatron's DotProductAttention (torch SDPA).
layer_spec = get_gpt_layer_local_spec(
    num_experts=None,
    moe_grouped_gemm=False,
    qk_layernorm=False,
    normalization="RMSNorm",
)
print("transformer_layer_spec module:", layer_spec.module.__name__)
# Show that self-attention core uses a local (non-TE) implementation
sa = layer_spec.submodules.self_attention
core_attn = sa.submodules.core_attention  # this is the class itself in local spec
print("self_attention.core_attention impl:", core_attn.__module__ + "." + core_attn.__name__)
linear_qkv = sa.submodules.linear_qkv
print("self_attention.linear_qkv impl:", linear_qkv.__module__ + "." + linear_qkv.__name__)

model = GPTModel(
    config=config,
    transformer_layer_spec=layer_spec,
    vocab_size=VOCAB,
    max_sequence_length=SEQ,
    pre_process=True,
    post_process=True,
    position_embedding_type="rope",
).cuda().to(DTYPE)

n_params = sum(p.numel() for p in model.parameters())
print(f"GPTModel built OK: {n_params/1e6:.2f}M params, dtype={DTYPE}")

# --- forward ---
torch.manual_seed(1)
input_ids = torch.randint(0, VOCAB, (1, SEQ), device="cuda")
position_ids = torch.arange(SEQ, device="cuda").unsqueeze(0)
# causal mask: megatron expects attention_mask shape [b,1,s,s] with True=masked, or None
attention_mask = torch.tril(torch.ones(SEQ, SEQ, device="cuda", dtype=torch.bool)).logical_not()
attention_mask = attention_mask.view(1, 1, SEQ, SEQ)

model.eval()
with torch.no_grad():
    logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

print("forward OK. logits shape:", tuple(logits.shape), "dtype:", logits.dtype)
print("logits finite:", torch.isfinite(logits).all().item())
print("logits[0,0,:5]:", logits[0, 0, :5].float().tolist())
print("PASS: Megatron GPTModel local-spec build + forward on torch-nightly succeeded.")

parallel_state.destroy_model_parallel()
dist.destroy_process_group()
