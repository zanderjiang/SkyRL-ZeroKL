"""Grad-capable flash attention for the Megatron TRAINER, matching the vLLM rollout kernel.

To get the rollout<->train logprob diff to EXACTLY 0 (not ~1e-4), the trainer's attention must
be the SAME kernel as the rollout's. The rollout (vLLM-GPTModel) uses vLLM's vendored flash;
the `flash_attn` package is bitwise-identical to it (verified) and is grad-capable. This module
swaps Megatron `SelfAttention.core_attention` with a flash_attn_varlen_func call (full-seq,
causal, num_splits-stable) so the trainer forward == the rollout forward bit-for-bit.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class FlashVarlenCoreAttn(nn.Module):
    """Drop-in for SelfAttention.core_attention using flash_attn (== vLLM vendored flash)."""

    def __init__(self, *, num_heads, num_kv_heads, head_dim, scale):
        super().__init__()
        from flash_attn import flash_attn_varlen_func
        self._fa = flash_attn_varlen_func
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        # Megatron sbhd: q [sq, b, np, hn]; trainer scoring forward uses b==1.
        sq, b = query.shape[0], query.shape[1]
        assert b == 1, "FlashVarlenCoreAttn supports the b=1 scoring/parity forward"
        q = query.reshape(sq, self.num_heads, self.head_dim)
        k = key.reshape(sq, self.num_kv_heads, self.head_dim)
        v = value.reshape(sq, self.num_kv_heads, self.head_dim)
        cu = torch.tensor([0, sq], device=q.device, dtype=torch.int32)
        out = self._fa(q, k, v, cu, cu, sq, sq, softmax_scale=self.scale,
                       causal=True, deterministic=True)        # [sq, np, hn], grad-capable
        return out.reshape(sq, b, self.num_heads * self.head_dim)


def swap_trainer_core_attention_flash(gpt_modules):
    """Replace each decoder layer's core_attention with flash_attn (trainer side)."""
    modules = gpt_modules if isinstance(gpt_modules, (list, tuple)) else [gpt_modules]
    n = 0
    cfg = None
    for m in modules:
        # fully unwrap DDP(Float16Module(GPTModel)) -> the GPTModel (the one with .decoder)
        inner = m
        for _ in range(4):
            if hasattr(inner, "decoder"):
                break
            if hasattr(inner, "module"):
                inner = inner.module
            else:
                break
        cfg = inner.config
        head_dim = getattr(cfg, "kv_channels", cfg.hidden_size // cfg.num_attention_heads)
        for layer in inner.decoder.layers:
            sa = getattr(layer, "self_attention", None)
            if sa is None:
                continue
            sa.core_attention = FlashVarlenCoreAttn(
                num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_query_groups,
                head_dim=head_dim, scale=head_dim ** -0.5)
            n += 1
    logger.info("[zerokl] swapped TRAINER core_attention -> flash_attn (== vLLM flash) on %d layers", n)
    return n
