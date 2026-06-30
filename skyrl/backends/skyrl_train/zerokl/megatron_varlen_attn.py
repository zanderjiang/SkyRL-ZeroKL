"""Grad-capable trainer attention that is BITWISE-identical to the zero-KL engine kernel.

The rollout engine (gptmodel_vllm + zerokl/varlen_backend) computes attention with
`torch.nn.attention.varlen.varlen_attn(..., num_splits=1, window_size=(-1, 0))` -- the single-split,
causal (unlimited-left / zero-right window) PyTorch varlen FlashAttention. To drive
`minibatch_rollout_logprobs_abs_diff` to EXACTLY 0 (not ~1e-3 with huge per-token outliers), the
Megatron TRAINER's `SelfAttention.core_attention` must call the SAME function with the SAME args.
torch SDPA (the local-spec default) and `flash_attn` (the production swap) are DIFFERENT kernels and
leave occasional catastrophic per-token logprob outliers (max ~10-17) -- that is the "KL is very high"
symptom even though the mean is tiny.

Verified: `varlen_attn(..., window_size=(-1,0))` matches SDPA-causal (causality correct) and supports
autograd. b==1 per micro-forward (micro_*_batch_size_per_gpu=1); right-padding after the real tokens is
harmless under causal attention.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class TorchVarlenCoreAttn(nn.Module):
    """Drop-in for ``SelfAttention.core_attention`` using torch ``varlen_attn`` == the engine kernel."""

    def __init__(self, *, num_heads, num_kv_heads, head_dim, scale):
        super().__init__()
        import torch.nn.attention.varlen as _V  # noqa: N814

        self._varlen_attn = _V.varlen_attn
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.enable_gqa = num_heads > num_kv_heads

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        # Megatron sbhd: q [sq, b, np, hn]; the scoring/training micro-forward uses b==1.
        sq, b = query.shape[0], query.shape[1]
        assert b == 1, "TorchVarlenCoreAttn supports the b=1 micro-forward (micro_*_batch_size_per_gpu=1)"
        q = query.reshape(sq, self.num_heads, self.head_dim)
        k = key.reshape(sq, self.num_kv_heads, self.head_dim)
        v = value.reshape(sq, self.num_kv_heads, self.head_dim)
        cu = torch.tensor([0, sq], device=q.device, dtype=torch.int32)
        out = self._varlen_attn(
            q, k, v, cu, cu, sq, sq,
            scale=self.scale,
            num_splits=1,             # single KV-reduction split -> bitwise == engine prefill/decode
            enable_gqa=self.enable_gqa,
            window_size=(-1, 0),      # unlimited left, zero right == causal (the engine's recipe)
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(sq, b, self.num_heads * self.head_dim)


def enable_trainer_batch_invariant():
    """Enable the SAME vLLM batch-invariant aten ops the engine runs under VLLM_BATCH_INVARIANT
    (mm/addmm/matmul/linear/_log_softmax/mean.dim/rms_norm), so the trainer's NON-attention ops are
    bitwise-identical to the rollout. Without this the trainer GEMM/RMSNorm/logits use ordinary
    (batch-variant) kernels and leave a small residual even after the attention kernel is matched.
    Uses vLLM's implementation (not megatron-core's) so trainer and engine share the exact same kernels.
    Idempotent (vLLM guards with a module-global flag)."""
    try:
        from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode
    except Exception as e:  # pragma: no cover
        logger.warning("[zerokl] vLLM batch_invariant unavailable, trainer non-attn not batch-invariant: %s", e)
        return False
    enable_batch_invariant_mode()
    print("[ZEROKL-TRAINER] enabled vLLM batch-invariant aten ops (mm/addmm/linear/log_softmax/mean) "
          "== engine -> bitwise non-attention", flush=True)
    return True


def swap_trainer_core_attention_varlen(gpt_modules):
    """Replace each decoder layer's core_attention with the torch-varlen kernel (== rollout engine)."""
    modules = gpt_modules if isinstance(gpt_modules, (list, tuple)) else [gpt_modules]
    n = 0
    for m in modules:
        inner = m
        for _ in range(4):  # unwrap DDP(Float16Module(GPTModel)) -> GPTModel (the one with .decoder)
            if hasattr(inner, "decoder"):
                break
            inner = getattr(inner, "module", inner)
        if not hasattr(inner, "decoder"):
            continue
        cfg = inner.config
        head_dim = getattr(cfg, "kv_channels", cfg.hidden_size // cfg.num_attention_heads)
        for layer in inner.decoder.layers:
            sa = getattr(layer, "self_attention", None)
            if sa is None:
                continue
            sa.core_attention = TorchVarlenCoreAttn(
                num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_query_groups,
                head_dim=head_dim, scale=head_dim ** -0.5)
            n += 1
    logger.info("[zerokl] swapped TRAINER core_attention -> torch varlen_attn (== engine kernel) on %d layers", n)
    print(f"[ZEROKL-TRAINER] swapped core_attention -> torch varlen_attn num_splits=1 window=(-1,0) "
          f"(bitwise == engine) on {n} layers", flush=True)
    return n
