"""vLLM-side configuration for SkyRL-ZeroKL.

vLLM already ships the batch-invariant kernels; enabling them is environment + engine-arg
configuration, not source edits. This module centralizes the env that must be set
**before** the vLLM engine (and its worker processes) are created, plus the engine-arg
overrides the caller should apply.

Reference (vLLM 0.20.2): init_batch_invariance() (gpu_worker) reads VLLM_BATCH_INVARIANT
and pins NCCL to a deterministic tree all-reduce, turns off split-K / custom all-reduce /
AOT compile, and forces FlashAttention with max_num_splits=1. enforce_eager is required
(batch invariance is incompatible with the captured CUDA graphs)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Env that must be present in the vLLM worker process environment before import/init.
ZEROKL_VLLM_ENV = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_USE_AOT_COMPILE": "0",
    # Deterministic, degree-stable NCCL all-reduce (matches the trainer pin).
    "NCCL_ALGO": "allreduce:tree",
    "NCCL_MIN_NCHANNELS": "1",
    "NCCL_MAX_NCHANNELS": "1",
}


def apply_vllm_zerokl_env(env: dict | None = None) -> dict:
    """Set the zero-KL env vars in-process. Must run before vLLM init.

    Returns the dict of vars set so callers can also forward them to Ray worker runtime
    envs / vLLM subprocesses (colocated workers inherit the driver env; for spawned
    engines pass these through the engine's env_vars/runtime_env)."""
    target = os.environ if env is None else env
    for k, v in ZEROKL_VLLM_ENV.items():
        target[k] = v
    logger.info("[zerokl] set vLLM batch-invariant env: %s", ZEROKL_VLLM_ENV)
    return dict(ZEROKL_VLLM_ENV)


def apply_flash_num_splits_patch() -> bool:
    """Force ``num_splits=1`` in vLLM's MAIN flash-attention forward under batch-invariant mode.

    THE zero-KL fix for long responses. vLLM pins ``num_splits=1`` (deterministic single-pass KV
    reduction == prefill) for the CASCADE attention path (flash_attn.py ~L1194/L1219) but NOT the
    MAIN non-cascade path (~L796), which only forwards ``scheduler_metadata`` -- and that is ``None``
    under batch-invariant (``aot_schedule`` is force-disabled at ~L411). So with prefix caching off
    (our config -> non-cascade) the paged DECODE uses FA's auto split-KV heuristic: 1 split for short
    KV (decode==prefill bitwise) but >1 split for long KV (decode != prefill -> the rollout_train
    residual that grows with response length: ~0 @64 tok, ~0.017 @256). Mirrors TorchTitan's
    ``num_splits=1``-in-batch-invariant-mode (rl/models/attention.py) but works WITHOUT torch's
    ``varlen_attn_out`` (absent in torch 2.11) by pinning vLLM's vendored kernel directly.

    Wraps the module-global ``flash_attn_varlen_func`` to inject ``num_splits=1`` when it is not
    already specified and VLLM_BATCH_INVARIANT=1. Idempotent. Returns True if applied."""
    try:
        import vllm.v1.attention.backends.flash_attn as _fa
    except Exception as e:  # pragma: no cover
        logger.warning("[zerokl] flash num_splits patch: cannot import flash_attn backend: %s", e)
        return False
    if getattr(_fa, "_zerokl_num_splits_patched", False):
        return True
    _orig = _fa.flash_attn_varlen_func

    def _wrapped(*args, **kwargs):
        if os.environ.get("VLLM_BATCH_INVARIANT") == "1" and "num_splits" not in kwargs:
            kwargs["num_splits"] = 1
        return _orig(*args, **kwargs)

    _wrapped._zerokl_orig = _orig
    _fa.flash_attn_varlen_func = _wrapped
    _fa._zerokl_num_splits_patched = True
    logger.info("[zerokl] patched flash_attn_varlen_func -> num_splits=1 (main decode path, batch-invariant)")
    print("[ZEROKL-ATTN] flash_attn_varlen_func pinned to num_splits=1 (decode==prefill fix)", flush=True)
    return True


def zerokl_engine_arg_overrides() -> dict:
    """Engine-arg overrides the caller must merge into the vLLM engine config.

    These cannot be set via env; the SkyRL InferenceEngineConfig / engine kwargs must
    carry them. enforce_eager is mandatory under batch invariance.
    """
    return {
        "enforce_eager": True,
        # Prefix caching reuses KV computed in a DIFFERENT batch/context than the trainer's clean
        # single-sequence forward -> the rollout (decode) logprobs drift ~0.01 from a recompute even
        # though VLLM_BATCH_INVARIANT makes the kernels deterministic. THIS is the dominant cause of
        # the zero-KL residual; it must be off (samples in a DAPO group share a prompt prefix).
        "enable_prefix_caching": False,
        # NOTE: chunked prefill is intentionally NOT forced off here. vLLM's batch-invariant mode
        # already makes a query attend to a chunk-split KV cache invariantly, and disabling it
        # requires max_num_batched_tokens >= max_model_len (else vLLM rejects), which in turn forces
        # a full-context KV reservation that OOMs a colocated engine. If a residual above the ~1e-5
        # floor remains for very long sequences, bound max_model_len = max_prompt+max_response and
        # set enable_chunked_prefill=False together.
    }


def zerokl_sampling_constraints() -> dict:
    """Sampling-side constraints for the MVP zero-KL target.

    temperature=1.0 makes vLLM raw_logprobs (no temperature) coincide with the trainer's
    logits.div_(temperature) convention. logprobs must be raw (untempered)."""
    return {"temperature": 1.0, "logprobs_mode": "raw_logprobs"}
