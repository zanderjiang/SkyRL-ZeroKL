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
