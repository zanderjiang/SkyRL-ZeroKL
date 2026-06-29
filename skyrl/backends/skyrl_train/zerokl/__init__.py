"""SkyRL-ZeroKL: bitwise-identical token logprobs between the vLLM rollout engine and the
Megatron trainer (importance ratio r_t == 1 exactly, not "small").

The numerical work reduces to two kernel patches on the Megatron side (RoPE -> fp32,
RMSNorm -> vLLM C++ kernel) plus enabling batch invariance on both engines; GEMM and
log_softmax already agree bitwise. See ``megatron_patches.py`` and the project report
``SkyRL-ZeroKL-EVALUATION.md`` for the grounding experiments.

Typical wiring:
  * vLLM worker init (before engine creation):
        from skyrl.backends.skyrl_train.zerokl import apply_vllm_zerokl_env
        apply_vllm_zerokl_env()
    and merge ``zerokl_engine_arg_overrides()`` into the engine config.
  * Megatron worker init (after model build, before first scoring forward):
        from skyrl.backends.skyrl_train.zerokl import apply_megatron_zerokl_patches
        apply_megatron_zerokl_patches()

Scope: Qwen3 dense. MoE / hybrid / MLA extension points are marked ``# EXTEND:``.
"""

from .megatron_patches import (
    apply_megatron_zerokl_patches,
    revert_megatron_zerokl_patches,
    enable_megatron_batch_invariant,
    apply_rope_fp32_patch,
    apply_vops_rmsnorm_patch,
    scoring_mode,
    zerokl_patch_status,
)
from .vllm_patches import (
    apply_vllm_zerokl_env,
    zerokl_engine_arg_overrides,
    zerokl_sampling_constraints,
    ZEROKL_VLLM_ENV,
)

__all__ = [
    "apply_megatron_zerokl_patches",
    "revert_megatron_zerokl_patches",
    "enable_megatron_batch_invariant",
    "apply_rope_fp32_patch",
    "apply_vops_rmsnorm_patch",
    "scoring_mode",
    "zerokl_patch_status",
    "apply_vllm_zerokl_env",
    "zerokl_engine_arg_overrides",
    "zerokl_sampling_constraints",
    "ZEROKL_VLLM_ENV",
]
