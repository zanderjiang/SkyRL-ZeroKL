# SkyRL-ZeroKL kernel patches

Make the **Megatron trainer** and the **vLLM rollout engine** compute **bitwise-identical**
token logprobs, so the PPO/GRPO importance ratio `r_t = π_train/π_rollout ≡ 1` exactly.

## Why this is small

Across vLLM's and Megatron's batch-invariant kernels, measured on real Qwen3-4B
(`SkyRL-ZeroKL-EVALUATION.md`):

| Op | vLLM ↔ Megatron | Action |
|---|---|---|
| GEMM (`matmul_persistent`) | already **bitwise** | enable `batch_invariant_mode` |
| log_softmax | already **bitwise** | enable `batch_invariant_mode` |
| RMSNorm | 1 ULP off | route forward to `vops.rms_norm` (this package) |
| RoPE | bf16 vs fp32 arith | fp32 multiply-add (this package) |

End-to-end on 36-layer Qwen3-4B, unifying **both** kernels → logits bitwise-identical,
`r ≡ 1.0`. Unifying only one buys nothing.

## What this package does (monkey patches only)

`megatron_patches.py`
- `enable_megatron_batch_invariant()` — turns on Megatron BIK (GEMM + log_softmax + TE patches).
- `apply_vops_rmsnorm_patch()` — overrides `BatchInvariantRMSNormFn.forward` to emit vLLM
  C++ `rms_norm` bits; keeps the original fp32 **backward** so training gradients are
  unchanged (safe to leave on). Works because Megatron's residual add is a separate bf16
  add, bitwise-identical to vLLM's fused add.
- `apply_rope_fp32_patch()` — patches `_apply_rotary_pos_emb_bshd` (covers `thd` too) to do
  the rotate-half multiply-add in fp32 with bf16 cos/sin, matching vLLM's CUDA kernel.
- `apply_megatron_zerokl_patches()` — all of the above, idempotent.

`vllm_patches.py`
- `apply_vllm_zerokl_env()` — sets `VLLM_BATCH_INVARIANT=1` + NCCL/AOT pins before engine init.
- `zerokl_engine_arg_overrides()` — `enforce_eager=True`, prefix-caching/chunked-prefill off.
- `zerokl_sampling_constraints()` — `temperature=1.0`, raw logprobs.

## Verify

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m skyrl.backends.skyrl_train.zerokl._selftest
```

## Preconditions / scope

- Qwen3 **dense** (standard attention, non-MoE, non-hybrid). `# EXTEND:` marks where MoE
  (grouped-GEMM/router parity) and MLA (interleaved RoPE) hooks go.
- `apply_rope_fusion=False` (SkyRL RL path already sets this) — the fused TE RoPE kernel is
  not intercepted.
- TP=1 on the parity/scoring forward for the MVP (TP all-reduce order is a separate
  systemic axis; the training gradient may still use higher TP).
- Same `(vLLM, Triton, Megatron)` version triple on both sides — the GEMM identity requires
  the same Triton version (automatic when both import from one venv).

## Still required for *engine-vs-engine* zero-KL (beyond these two kernels)

These are systemic and tracked separately (see report §6, §8): attention prefill-vs-decode
shape (use L2 full-prefill rescore), TP all-reduce order (use TP=1 parity forward), and
serving precision (no quantization; parity forward must read the synced bf16 weights).
