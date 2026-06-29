# SkyRL-ZeroKL — implementation status

Isolated working copy of SkyRL with bitwise vLLM↔Megatron logprob parity ("zero KL").
All engine changes are **monkey patches** — nothing in the installed megatron-core / vLLM
packages is edited.

## Isolation
- Source: `/home/ray/default/SkyRL-ZeroKL` (copy of `SkyRL`).
- venv: `/home/ray/skyrl-zerokl-venv` (clone of `/home/ray/skyrl-venv`); `.venv` symlink +
  editable finder repointed so `import skyrl` resolves to THIS copy. The original SkyRL and
  its venv are untouched (verified).

## Done (this increment)
- **`skyrl/backends/skyrl_train/zerokl/`** monkey-patch package:
  - `megatron_patches.py`
    - `apply_rope_fp32_patch()` — `_apply_rotary_pos_emb_bshd` (covers `thd`) → fp32 rotate-half.
    - `apply_vops_rmsnorm_patch()` — overrides `BatchInvariantRMSNormFn.forward`: main norms
      (over hidden) → vLLM `fused_add_rms_norm` via a zero residual (exact); q/k head norms →
      vLLM Triton no-residual kernel. Original fp32 backward kept → safe during training.
    - `enable_megatron_batch_invariant()` — GEMM + log_softmax BIK (already bitwise).
    - `apply_megatron_zerokl_patches()` orchestrator; idempotent + reversible.
  - `vllm_patches.py` — `apply_vllm_zerokl_env()` (VLLM_BATCH_INVARIANT + NCCL/AOT pins),
    `zerokl_engine_arg_overrides()` (enforce_eager, prefix/chunked-prefill off),
    `zerokl_sampling_constraints()` (temperature=1.0, raw logprobs).
  - `_selftest.py`, `README.md`.
- **Env-gated wiring** (`SKYRL_ZERO_KL=1`):
  - `inference_engines/vllm/vllm_engine.py::setup_envvars_for_vllm` → `apply_vllm_zerokl_env()`.
  - `workers/megatron/megatron_worker.py::init_model` (after model build) →
    `apply_megatron_zerokl_patches()`.

## Verified (bitwise, through the REAL patched Megatron functions)
```
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m skyrl.backends.skyrl_train.zerokl._selftest
  [GEMM]    Megatron BIK == vLLM BIK                    : bitwise=True
  [RoPE]    patched Megatron == vLLM CUDA              : bitwise=True (0/1,048,576)
  [RMSNorm] main(hidden) == vLLM fused (5 draws)       : bitwise=True (0 total)
  [RMSNorm] q/k head(128) == vLLM Triton               : bitwise=True (0/262,144)
  [grad]    backward through patched norm finite        : True
```
(See repo-root `SkyRL-ZeroKL-EVALUATION.md` for the end-to-end Qwen3-4B study showing both
kernels unified ⇒ r ≡ 1.0.)

## R3 — real Megatron GPTModel forward vs vLLM (measured; residual driven down, not yet bitwise)
Built Qwen3-4B `GPTModel` via `megatron.bridge` (TP=1, TE-flash, batch_invariant_mode,
apply_rope_fusion=False) + patches; compared to real vLLM batch-invariant on 223 tokens.

Mean \|Δlogp\| progression as more components were unified:
- native Megatron BIK (no patches): **3.36e-3**, r∈[0.900,1.111]
- + fp32 RoPE + vLLM standalone-norm (q/k+final): **2.80e-3**
- + fused-LayerNormLinear main norms intercepted: **2.41e-3**, r∈[0.919,1.100]

Patch-fire verified: rope=72, standalone-rmsnorm=73 (q/k 72 + final 1), fused applynorm=72.

**Two key corrections to the earlier guess:**
1. **Attention is NOT the blocker.** Proven: TE-flash == vLLM vendored flash == flash_attn
   pkg, all **bitwise** (`zerokl_experiments/attn_probe.py`), and `batch_invariant_mode`
   forces flash. Set `NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0`.
2. **The big gap was the fused TE main norms.** They go through
   `transformer_engine.pytorch.module._common.apply_normalization` (imported by-name into
   `layernorm_linear`/`layernorm_mlp`), which Megatron's own BIK never patches. Now caught by
   `apply_te_fused_norm_patch()` under `scoring_mode()`.

**Per-op bisection (R2) DONE — residual is the bf16 noise floor, not a missing kernel.**
Built a faithful pure-torch vLLM forward (`faithful_vllm.py`). Three-way mean \|Δlogp\|:
faithful-vs-HF 3.17e-3, realvLLM-vs-HF 5.47e-3, faithful-vs-realvLLM 3.63e-3,
**Megatron-vs-realvLLM 2.41e-3** — all four bf16 forwards differ by the same 2–5e-3 band;
Megatron-patched is actually the *closest* to real vLLM. Per-layer divergence grows smoothly
(L0 2.6e-4 → L16 8e-2 → L32 0.30), no jump → distributed accumulation, attention-path
dominated. §5 proved: shared attention + unified kernels → bitwise 0.
=> **True bitwise zero-KL vs the vLLM engine is not reachable by kernel-patching alone**
(paged-attention prefill path isn't bit-reproducible from the trainer). Patches still cut the
floor 3.36→2.41e-3 (~28%) and tighten r. Paths to actual zero-KL: L3, L2-with-shared-attn, or
S4 (smaller floor + TIS).

Scripts: `zerokl_experiments/{r3_vllm_ref,r3_megatron,attn_probe}.py` (`ZK_PATCHES=on|off`,
`ZK_FLASH=1|cudnn`).
=> All individually-verified kernels + flash attention + fused norms unified and firing on the
real stack; residual reduced 3.36e-3→2.41e-3. **Bitwise needs per-op bisection (next step).**

## ✅ ZERO-KL RUN WORKING (unified Megatron-GPTModel route)
`zerokl_experiments/zerokl_rl_run.py` — a working GRPO run on Qwen3-4B (TP=1) where the rollout
engine and the trainer compute **bitwise-identical** logprobs:
- rollout = vLLM running Megatron GPTModel; behavior logprob (decode) == train/old logprob
  (vLLM-GPTModel prefill rescore) **== 0.00e+00 every step** (vLLM prefill==decode is bitwise).
- Megatron-trainer-forward (TE-flash, separate runtime) vs vLLM-GPTModel: ~1.5e-5 (near-bitwise;
  same model in both runtimes collapsed the old 2-4e-3 floor). is_ratio in [1.0, 1.00008].
- Loop: vLLM-GPTModel rollout -> GRPO advantage -> Megatron gradient -> SGD -> native (no-HF) sync.
Final integration fix: `_PositionIndexedRoPE` (GPTModel applies RoPE by sequence-index; vLLM
paged decode needs absolute positions) -> GPTModel-in-vLLM prefill==decode bitwise.
See ZEROKL_WORKLOG.md for the run table + reasoning.

## UNIFIED-MODEL ROUTE (retain Megatron: vLLM runs GPTModel too) — prerequisites PROVEN
Decision: both serving (vLLM) and training run Megatron's GPTModel (one model, two runtimes),
per TorchTitan's approach. All numerical prerequisites verified on this stack:
- **Native (no-HF) weight sync** (`zerokl/native_weight_sync.py`): build 2 GPTModels, sync
  A->B via `named_parameters()` (bf16, native layout) — 290/290 params bit-identical, forwards
  bitwise. No `export_hf_weights`, no vLLM HF-repack. (`zerokl_experiments/native_sync_test.py`)
- **vLLM prefill==decode parity** (`zerokl_experiments/prefill_decode_parity.py`): batch-invariant
  vLLM, generate 64 toks (decode) vs re-score (prefill) = **bitwise, 64/64**. Generation matches a
  full-seq trainer pass.
- TE-flash == vLLM vendored flash (full seq, identical inputs) — bitwise (`attn_probe.py`).
- GEMM / log_softmax / RMSNorm / RoPE bitwise under batch-invariant (zerokl patches).
- vLLM 0.20.2(dev) HAS the APIs to register a custom model+attention backend
  (`ModelRegistry.register_model`, `register_backend`, `get_attention_context`,
  `register_config_parser`); only `breakable_cudagraph` missing -> use `enforce_eager`.
- Dtype recipe satisfied: both emit fp32 logits before log_softmax (Megatron Float16Module +
  vLLM raw_logprobs), bf16 body, bf16 sync. lm_head GEMM is the same BIK matmul both sides.

=> Remaining is ENGINEERING: register GPTModel into vLLM so its per-op inputs match the trainer
(task #20). Env caveat: torch 2.11 lacks `varlen_attn_out`/FA3 (TorchTitan's torch-native paged
attn), so the generator uses vLLM's vendored flash backend (proven == TE-flash) instead.

### GPTModel-in-vLLM integration (task #20) — MECHANICALLY RUNS, correctness WIP
Module `zerokl/gptmodel_vllm.py`: `MegatronCoreAttnToVLLM` (swaps SelfAttention.core_attention
-> vLLM paged Attention), `GPTModelVLLMWrapper` (vLLM model interface over a bridge-built
GPTModel), `register_gptmodel_to_vllm` (ModelRegistry). Smoke test
(`zerokl_experiments/smoke_gptmodel_vllm.py`, run with `VLLM_ENABLE_V1_MULTIPROCESSING=0`):
- **MILESTONE: vLLM instantiates + runs Megatron GPTModel end-to-end** — engine builds,
  core_attention swapped on all 36 layers, KV cache allocated (234K tokens), prefill+decode
  generation loop executes. Bring-up fixes done: (1) add `embed_input_ids` (vLLM
  VllmModelForTextGeneration protocol); (2) `apply_megatron_zerokl_patches(skip_aten_registration
  =True)` so Megatron doesn't re-register the aten ops vLLM already registered (only TE
  GEMM/RMSNorm + RoPE monkey-patches).
- **PREFILL forward now CORRECT.** Two bug fixes: (a) attention layout — vLLM's `Attention`
  wants FLATTENED 2D `[num_tokens, np*hn]` (native Qwen3 passes that), not 3D `[T,np,hn]`;
  (b) fused-norm vops gate now also activates under `not is_grad_enabled()` so the vLLM
  generator (no_grad) uses vLLM-matched norms, not native TE. Result: GPTModel-in-vLLM prefill
  mean logprob **-0.1977** (sane), within **~2e-3 of vLLM-native** and **~2-4e-3 of the Megatron
  trainer GPTModel** (`zerokl_experiments/smoke_prefill_check.py`).
- **Not bitwise vs trainer yet (~2-4e-3).** Both run GPTModel, so the residual is the
  TWO-RUNTIMES attention path: trainer TE-flash full-seq vs generator vLLM-paged-flash, plus
  op-composition differences between Megatron's forward_backward_func and vLLM's engine. This is
  the same wall TorchTitan cleared by using ONE torch-native varlen kernel (paged==full proven
  bitwise) on BOTH sides; we have two kernels (env lacks torch varlen_attn_out/FA3).
- **DECODE (generation) still needs the RoPE-absolute-position fix** (Megatron applies RoPE by
  sequence-index; vLLM paged decode needs the token's absolute position).
- **Clean path to true bitwise (recommended):** L2 — have the vLLM-GPTModel compute BOTH the
  behavior logprob (generation/rescore) AND the trainer's "old" logprob (a vLLM prefill rescore).
  Same code => old==behavior bitwise by construction (TorchTitan's pattern). Megatron-core still
  owns the gradient/optimizer/checkpointing on the same GPTModel. This sidesteps the
  trainer-TE-flash vs generator-vLLM-flash mismatch entirely.
  Then: wire native sync (load_weights=False + push trainer params each step), GRPO with is_ratio==1.

## Remaining (task #12) — to actually run Qwen3 zero-KL end to end
1. **R3 model check:** build real Qwen3 GPTModel via the bridge (TE) with `SKYRL_ZERO_KL=1`
   and compare full-sequence logprobs to vLLM batch-invariant, TP=1 — confirm bitwise on a
   real forward (not just isolated kernels).
2. **L2 behavior logprobs:** route rollout behavior logprobs through a vLLM full-prefill
   rescore (so trainer and rollout score the same full-sequence shape; sidesteps the
   decode-vs-prefill attention divergence).
3. **Config + gates:** promote `SKYRL_ZERO_KL` env to a real `zero_kl` config flag; force
   `temperature=1.0`, `enforce_eager=True`, `tis_ratio_type=None`; ensure the trainer parity
   forward reads the synced bf16 weights (not the fp32 master).
4. **GRPO run:** run a Qwen3-4B GRPO step (colocated, TP=1) and assert
   `rollout_train_logprobs_abs_diff_max == 0`, `is_ratio ≡ 1`, ESS == N.

## Scope / preconditions
Qwen3 dense, `apply_rope_fusion=False`, TP=1 on the parity forward, no serving-side
quantization, same `(vLLM, Triton, Megatron)` triple both sides. `# EXTEND:` markers in
`megatron_patches.py` flag where MoE (grouped-GEMM/router parity) and MLA (interleaved RoPE)
support will go.
```
