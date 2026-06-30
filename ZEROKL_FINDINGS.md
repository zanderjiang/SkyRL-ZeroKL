# SkyRL-ZeroKL — consolidated findings + zero-KL fix plan (2026-06-30)

## TL;DR
- The `rollout_train_logprobs_abs_diff` residual is **NOT** weight delivery and **NOT** the trainer
  forward. It is the **vLLM paged-attention DECODE diverging from a full-sequence forward (prefill)**,
  a gap that **grows with response length** (0 @64 tok, ~0.017 @256, is_ratio up to 1.45 @512).
- **The fix is proven:** torch-native `torch.nn.attention.varlen.varlen_attn_out` + `num_splits=1`
  gives **bitwise (0.000e+00) decode==prefill at all lengths** on this H100. It needs torch-nightly
  + vLLM-nightly (migration in progress) — `varlen_attn_out` is absent in torch 2.11.
- **OOM resolved:** run at 4 engines / DP4 (peak 297GB of 1999) + host-RAM watchdog. The 8-engine
  config OOM'd the headnode at the 2nd weight sync (Adam m/v allocate after step 0).

## What was ruled out (each verified, 1-GPU MiMo-7B repro)
1. **Weight sync** — bitwise correct: first sync reports `miss=[]`, `cum 255`,
   `RECEIVER==SENDER==89,866,863`. The handoff's "10.5M vs 89.8M" was a misread of the receiver's
   CUMULATIVE per-chunk checksum (10.5M = partial at chunk ~14/255).
2. **Chunked prefill** — off gives byte-identical logprobs.
3. **Trainer Megatron forward** — bitwise clean: `forward_backward_func` == direct unpadded
   bare-GPTModel forward == `from_parallel_logits_to_logprobs` extraction, all ~2e-6
   (`SKYRL_ZEROKL_FWD_PROBE=1`).
4. **num_splits=1 on vLLM 0.23's vendored FA3** — no-op (vLLM's FA3 was already 1-split; its
   decode/prefill paths differ structurally, unlike torch's `varlen_attn_out`).

## The fix, measured (scratchpad/varlen_decode_vs_prefill.py, torch 2.14 nightly, H100/FA3)
| L | auto-split max\|Δ\| | num_splits=1 max\|Δ\| |
|---|---|---|
| 256 | 0.000e+00 | 0.000e+00 |
| 512 | 0.000e+00 | 0.000e+00 |
| 1024 | 2.44e-04 | **0.000e+00** |
| 2048 | 2.44e-04 | **0.000e+00** |
vLLM's vendored FA3 drifts at 256 already (0.017); torch's `varlen_attn_out`+`num_splits=1` is bitwise.

## Reference: TorchTitan does exactly this
`torchtitan/experiments/rl/models/attention.py` — `PyTorchVarlenAttentionBackend` registers a vLLM
CUSTOM backend that calls `varlen_attn_out` for BOTH prefill and decode, with `num_splits=1` in
batch-invariant mode. Its vLLM-1.0 model wrapper (`vllm_wrapper.py`) has the SAME interface as our
`GPTModelVLLMWrapper` (forward/compute_logits/embed_input_ids/load_weights) — so the port is moderate.

## Migration plan (Track 2 — true zero-KL in the full pipeline)
Stack: torch 2.14-nightly cu130 + vLLM 1.0-nightly + flash-attn-3 + Megatron + TorchTitan varlen backend.
- torch-nightly + `varlen_attn_out` + flash-attn-3: installed & validated (isolated venv
  `/mnt/local_storage/zerokl-nightly-venv`).
- vLLM 1.0.0.dev cu130 (x86_64): resolves with `--index-strategy unsafe-best-match`.
- **Blocker:** TE 2.16 prebuilt is ABI-incompatible with this torch-nightly (`undefined symbol
  _ZN3c104impl3cow23materialize_cow_storage...`). Paths: (a) Megatron **local-spec (no TE)**
  [preferred]; (b) build TE from source vs torch-nightly (CUDA-13 toolkit).
- Then: register the varlen CUSTOM backend, port `GPTModelVLLMWrapper` to vLLM-1.0, keep native
  weight sync, validate `rollout_train ≈ 0`, run OOM-safe.

## Interim: full pipeline IS running now (working stack)
`examples/zerokl/run_zerokl_tune_mimo.sh` (4 engines, DP4, **TIS on**, watchdog): full DAPO pipeline
runs OOM-free (peak 297GB), TIS corrects the ~0.007 decode-vs-prefill gap. This is the working
"full pipeline running" milestone while the bitwise migration lands.

## Safety (headnode)
Single 8xH100 headnode; a host-RAM OOM crashes ray/gcs. ALWAYS run `scratchpad/ram_watchdog*.sh`
(kills the job sparing ray/gcs before the kernel OOM-killer). **Never `ray stop`.**
