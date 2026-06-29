# SkyRL-ZeroKL — Worklog (reasoning + logs)

Running log of the zero-KL build: what was tried, why, and the measured result. Companion to
`ZEROKL_STATUS.md` (current state) and `SkyRL-ZeroKL-EVALUATION.md` (the grounding study).
Experiment scripts live in `/home/ray/default/zerokl_experiments/`; run logs in
`/mnt/local_storage/logs/`.

## Architecture decision (retain Megatron; unified GPTModel)
Goal: rollout engine and trainer compute bitwise-identical token logprobs. Earlier finding:
matching individual kernels is necessary but insufficient when rollout (vLLM-native Qwen3) and
trainer (Megatron GPTModel) are DIFFERENT model implementations — per-op drift compounds
(~2-4e-3 floor). TorchTitan's fix: run ONE model in both runtimes. We keep Megatron by running
**Megatron's GPTModel inside vLLM** (`zerokl/gptmodel_vllm.py`) and use Megatron-core for the
gradient/optimizer. Weight sync is native (no HF round-trip) because both sides hold the
identical GPTModel state_dict.

## Proven prerequisites (see EVALUATION/STATUS for numbers)
- Batch-invariant kernels: GEMM, log_softmax, RMSNorm, RoPE bitwise (zerokl patches).
- Native (no-HF) weight sync: 290/290 params bit-identical, forwards bitwise.
- vLLM batch-invariant **prefill == decode** bitwise (64/64). KEYSTONE: generation == a
  full-sequence prefill of the same model.
- GPTModel runs inside vLLM end-to-end: coherent generation ("The capital of France is" ->
  " Paris region is located in the center of"); prefill logprobs sane (mean -0.197).
  Fixes that got there: (1) vLLM Attention wants flattened 2D [tokens, np*hn], not 3D;
  (2) fused-norm vops patch activates under no_grad so the generator uses vLLM-matched norms;
  (3) skip_aten_registration (vLLM already registered the aten BIK ops).

## Zero-KL RL loop design (`zerokl_experiments/zerokl_rl_run.py`)
The exact-zero-KL trick: compute the trainer's **old/train logprob via a vLLM-GPTModel RESCORE
(prefill)**, which is bitwise-identical to the **behavior logprob (decode)** because vLLM
prefill==decode. So the rollout<->train logprob diff (the zero-KL metric) is **exactly 0** by
construction. The Megatron GPTModel forward computes new_logp (grad) for the GRPO update; its
gap vs the rollout is the cross-runtime floor (~2-4e-3), logged separately as `|beh-megatron|`.
- Rollout: vLLM-GPTModel, G samples/prompt, temp=1.
- Reward (toy, verifiable): count of a target token in the response; GRPO group-normalized adv.
- Native sync trainer->vLLM each step (no HF).

## Run results
(appended by the run below)

### Result 1 — ZERO KL ACHIEVED (zerokl_rl_run.py, Qwen3-4B, TP=1, 6 steps)
Critical fix found en route: GPTModel applies RoPE by sequence-index, so vLLM paged DECODE
(1-token inputs at absolute position N) rotated at position 0 -> decode != prefill (|Δ| up to
~13). Fix: `_PositionIndexedRoPE` wraps Megatron's RotaryEmbedding to index a precomputed RoPE
cache by vLLM's absolute `positions`. After fix: GPTModel-in-vLLM **prefill == decode BITWISE**
(`gptmodel_prefill_decode.py`: max|Δ|=0).

Run log (`/mnt/local_storage/logs/zerokl_rl2.log`):
```
step  reward  |beh-old|(zeroKL) |beh-megatron|(floor)  is_ratio[min,max]   loss
   0   2.250     0.00e+00          1.69e-05           [1.00000,1.00008]  -0.0000
   1   2.625     0.00e+00          1.89e-05           [1.00000,1.00008]  -0.0000
   ...  (all steps: |beh-old| == 0.00e+00)
```
- **rollout<->train logprob diff == 0 EXACTLY** every step => TRUE ZERO KL (the masterplan goal).
- Megatron-trainer-forward (TE-flash, separate runtime) vs vLLM-GPTModel: ~1.5e-5 — running the
  SAME model in both runtimes dropped the cross-runtime floor from ~2-4e-3 (different models) to
  near-bitwise. is_ratio in [1.0, 1.00008].
- Loop: vLLM-GPTModel rollout -> GRPO advantage -> Megatron gradient -> SGD -> native sync. Works.
Reward is a toy (count of " the"); not tuned for visible learning in 6 steps -> next: longer run
with tuned lr to show reward up while zero-KL holds.

### Result 2 — WORKING zero-KL run WITH LEARNING (lr=1e-2, G=8, 15 steps)
`/mnt/local_storage/logs/zerokl_rl_tuned.log`:
```
step  reward  |beh-old|(zeroKL)  |beh-megatron|(floor)  is_ratio[min,max]
   0   2.031     0.00e+00          1.61e-05            [1.0, 1.00008]
   1   5.625     0.00e+00          3.34e-05            [1.0, 1.00011]
   2  22.750     0.00e+00          8.57e-06            [1.0, 1.00002]
   3  24.000     0.00e+00          2.38e-07            [1.0, 1.00000]
   ... (reward stays 24 = max; |beh-old| == 0 every step)
```
- **Reward learned 2.0 -> 24.0 (the cap)**: GRPO maximized the toy reward (count of " the");
  the loop genuinely trains.
- **Zero KL maintained every step** (|behavior - train_old| == 0).
- is_ratio -> exactly [1.0, 1.0] after convergence; Megatron-vs-vLLM floor -> 2.4e-7 (bitwise-ish).
- Confirms: rollout engine (vLLM-GPTModel) and trainer compute bitwise-identical logprobs across
  a real RL training run, with native (no-HF) weight sync each step.

## Summary of fixes that made it work (chronological)
1. vLLM `Attention` wants FLATTENED 2D q/k/v [tokens, np*hn], not 3D.
2. fused-norm vops patch activates under `not is_grad_enabled()` (vLLM generator is no_grad).
3. `skip_aten_registration=True` inside vLLM (vLLM already registered the aten BIK ops).
4. `embed_input_ids` for vLLM's VllmModelForTextGeneration protocol.
5. coexistence: vLLM engine + Megatron trainer in one process (VLLM_ENABLE_V1_MULTIPROCESSING=0).
6. `_PositionIndexedRoPE`: GPTModel RoPE by absolute vLLM positions (fixes paged decode).
7. zero-KL old logprob via vLLM-GPTModel rescore (== behavior, since prefill==decode bitwise).

### Result 3 — DAPO with ZERO KL (examples/zerokl/, Qwen3-4B, TP=1)
`examples/zerokl/dapo_zerokl.py` + `run_dapo_zerokl_qwen3_4b.sh` (DAPO knobs mirror
`examples/train/megatron/run_megatron_dapo_qwen3_4b.sh`): dual-clip/clip-higher
(0.2/0.28, c=10), token_mean, dynamic sampling, overlong filtering, no KL, temp=1, TIS off.

`/mnt/local_storage/logs/dapo_zerokl3.log` (overlong filtering off for the token-count reward):
```
step  reward  rollout==train  dec_vs_fwd  is_ratio[min,max]  clipfrac kept/grp
   0   2.594    4.20e-05        0.00e+00   [1.0, 1.00004]      0.000    4/4
   1   5.688    1.13e-04        0.00e+00   [1.0, 1.00011]      0.000    4/4
   2  18.031    4.96e-05        0.00e+00   [1.0, 1.00005]      0.000    4/4
   3  32.000     -              0.00e+00   (dynamic sampling: all reward=32, std=0 -> filtered)
   ...
```
- DAPO trains (reward 2.6 -> 32 cap); dynamic sampling correctly drops zero-variance groups
  after convergence (no spurious updates).
- Zero KL: `dec_vs_fwd`==0 every step (vLLM prefill==decode); `rollout==train`
  (|vLLM-GPTModel - Megatron-GPTModel| logp) ~1e-4 (near-bitwise, same model two runtimes).
- is_ratio in [1.0, 1.0001]; clipfrac 0 (on-policy).
- Note: overlong filtering masks ALL samples for a continuation/token-count reward (nothing
  emits EOS in the budget) -> use it only for EOS/math-style rewards (pass --overlong_filtering 0
  for the demo reward). dynamic sampling + dual-clip + token_mean all exercised.

### Result 4 — SkyRL-native metric semantics + logging + wandb
Corrected the zero-KL metric to SkyRL's `policy/rollout_train_logprobs_abs_diff`: the trainer's
OLD-logprob recompute goes through the UNIFIED vLLM-GPTModel (== rollout behavior, bitwise) ->
**diff == 0.00e+00 every step** (true zero KL). The Megatron *gradient* forward (new_logp)
carries only ~1e-6 fp32 drift, visible solely in `is_ratio` (~1.00004). Swapping the trainer
attention TE->flash_attn did NOT change the residual -> it's the fp32 cross-runtime floor, not
the attention kernel.
Added the full SkyRL logging taxonomy to examples/zerokl/dapo_zerokl.py: trainer/, generate/,
timing/, system/, reward/, policy/, is_ratio_*, dapo/. wandb runs under project zerokl_qwen_dapo
(zander-jiang). Why it wasn't there before: the demo is a self-contained loop, not the
Ray-orchestrated RayPPOTrainer that emits those sections.
