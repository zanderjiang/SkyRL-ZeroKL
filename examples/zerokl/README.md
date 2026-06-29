# examples/zerokl — DAPO with zero KL (unified Megatron-GPTModel)

Zero-KL RL examples where the **rollout engine and the trainer run the same Megatron `GPTModel`**
(vLLM runs GPTModel via `skyrl.backends.skyrl_train.zerokl.gptmodel_vllm`), so their token
logprobs are bitwise / near-bitwise identical and the importance ratio is ≈1 without TIS.

## Files
- `dapo_zerokl.py` — self-contained DAPO loop with the DAPO algorithm pieces:
  dual-clip / clip-higher (`eps_low=0.2`, `eps_high=0.28`, `clip_ratio_c=10`), `token_mean`
  loss, dynamic sampling (drop zero-advantage groups), overlong filtering (mask truncated
  samples), no KL loss, temperature 1.0. TIS is OFF (unnecessary at zero KL).
- `run_dapo_zerokl_qwen3_4b.sh` — launcher mirroring the DAPO knobs in
  `examples/train/megatron/run_megatron_dapo_qwen3_4b.sh` (the non-zerokl reference).

## How the zero KL works
- Rollout: vLLM running Megatron `GPTModel` (`GPTModelVLLMWrapper`, attention -> vLLM paged,
  RoPE indexed by absolute positions). Generation is coherent and `prefill == decode` bitwise.
- old / train logprob: a vLLM-GPTModel **prefill rescore** of the realized tokens.
- new logprob (grad): the **Megatron GPTModel** forward (TE attention) -> DAPO update.
- Weight sync: **native, no HF** — `named_parameters` copy trainer -> rollout each step
  (`zerokl/native_weight_sync.py`); both sides hold the identical state_dict.

## Logging taxonomy (SkyRL-native sections, logged to wandb)
This standalone loop emits the same metric sections as the full SkyRL trainer:
- `trainer/` — `global_step`, `epoch`, `tokens_per_second`
- `generate/` — `batch_num_seq`, `response_length_mean`, `response_length_max`
- `timing/` — `generate`, `old_logprob`, `train_step`, `weight_sync`, `step`
- `system/` — `gpu_mem_alloc_gb`, `gpu_mem_reserved_gb`
- `reward/` — `mean`, `avg_raw_reward`, `mean_positive_reward`, `num_zero_variance_filtered`
- `policy/` — `rollout_train_logprobs_abs_diff_{mean,max,min,std}`, `clipfrac`, `policy_loss`
- `is_ratio_{mean,std,max,min}`, `dapo/groups_{kept,total}`

(Note: the demo is a self-contained loop on the `zerokl` components, not the Ray-orchestrated
`RayPPOTrainer`; these sections are emitted manually to mirror SkyRL's taxonomy. Wiring the
zerokl model/sync into the real trainer would get the full set for free.)

## Key zero-KL / DAPO metrics
- `policy/rollout_train_logprobs_abs_diff_{mean,max,min,std}` — `|rollout behavior − trainer
  old-logprob recompute|`. The trainer's old-logprob recompute goes through the **unified
  vLLM-GPTModel** (the same model the rollout uses), so this is **exactly 0** = true zero KL.
- `is_ratio_{mean,std,max,min}` — `exp(new_logp[Megatron grad forward] − old_logp)`. ≈1; the
  only residual is the fp32 drift between Megatron's `forward_backward_func` and the unified
  forward (~1e-6 typical), since the gradient forward must be grad-capable (Megatron).
- `policy/clipfrac`, `policy/policy_loss`, `reward/mean`, `dapo/groups_kept`, `dapo/groups_total`.

To get is_ratio EXACTLY 1 too, the Megatron gradient forward would need the same execution path
as the unified model (TorchTitan does this with one torch-native varlen kernel both sides; our
env lacks `torch.nn.attention.varlen_attn_out`/FA3 — see ../../ZEROKL_STATUS.md).

## Run
```bash
# single-GPU demo (TP=1). Uses the cached Qwen3-4B by default.
bash examples/zerokl/run_dapo_zerokl_qwen3_4b.sh

# the token-count demo reward truncates every continuation, so overlong filtering masks all
# samples; pass --overlong_filtering 0 for that reward (it matters for math-style EOS tasks):
bash examples/zerokl/run_dapo_zerokl_qwen3_4b.sh --overlong_filtering 0 --steps 12
```

## Scope / notes
Single-GPU TP=1 demo built on the SkyRL-ZeroKL `zerokl` package (not the full Ray-orchestrated
multi-node pipeline). The reward is a learnable token-count proxy so DAPO's dynamic sampling and
clipping are exercised with a clear learning signal; swap `reward_fn` for a real verifier
(math/code) for actual DAPO training. See `../../ZEROKL_STATUS.md` / `../../ZEROKL_WORKLOG.md`.
