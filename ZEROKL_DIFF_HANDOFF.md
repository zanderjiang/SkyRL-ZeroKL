# Zero-KL residual: problem handoff

## The problem
In the full SkyRL pipeline (`examples/zerokl/run_megatron_dapo_mimo_7b_zerokl*.sh`, MiMo-7B, TP1),
`policy/rollout_train_logprobs_abs_diff_mean ≈ 0.0104` (max ~0.49, min 0). That is **worse than
normal SkyRL (~2–4e-3)** and far from the zero-KL target (~1e-6). It is byte-stable across reruns.

**What the metric is:** `|rollout_logprobs − trainer_logprobs|` over response tokens, where
- rollout_logprobs = engine (vLLM running Megatron GPTModel) logprobs captured during generation
- trainer_logprobs = the Megatron GPTModel forward recompute (same synced weights)
With DAPO + TIS off, the *logged* number is the **minibatch** variant computed in
`workers/worker_utils.py::compute_minibatch_rollout_logprob_diff_metrics` (not the
`trainer.py:1306` block, which is skipped — don't instrument that one).

## Confirmed: bitwise IS achievable for MiMo
`examples/zerokl/dapo_zerokl.py --model /mnt/local_storage/models/MiMo-7B-RL --dynamic_sampling 0`
(single process, engine + a Megatron model both built by the zerokl components) gives
**`is_ratio = [1.00000, 1.00000]`** and `rt_abs_diff = 0` → the unified GPTModel + flash kernels
reach ~1e-6 for MiMo. So the 0.0104 is a **SkyRL-pipeline-specific regression**, not MiMo arch,
not the attention kernel per se.

## Ruled out (diff stayed ≈0.0104, usually byte-identical)
- RoPE base (MiMo rope_theta=640000; verified correct on BOTH engine and trainer)
- RoPE fusion (`apply_rope_fusion=False` forced on trainer)
- sample packing (`remove_microbatch_padding=false`) — no-op anyway at micro_batch=1
- `variable_seq_lengths=False`, `masked_softmax_fusion=False`, `gradient_accumulation_fusion=False`
- softmax precision (trainer logprobs are fp32; `_compute_distributed_log_softmax`)
- position_ids (`cumsum(mask)-1` == arange == engine absolute positions)
- **attention kernel**: swapped trainer TE → `flash_attn_varlen_func` (== engine vLLM flash,
  TorchTitan approach) → **NO change**. Also batch-invariant patches on/off → NO change.
- logprob extraction (`from_parallel_logits_to_logprobs`: fp32, correct -1 target shift)

## Key established facts
- **Engine is internally perfect**: `scratchpad/mimo_decode_vs_prefill.py` shows engine
  decode == engine prefill rescore EXACTLY (|diff|=0.00000). So rollout_logprobs are trustworthy.
- **The 0.0104 is diffuse**: most tokens match, worst ~0.1–0.3 (e.g. train −0.815 vs rollout
  −0.684). NOT a systematic offset, NOT a few catastrophic outliers, NOT an off-by-one.
- **Robust to the entire forward path** → strongly implies the cause is NOT in attention/norm/
  matmul/rope. Remaining suspects: the multi-process **weight delivery** (engine weights via the
  cumem-materialize sync vs the trainer's forward weights) or a cross-actor numerical difference
  the single-process standalone doesn't have.

## *** RESULT: weight delivery IS broken (root cause found) ***
Checksum probe (dbg25) shows engine weights != trainer weights:
```
SENDER (trainer)  sent 255 params, abs-sum = 89,866,863
RECEIVER (engine) recv-abs-sum = 10,475,048   engine-gpt-abs-sum = 10,475,048
```
The trainer sends 89.8M (abs-sum) but the engine holds only 10.5M. `recv == engine-gpt`, so the
receiver loads faithfully what it *gets* — the loss/corruption is between SENT (89.8M) and RECEIVED
(10.5M). The engine only looks coherent because ENGINE_LOAD_WEIGHTS=1 bridge-loaded MiMo at init;
the sync then partially overwrites, leaving engine != trainer -> the ~0.01 rollout_train diff.
Investigate: (1) do the big params (embedding ~152k×4096, output_layer) actually flow through the
`[ZEROKL-SYNC]` materialize branch, or a different load path / stay meta? (2) is the IPC/NCCL
transfer (cuda_ipc_strategy) handing the receiver stale handles or a wrong dtype/shape view that
shrinks magnitude ~8.6x? (3) does "cum 255" copies actually mean 255 *distinct* params or repeats?
Add per-name logging in the receiver to see which names arrive with what abs-sum vs the sender's.

## Leading hypothesis (CONFIRMED above)
Weights delivered to the engine by the sync may not be bitwise-identical to the trainer's forward
weights. Checksum probe added: `[ZEROKL-CKSUM] SENDER ...` (native_weight_sync.extract_native_weights)
vs `[ZEROKL-CKSUM] RECEIVER recv-abs-sum / engine-gpt-abs-sum` (vllm_worker.load_weights). Run
`zerokl_dbg25` produces them in the infra log. If SENDER == RECEIVER == engine-gpt → weights are
fine and it's a forward residual; if they differ → the sync is the bug.
Candidate sync issues: precision-aware optimizer fp32 master vs bf16 param sent (OPTIMIZER_OFFLOAD=true),
or the cumem materialize replacing only a subset / with a dtype/layout change.

## Where to look / how to iterate fast
- Fast repro (≈8 min/run, console logger): `bash examples/zerokl/run_megatron_dapo_mimo_7b_zerokl_debug.sh`
  (256-token responses, MiMo, TP1, all zero-KL switches on).
- **All zero-KL diagnostics print to `/tmp/skyrl-logs/infra-<ts>.log`, NOT the driver log** (SkyRL
  redirects actor stdout via SKYRL_LOG_FILE). Set `SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1` to send to stdout.
  Markers: `[ZEROKL-DIFF]` (per-token worst diffs), `[ZEROKL-CKSUM]`, `[ZEROKL-WRAP]` (engine build/
  forward norm+entropy), `[ZEROKL-SYNC]` (sync copy/materialize), `[ZEROKL-TRAINER]` (patches/swap).
- Bitwise reference: `examples/zerokl/dapo_zerokl.py` (its `trainer_logp` is the simple direct
  forward + `torch.log_softmax` that achieves 1e-6 — diff this against SkyRL's forward path).
- Toggles (env, forwarded to actors): `SKYRL_ZEROKL_FLASH_ATTN`, `SKYRL_ZEROKL_SCORING_FORWARD`,
  `SKYRL_ZEROKL_TRAINER_PATCHES`, `SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS`.

## Files touched (all gated by SKYRL_ZERO_KL)
- `zerokl/gptmodel_vllm.py` — GPTModelVLLMWrapper (engine model), rotary_base fix, forward probe
- `zerokl/native_weight_sync.py` — native (no-HF) extract/load + cksum probe
- `zerokl/megatron_flash_attn.py` — FlashVarlenCoreAttn + swap (now wired)
- `workers/megatron/megatron_worker.py` — patches + flash swap + provider config + native sender
- `workers/megatron/megatron_model_wrapper.py` — scoring_mode wrap + attention_mask=None for zero-KL
- `inference_servers/vllm_worker.py` — receiver: materialize meta params + cksum probe
- `inference_engines/vllm/vllm_engine.py` — engine hook (register GPTModel + hf_overrides)
- `inference_engines/ray_wrapped_inference_engine.py` + `vllm_engine.py` — sleep_level=1
- `train/utils/utils.py` + `inference_engines/utils.py` — forward SKYRL_ZERO_KL* to ray actors

## Separate open item (wandb charts)
The zerokl run is missing the off-policy/TIS charts (`is_ratio_*`, `tis_ratio`, `tokens_capped`)
vs the baseline, because SkyRL only computes them when `tis_ratio_type` is set and zero-KL turns TIS
off. To match the baseline dashboard, compute+log the `is_ratio` diagnostics even with TIS off.
