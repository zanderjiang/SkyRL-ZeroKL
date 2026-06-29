# Wiring zero-KL into SkyRL's `main_dapo` pipeline

Goal: run a **real SkyRL DAPO run** (`examples.train.algorithms.dapo.main_dapo`, RayPPOTrainer)
that is **zero-KL** — the rollout engine and the trainer compute bitwise-identical logprobs, so
`policy/rollout_train_logprobs_abs_diff == 0` and TIS is unnecessary. Target model: MiMo-7B-RL
(dense). Strategy: **unified model** — vLLM runs Megatron's `GPTModel`; Megatron-core owns the
gradient/optimizer/checkpointing. All gated by env `SKYRL_ZERO_KL=1` so the regular pipeline is
untouched when off.

Plan: get it correct at **TP=1** first (validate `rollout_train_abs_diff==0`), then matched-TP4.

## Integration points

### 1. Rollout engine runs GPTModel-in-vLLM  (task #23 — hook added)
`inference_engines/vllm/vllm_engine.py::setup_envvars_for_vllm` (gated by `SKYRL_ZERO_KL`):
- `register_gptmodel_to_vllm()` — vLLM **string** registration `module:GPTModelVLLMWrapper`.
- inject `kwargs["hf_overrides"]={"architectures":["MegatronGPTModelForCausalLM"]}` so vLLM builds
  our wrapper instead of MiMo's native arch.
- wrapper now derives `model_path` from `vllm_config.model_config.model` (no closure needed).

**OPEN — cross-process registration.** vLLM mp/async workers are spawned fresh, so the registry
populated in the engine actor doesn't reach them. Two fixes:
  - (a) in-process: `VLLM_ENABLE_V1_MULTIPROCESSING=0` (set by the hook) — validate first.
  - (b) production: expose the wrapper as a vLLM **general plugin** (entry point
    `vllm.general_plugins`) so every worker registers it on startup.
The MiMo script uses `async_engine=true` + `_SKYRL_USE_NEW_INFERENCE=0`; for the first zero-KL
validation force the sync/in-process engine.

### 2. Native (no-HF) weight sync  (task #24 — TODO)
Today SkyRL: `MegatronWeightExtractor.export_hf_weights` -> HF layout -> vLLM `load_weights`
repack. For the unified model both sides hold the identical GPTModel state_dict, so:
- sender: replace `export_hf_weights` with `zerokl.native_weight_sync.extract_native_weights`
  (native names, bf16) under `SKYRL_ZERO_KL`.
- receiver: the wrapper's `load_weights` is a no-op; instead copy native tensors straight into
  `GPTModelVLLMWrapper.gpt` via `load_native_weights`. Reuse SkyRL's CUDA-IPC/NCCL transport
  (only the extract/load ends change). At TP=1 it's a direct copy; TP>1 needs native reshard.

### 3. old-logprob via the unified model + TIS off + batch-invariant  (task #25 — TODO)
- old/train logprob recompute routed through the unified model (== rollout behavior) so
  `policy/rollout_train_logprobs_abs_diff == 0`. (megatron_model_wrapper recompute path.)
- `trainer.algorithm.off_policy_correction.tis_ratio_type=null` (TIS off — not needed at zero KL).
- megatron worker init applies `apply_megatron_zerokl_patches()` (already hooked by `SKYRL_ZERO_KL`
  in `megatron_worker.init_model`).

### 4. Matched TP (after TP=1 validation)
Rollout (vLLM-GPTModel) and trainer at the same TP with bitwise-matched NCCL all-reduce
(`NCCL_ALGO=allreduce:tree`, 1 channel, identical topology). Requires GPTModel-in-vLLM at TP>1
(reconcile Megatron `parallel_state` with vLLM's TP world) — the genuinely hard step; all current
proofs are TP=1.

## Launch plan (zero-KL twin of `run_megatron_dapo_mimo_7b_rl.sh`)
Identical config (project `mimo_7b_rl_dapo`, DAPO knobs, data, ckpt) EXCEPT:
`SKYRL_ZERO_KL=1`, `tis_ratio_type=null`, run_name `zerokl_dapo_mimo_7b_...`, (TP=1 for the first
validation; matched-TP4 once step 4 lands). Throughput will be lower (batch-invariant + eager +
unified model) — expected; we're measuring reward convergence at zero KL.

## What "zero KL" actually means inside SkyRL (grounded in trainer.py)
`policy/rollout_train_logprobs_abs_diff = |rollout_logprobs (vLLM) − action_log_probs (Megatron
policy forward)|` (trainer.py:1303-1320). The PPO `is_ratio` is **already exactly 1 at the first
inner step** by SkyRL design: `recompute_old_logprobs_per_minibatch=True` runs the old forward
with the SAME per-minibatch packing as `forward_backward` → old==new — **but only if both forwards
use identical numerics.** The fused-norm gate (`not is_grad_enabled()`) violated that (no-grad old
forward used vops, grad train forward used native TE). #25 fixes it: BOTH megatron forwards run
under `scoring_mode()` (vops) in zero-KL mode → old==new AND both match the vLLM rollout.
So the zero-KL win = `rollout_train_logprobs_abs_diff` drops from the ~2-4e-3 HF baseline to the
cross-engine fp32 floor (~1e-6 at TP=1). Exactly-0 on that metric needs the unsolved
same-execution-path (matched-TP4, step 4).

## Status
- [x] #23 engine hook: `setup_envvars_for_vllm` registers GPTModel (string form) + injects
      `hf_overrides.architectures` + forces in-process engine. Gated by SKYRL_ZERO_KL.
- [x] #24 native weight sync: sender (`megatron_worker.extract_weights`) yields native params;
      receiver (`vllm_worker.load_weights`) copies into `model.gpt`. Both gated, compile + import OK.
      (TP=1 only; TP>1 sender needs a Megatron TP-gather.)
- [x] #25 unified numerics: both megatron forwards wrapped in `scoring_mode()`
      (`megatron_model_wrapper._zerokl_scoring_ctx`); TIS off via launch script
      (`tis_ratio_type=null`); BIK patches at worker init (existing hook).
- [x] launch script `examples/zerokl/run_megatron_dapo_mimo_7b_zerokl.sh` (TP=1, opt offload,
      in-process engine, run `zerokl_dapo_*`, project `mimo_7b_rl_dapo`).
- [x] ENV FORWARDING (required, cost a launch): `SKYRL_ZERO_KL` must be forwarded to ray actors
      explicitly — added to `utils.py::prepare_runtime_environment` (job env, megatron workers) AND
      `inference_engines/utils.py::build_engine_runtime_env` (engine actors). A driver `export`
      alone does NOT reach actors (esp. with a pre-existing ray cluster). Confirmed: all 16 actors
      have SKYRL_ZERO_KL=1; engine actors load `libtransformer_engine.so` ⇒ running GPTModel.
- [~] LAUNCHED (tmux `zerokl_dapo`) — env forwarding ✓, engine runs unified GPTModel ✓, first
      generation in progress. Awaiting first-step `rollout_train_logprobs_abs_diff` (target ~1e-6
      vs ~2-4e-3 HF baseline). NOTE: SkyRL suppresses module-logger INFO, so `[zerokl]`/`copied N
      tensors` lines don't show — verify via /proc maps + the metric, not the logs.
- [ ] matched-TP4 (GPTModel-in-vLLM at TP>1 + matched NCCL) — the hard follow-up.

## DEBUG LOG — first SkyRL run generated gibberish (in progress)
Symptom: full SkyRL pipeline generates gibberish (reward ~-2.0, every response hits max len =
never EOS). Isolation chain:
1. Env forwarding works — all 16 actors have SKYRL_ZERO_KL=1; engine actors load
   `libtransformer_engine.so` ⇒ GPTModel built in vLLM. (NOT the bug.)
2. **GPTModel-in-vLLM forward is CORRECT for MiMo** — standalone `scratchpad/mimo_gen_test.py`
   (closure-form register + bridge load_weights=1) generates COHERENT text in BOTH vLLM 0.20.2
   (skyrl-zerokl-venv) AND vLLM 0.23.0 (SkyRL's `uv run --isolated --extra megatron`). So neither
   the forward nor the vLLM version is the bug. (run via:
   `CUDA_VISIBLE_DEVICES=0 uv run --isolated --extra megatron python scratchpad/mimo_gen_test.py`)
3. The ONLY thing the full pipeline does that standalone doesn't: **run `sync_weights`** before
   generation. Leading hypothesis: with ENGINE_LOAD_WEIGHTS=1 the engine starts coherent
   (bridge-loaded) but sync corrupts it — my receiver branch (`vllm_worker.py WorkerWrap.load_weights`)
   is NEVER called (no `[ZEROKL-PROBE]` line), so the sync goes through a different path.
   Diagnostics added (print(), bypass log-suppression):
   - `[ZEROKL-PROBE]` at top of `WorkerWrap.load_weights` (did NOT fire → not the receiver).
   - `[ZEROKL-WRAP]` in `GPTModelVLLMWrapper.__init__/forward/load_weights` (weight norm at build
     vs forward → detects corruption; whether wrapper.load_weights is the sync no-op).
   NEXT: read the WRAP probes to see if weights change between build and forward.
   Real receiver path: `engine.update_named_weights -> _weight_loader.load_weights ->
   collective_rpc("load_weights")` (vllm_engine.py:772). worker_extension_cls wired at
   ray_wrapped_inference_engine.py:308 = vllm_engine.WorkerWrap (re-export of inference_servers one).

### ROOT CAUSE FOUND (resolved)
- **Actor stdout is redirected to `/tmp/skyrl-logs/infra-<ts>.log`** (via SKYRL_LOG_FILE +
  redirect_actor_output_to_file). ALL `print()`/module-logger output goes there, NOT the driver
  log. (Set `SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1` to send to stdout.) I was reading the wrong file.
- The wrapper DOES build correctly: `[ZEROKL-WRAP] built load_weights=True first_w_norm=359.957`,
  and `forward#1 first_w_norm=359.957 top3_logits sane` ⇒ **engine has correct weights + forward
  works**. Standalone confirms coherent gen even with ALL SkyRL vLLM args (prefix-cache + chunked +
  packed/batched prefill, vLLM 0.23.0).
- **First full run gibberish = `ENGINE_LOAD_WEIGHTS=0`** (bridge didn't materialize weights + sync
  was a no-op ⇒ random). Debug runs (`=1`) generate coherently; `reward -2.0/len-256` was MiMo
  reasoning truncated at 256 tokens on AIME, not gibberish.
- **THE bug: the weight sync never reached the engine.** (a) `get_weight_metadata` + `extract_weights`
  used `export_hf_weights` (451 HF names) — a metadata-driven HF transfer; (b) the wrapper's
  `load_weights` was a NO-OP. So trainer updates never propagated; only the step-0 bridge weights
  (which happen to equal the trainer's at step 0) were ever in the engine.
- **FIX**: native end-to-end — `get_weight_metadata` (native) + `extract_weights` (native) +
  `GPTModelVLLMWrapper.load_weights` copies native into `self.gpt` (return ALL names so vLLM's
  build check passes). Sender fully unwraps DDP/Float16Module (was leaving a `module.` prefix);
  sender yields 255 native (engine has 266 incl. an extra MTP layer — harmless).

### CURRENT BLOCKER (run reaches full steps but is NOT zero-KL)
The job RUNS end-to-end in SkyRL (all phases, metrics, no crash) but:
- `policy/rollout_train_logprobs_abs_diff_mean ≈ 2.78` (max ~40) — ~1000× the ~3e-3 HF baseline,
  so NOT a kernel/numerics gap; structural.
- `policy/policy_entropy ≈ 8.15` (vocab 151680, max ln≈11.9) — the **trainer's Megatron forward is
  near-uniform** while the engine generates confidently (engine proven coherent standalone). So the
  **trainer-side zero-KL patches break the Megatron training forward.** Baseline non-zerokl MiMo
  trains fine ⇒ the patches are the delta. Prime suspect: the #25 `scoring_mode` wrap forcing the
  vops fused-norm in the GRAD forward (engine forward is no_grad inference; trainer is grad).
- Native sync still shows `copied 0` at the receiver — those are vLLM BUILD-TIME HF loads; need to
  confirm the trainer-SYNC call delivers native names to `wrapper.load_weights` (separate from the
  forward bug; at step 0 weights≈equal so it doesn't cause the 2.78).
- BISECT toggle added: `SKYRL_ZEROKL_SCORING_FORWARD=0` disables the scoring wrap (forwarded via
  utils.py). If entropy normalizes (~1-3) + diff → ~3e-3, the vops-norm-in-grad wrap is the breakage.

### REAL ROOT CAUSE (the gibberish) — colocate sleep/wake zeroes the engine weights
Bisect results: scoring-wrap OFF → still 8.15/2.78; trainer-patches OFF (vanilla Megatron) → still
8.15/2.78. So neither trainer patches nor the engine forward are the bug. A generation-time probe
(`forward#30/100`) showed the smoking gun:
  - `forward#1` (BUILD, pre-sleep): `first_w_norm=359.957 out_entropy=1.892` (correct, confident)
  - `forward#30` (GENERATION, post-wake): **`first_w_norm=0.000 out_entropy=11.930`** (= ln(vocab), uniform)
**The engine's GPTModel weights are ZEROED at generation.** With colocate_all, SkyRL sleeps the
engine at **sleep_level=2** (default; `ray_wrapped_inference_engine.py:119`), which FREES the cumem
weight region. Standard vLLM models get refilled by vLLM's loader on wake; our bridge-loaded
GPTModel weights are NOT vLLM-cumem-tracked, so wake leaves them zero. The native sync DOES copy
correct weights into model.gpt (`[ZEROKL-SYNC] copied 255 missed=0`) but a subsequent wake re-zeros
the live buffers.
**FIX**: force `sleep_level=1` under SKYRL_ZERO_KL (`ray_wrapped_inference_engine.py:341` +
`vllm_engine.py` both sleep methods). NOT SUFFICIENT alone — see below.

### DEFINITIVE ROOT CAUSE: cumem leaves the bridge params on META device
sleep level 1/2 both still gave norm 0.0 at generation. The crash that revealed it:
`RuntimeError: Tensor.item() cannot be called on meta tensors` in the sync's diagnostic. At sync
time `model.gpt.named_parameters()` are **META tensors** — vLLM cumem sleep frees the storage
(`Sleep mode freed 43.09 GiB`) and the bridge-loaded (non-vLLM-tracked) GPTModel params come back
as meta placeholders. So the native sync's `dest.copy_(tensor)` into a meta tensor is a **silent
no-op** → weights never materialize → norm 0.0 → uniform/gibberish generation.
**FIX**: in the WorkerWrap `[ZEROKL-SYNC]` receiver, MATERIALIZE meta params by assigning
`dest.data = src` (real GPU tensor) instead of `copy_` when `dest.is_meta` (or shape differs).
This makes the per-step sync allocate real storage with the trainer's weights after each wake.
Validation: run zerokl_dbg13 — expect `[ZEROKL-SYNC] copied N (materialized N) ... first_w_norm=359.957`,
`forward#30 first_w_norm≈359.957` low entropy, and `rollout_train_logprobs_abs_diff` small.
(Also: another caching bug fixed — the sync's dst dict must be rebuilt each call, not cached, since
cumem swaps the param storage across wakes.)

### ✅ FIXED (dbg15) — engine generates coherently in the full SkyRL pipeline
Materialization via module replacement works (meta param -> `mod._parameters[attr] =
nn.Parameter(real_gpu_tensor)`; plain `dest.data=` fails on meta with "incompatible tensor type").
Confirmed: `[ZEROKL-SYNC] copied ... materialized ... live first_w_norm=359.957` and
`forward#100 first_w_norm=359.957 out_entropy=0.000` (real weights + confident, coherent output;
was 0.000 norm / 11.93 uniform before). The native zero-KL weight path now works end to end:
trainer Megatron GPTModel -> native sync (no HF) -> materialize into the vLLM GPTModel each step.
Summary of the full fix chain: env forwarding (utils.py) + GPTModel-in-vLLM (engine hook) + native
metadata/extract/load (no HF) + fully-unwrap names + sleep_level=1 + rebuild-dst-each-call +
**materialize meta params on sync**. Next: read step-1 `rollout_train_logprobs_abs_diff`.
