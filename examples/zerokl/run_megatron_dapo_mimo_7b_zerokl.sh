set -x

# ==========================================================================================
# ZERO-KL DAPO for XiaomiMiMo/MiMo-7B-RL  (TP=1 first-validation variant)
# ==========================================================================================
# Zero-KL twin of run_megatron_dapo_mimo_7b_rl.sh. The rollout engine runs Megatron's GPTModel
# (GPTModelVLLMWrapper) instead of MiMo's native HF arch, weights are synced NATIVELY (no HF
# conversion), and the Megatron forward runs under the unified (vops) RMSNorm. Goal: drive
# `policy/rollout_train_logprobs_abs_diff` (== |vLLM rollout - Megatron forward|) from the
# ~2-4e-3 HF baseline down to the cross-engine floor (~1e-6), with is_ratio == 1 at the first
# inner step. TIS is OFF (unnecessary at zero KL).
#
# All DAPO knobs / data / reward / ckpt are kept IDENTICAL to the non-zerokl reference so the two
# runs are directly comparable on reward convergence. The differences are ONLY:
#   * SKYRL_ZERO_KL=1                       -> unified model + native sync + scoring_mode forward
#   * tis_ratio_type=null                   -> TIS off
#   * TP=1 (DP=8) + optimizer offload       -> first validation: native sync + unified model are
#                                              proven at TP=1; 7B+Adam fits TP=1 only with offload.
#                                              (matched-TP4 is the follow-up once GPTModel-in-vLLM
#                                              at TP>1 + matched NCCL lands -- see ZEROKL_SKYRL_INTEGRATION.md)
#   * async_engine=false + in-process vLLM  -> the GPTModel string-registration done in the engine
#                                              actor must reach the model build (no fresh mp worker).
#   * run_name / paths                      -> zerokl_* so it doesn't collide with the baseline.
#
# Runs on 1 node of 8xH100s (80GB each). Throughput WILL be lower (batch-invariant + eager +
# unified model + optimizer offload) -- expected; we are measuring reward gain from zero KL.
#
# Prepare data first (same as baseline):
#   DATA_DIR=/mnt/local_storage/data/dapo bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# Launch (log to the fast local disk, NOT the ~/default quota):
#   WANDB_API_KEY=<key> bash examples/zerokl/run_megatron_dapo_mimo_7b_zerokl.sh \
#       > /mnt/local_storage/logs/zerokl_dapo_mimo_7b.log 2>&1

MODEL_NAME="XiaomiMiMo/MiMo-7B-RL"
DATA_DIR="/mnt/local_storage/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
NUM_NODES=1
NUM_GPUS_PER_NODE=8
# One vLLM engine per GPU (TP=1), each running the unified GPTModel.
NUM_INFERENCE_ENGINES=8
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1
LOGGER="wandb"

# ----- DAPO knobs (IDENTICAL to the non-zerokl reference) -----
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=true
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 8))
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
EVAL_N_SAMPLES_PER_PROMPT=16
ENFORCE_EAGER=true
LR=1e-6

# ----- parallelism: TP=1 (DP=8) for the first zero-KL validation -----
MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=1
MEGATRON_ETP=null

# ----- optimizer offload ON: required to fit 7B + Adam at TP=1 on one 80GB GPU -----
OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

REMOVE_MICROBATCH_PADDING=true

# ===== zero-KL switches =====
export SKYRL_ZERO_KL=1
# In-process vLLM so the GPTModel registration (done in the engine actor) reaches the model build.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# Engine-side GPTModel does NOT load HF weights from disk (it can't consume HF layout); the first
# sync_weights (runs before the first generation) populates it via native sync.
export SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS=1
export _SKYRL_USE_NEW_INFERENCE=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
DISTRIBUTED_EXECUTOR_BACKEND="mp"

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.overlong_buffer_len=$OVERLONG_BUFFER_LEN \
  trainer.algorithm.overlong_buffer_penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  generator.inference_engine.distributed_executor_backend="$DISTRIBUTED_EXECUTOR_BACKEND" \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.algorithm.off_policy_correction.tis_ratio_type=null \
  trainer.remove_microbatch_padding=$REMOVE_MICROBATCH_PADDING \
  trainer.epochs=10 \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=5 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=false \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.logger="$LOGGER" \
  trainer.project_name="mimo_7b_rl_dapo" \
  trainer.run_name="zerokl_dapo_mimo_7b_rl_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}" \
  trainer.export_path="/mnt/local_storage/exports/zerokl_dapo_mimo_7b_rl_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}" \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_path="/mnt/local_storage/ckpts/zerokl_dapo_mimo_7b_rl_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}" \
  $@
