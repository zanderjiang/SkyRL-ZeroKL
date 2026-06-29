set -x

# DAPO with ZERO KL for Qwen3-4B (unified Megatron-GPTModel route).
# Rollout engine and trainer compute BITWISE-IDENTICAL logprobs (|behavior - train_old| == 0).
#
# Mirrors the DAPO algorithm config from
#   examples/train/megatron/run_megatron_dapo_qwen3_4b.sh
# but uses the SkyRL-ZeroKL unified model (vLLM runs Megatron's GPTModel) + native (no-HF)
# weight sync, so TIS is unnecessary (is_ratio == 1). Single-GPU TP=1 demo loop.
#
# Run:
#   bash examples/zerokl/run_dapo_zerokl_qwen3_4b.sh

MODEL_NAME="${MODEL_NAME:-/mnt/local_storage/hf/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"

# --- DAPO algorithm config (same knobs as the non-zerokl megatron DAPO recipe) ---
CLIP_RATIO_LOW=0.2          # eps_clip_low
CLIP_RATIO_HIGH=0.28        # eps_clip_high (clip-higher)
CLIP_RATIO_C=10.0           # dual-clip lower bound
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=1
DYNAMIC_SAMPLING=1
USE_KL_LOSS=false           # DAPO: no KL loss
TEMPERATURE=1.0
TOP_P=1.0
# TIS is OFF for zero-KL (is_ratio == 1 because rollout==train bitwise)

# --- run sizing (small for a single-GPU demo; scale up for real training) ---
STEPS=20
PROMPTS_PER_STEP=8
N_SAMPLES_PER_PROMPT=8      # DAPO group size
MAX_RESPONSE_LENGTH=32
LR=1e-2
GPU_MEM_UTIL=0.40

cd /home/ray/default/SkyRL-ZeroKL
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
/home/ray/skyrl-zerokl-venv/bin/python -m examples.zerokl.dapo_zerokl \
  --model "$MODEL_NAME" \
  --steps $STEPS \
  --prompts_per_step $PROMPTS_PER_STEP \
  --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
  --max_response_length $MAX_RESPONSE_LENGTH \
  --lr $LR \
  --eps_clip_low $CLIP_RATIO_LOW \
  --eps_clip_high $CLIP_RATIO_HIGH \
  --clip_ratio_c $CLIP_RATIO_C \
  --loss_reduction $LOSS_REDUCTION \
  --dynamic_sampling $DYNAMIC_SAMPLING \
  --overlong_filtering $APPLY_OVERLONG_FILTERING \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --gpu_mem_util $GPU_MEM_UTIL \
  "$@"
