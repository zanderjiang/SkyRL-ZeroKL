# True bitwise zero-KL on the nightly stack (VALIDATED on MiMo-7B)

## Result (measured)
Real MiMo-7B-RL, local-spec Megatron GPTModel served in vLLM-1.0 with the PyTorch-varlen
attention backend (`num_splits=1`):
```
coherent generation: "The capital of France is Paris, the city of love, of lights, of art..."
weight load: matched 255/255, missing 0, embedding_norm 359.957 (exact)
MAX |decode - prefill| over 256 tokens = 0.000000e+00   tokens EXACT 0.0: 256/256
RESULT: BITWISE-IDENTICAL (max==0)
```
=> rollout logprobs (decode) are bitwise-identical to a clean forward (prefill) =>
`policy/rollout_train_logprobs_abs_diff -> 0`. True zero-KL, no compromise.

## Why this works (and the working-stack/0.23 path did not)
- The residual was NEVER weight delivery or the trainer forward (both bitwise). It was vLLM's
  paged-attention DECODE drifting from a full-sequence PREFILL, growing with length (the FA3
  split-k heuristic at num_splits=auto). `num_splits=1` makes the attention bitwise.
- BUT bitwise decode==prefill needs the WHOLE forward batch/length-invariant. vLLM's *native*
  MiMo model has a non-attention op that stays batch-variant. Our **Megatron local-spec GPTModel**
  uses only plain torch ops (SDPA, torch RMSNorm, `F.linear`) which vLLM's `VLLM_BATCH_INVARIANT`
  covers -> fully invariant -> bitwise. That's the whole trick.
- torch 2.11 / vLLM 0.23 lack the pieces (`varlen_attn_out`; vLLM-0.23 FA3 ignores num_splits).
  The nightly stack has them.

## Environment (`/mnt/local_storage/zerokl-nightly-venv`)
torch 2.14.0.dev20260620+cu130 · vllm 1.0.0.dev20260620+cu130 (internal 0.22.1rc1.dev) ·
megatron-core 0.18.0 (git 71e418ea7, **no Transformer Engine**) · flash-attn-3 3.0.0.
TE is ABI-incompatible with this torch nightly and is NOT needed (local spec avoids it).

## Recipe
1. **Weights** (`prod_local_build.py`, run on the PRODUCTION venv which has megatron-bridge):
   `AutoBridge.from_hf_pretrained(MiMo).to_megatron_provider(load_weights=True)`, set
   `transformer_impl='local'`, `gradient_accumulation_fusion=False`, `apply_rope_fusion=False`,
   `add_qkv_bias` is implied by the config; save `gpt.state_dict()` (bf16) to
   `/mnt/local_storage/mimo_local_sd.pt`. (The bridge emits TE-FUSED norm names even for 'local'.)
2. **Engine model** (`mimo_megatron_vllm.py`): build a local-spec GPTModel from megatron-core
   (`get_gpt_layer_local_spec(qk_layernorm=False, normalization="RMSNorm")`, `add_qkv_bias=True`,
   `add_bias_linear=False`, `gradient_accumulation_fusion=False`), then load the saved state_dict
   with the fused->local **remap**: `self_attention.linear_qkv.layer_norm_weight ->
   input_layernorm.weight`, `mlp.linear_fc1.layer_norm_weight -> pre_mlp_layernorm.weight`
   (all other names identical). Swap `core_attention -> vLLM paged Attention`; index RoPE by
   absolute positions for decode.
3. **Attention backend** (`varlen_backend.py`): register `@register_backend(CUSTOM)` ->
   `varlen_attn_out(..., num_splits=1)`; select via `LLM(..., attention_backend="CUSTOM")`.
4. **Serve**: `LLM(model=MiMo_dir, config_format="megatron_mimo", load_format="dummy" (+ no-op
   dummy weight init), enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False)`,
   `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_BATCH_INVARIANT=1`.
5. **Verify** (`mimo_parity_test.py`): coherent gen + decode-vs-prefill max==0.

## Remaining: the DAPO loop (port of examples/zerokl/dapo_zerokl.py)
Build a SECOND local-spec MiMo GPTModel (same state_dict) as the trainer; loop: vLLM rollout ->
reward -> GRPO/DAPO advantage -> trainer GPTModel forward (new_logp, grad) -> dual-clip loss ->
SGD/Adam -> native weight sync (copy trainer params -> engine `gpt` params, identical local names).
`rollout_train_abs_diff` is bitwise 0 (decode==engine-prefill) and is_ratio==1 at the first inner
step. TIS OFF (genuinely unnecessary now).
