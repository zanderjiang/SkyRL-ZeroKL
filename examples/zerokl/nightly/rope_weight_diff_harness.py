"""In-process isolation of the engine-vs-trainer forward divergence (RoPE vs weight-sync).

Builds the SkyRL zero-KL engine (gptmodel_vllm wrapper) in-process, then:
  (1) RoPE test: compare the engine's position-indexed RoPE (_emb_full[:L]) to the stock fresh RoPE
      (orig(L)) -- the docstring CLAIMS they're equal for prefill; if not, RoPE is the residual.
  (2) full-forward RoPE effect: run wrapper.gpt forward twice through vLLM is not possible (swapped
      attention needs vLLM context), so we compare the RoPE embeddings directly + (if equal) move on.
  (3) weight test: build a SECOND bridge GPTModel (local_layer_spec, same HF weights, same rope_theta
      workaround, MTP off) exactly like megatron_worker, and compare param-by-param to wrapper.gpt.
"""
import os
os.environ.setdefault("SKYRL_ZERO_KL", "1")
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")
os.environ.setdefault("SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("ZEROKL_MODEL", "/mnt/local_storage/models/MiMo-7B-RL")
L = 256

import torch  # noqa: E402
import skyrl.backends.skyrl_train.zerokl.varlen_backend as varlen_backend  # noqa: E402,F401
from skyrl.backends.skyrl_train.zerokl.gptmodel_vllm import (  # noqa: E402
    register_gptmodel_to_vllm, VLLM_MODEL_NAME, find_inprocess_gptmodel)
from skyrl.backends.skyrl_train.zerokl import apply_vllm_zerokl_env  # noqa: E402
from vllm import LLM  # noqa: E402

apply_vllm_zerokl_env()
register_gptmodel_to_vllm()
print("=== building engine ===", flush=True)
llm = LLM(model=MODEL, hf_overrides={"architectures": [VLLM_MODEL_NAME]}, attention_backend="CUSTOM",
          dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.45, max_model_len=2048,
          enable_prefix_caching=False, enable_chunked_prefill=False, trust_remote_code=True)
wrapper = find_inprocess_gptmodel(llm)
assert wrapper is not None and hasattr(wrapper, "gpt"), "could not reach in-process wrapper"
rope = wrapper._rope
print(f"engine wrapper reached. _rope type={type(rope).__name__}", flush=True)

# ---------- (1) RoPE test: indexed precompute vs fresh ----------
with torch.no_grad():
    emb_full = rope._emb_full            # [max_pos, 1, 1, dim], precomputed once at build
    idx = torch.arange(L, device=emb_full.device)
    indexed = emb_full[idx]              # what the engine uses for positions 0..L-1
    fresh = rope._orig(L)                # what the trainer (stock rotary_pos_emb) computes
    fresh = fresh.to(indexed.device)
    rope_max = (indexed.float() - fresh.float()).abs().max().item()
    print(f"\n[ROPE] max|engine_indexed_rope - stock_fresh_rope| over {L} pos = {rope_max:.3e}", flush=True)
    print(f"[ROPE] emb_full.dtype={emb_full.dtype} fresh.dtype={fresh.dtype} shape={tuple(indexed.shape)}", flush=True)
    if rope_max > 0:
        # which positions diverge
        per_pos = (indexed.float() - fresh.float()).abs().flatten(1).max(dim=1).values
        bad = (per_pos > 1e-6).nonzero(as_tuple=True)[0].tolist()
        print(f"[ROPE] RoPE DIFFERS -> RoPE is (a/the) cause. #bad_pos={len(bad)} first={bad[:10]} "
              f"(pos0 diff={float(per_pos[0]):.3e}, posL-1 diff={float(per_pos[-1]):.3e})", flush=True)
    else:
        print("[ROPE] RoPE is bitwise-identical -> NOT the cause; checking weights next.", flush=True)

ROPE_IS_CAUSE = rope_max > 0

# ---------- (3) weight test (only meaningful if RoPE matches) ----------
if not ROPE_IS_CAUSE:
    print("\n=== building a second (trainer-style) bridge GPTModel to compare weights ===", flush=True)
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.gpt_provider import local_layer_spec
    from transformers import AutoConfig
    b = AutoBridge.from_hf_pretrained(MODEL, trust_remote_code=True)
    mp = b.to_megatron_provider(load_weights=True)
    mp.tensor_model_parallel_size = 1
    mp.pipeline_model_parallel_size = 1
    mp.pipeline_dtype = torch.bfloat16
    mp.apply_rope_fusion = False
    mp.gradient_accumulation_fusion = False
    mp.transformer_layer_spec = local_layer_spec
    if getattr(mp, "mtp_num_layers", None):
        mp.mtp_num_layers = None
    _hf = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    _rp = getattr(_hf, "rope_parameters", None) or getattr(_hf, "rope_scaling", None)
    if isinstance(_rp, dict) and _rp.get("rope_theta"):
        mp.rotary_base = _rp["rope_theta"]
    elif getattr(_hf, "rope_theta", None):
        mp.rotary_base = _hf.rope_theta
    mp.finalize()
    gpt2 = mp.provide_distributed_model(wrap_with_ddp=False)
    gpt2 = gpt2[0].module if hasattr(gpt2[0], "module") else gpt2[0]
    eng = dict(wrapper.gpt.named_parameters())
    trn = dict(gpt2.named_parameters())
    print(f"engine params={len(eng)} trainer params={len(trn)}", flush=True)
    worst = 0.0; worst_name = ""; nmiss = 0; ndiff = 0
    for n, p in trn.items():
        if n not in eng:
            nmiss += 1; continue
        with torch.no_grad():
            d = (eng[n].float() - p.float().to(eng[n].device)).abs().max().item()
        if d > worst:
            worst = d; worst_name = n
        if d > 0:
            ndiff += 1
    print(f"\n[WEIGHTS] params with any diff: {ndiff}/{len(trn)} missing={nmiss} "
          f"| worst max|diff|={worst:.3e} @ {worst_name}", flush=True)
    if worst > 0:
        print("[WEIGHTS] WEIGHTS DIFFER between two bridge-loads -> sync/build is the cause.", flush=True)
    else:
        print("[WEIGHTS] bridge-loaded weights are bitwise-identical -> not the build; "
              "if the run still diverges it's the native SYNC overwrite, not the load.", flush=True)

print("\nRESULT:", "RoPE" if ROPE_IS_CAUSE else "see [WEIGHTS] above", flush=True)
