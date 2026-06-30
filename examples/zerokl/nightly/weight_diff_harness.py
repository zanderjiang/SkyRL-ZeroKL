"""Compare the engine's bridge-loaded GPTModel weights to a SECOND bridge-built GPTModel (same HF
checkpoint, same local-spec config). RoPE + attention proven bitwise; if weights differ, that (the
bridge load being non-deterministic / some params left at random init) is the run's residual cause.
NVTE crash fixed: attention_backend=local (no-TE) + NVTE_* unset.
"""
import os
for _v in ("NVTE_FUSED_ATTN", "NVTE_FLASH_ATTN", "NVTE_UNFUSED_ATTN"):
    os.environ.pop(_v, None)
os.environ.setdefault("SKYRL_ZERO_KL", "1")
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")
os.environ.setdefault("SKYRL_ZEROKL_ENGINE_LOAD_WEIGHTS", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
MODEL = os.environ.get("ZEROKL_MODEL", "/mnt/local_storage/models/MiMo-7B-RL")

import torch  # noqa: E402
import skyrl.backends.skyrl_train.zerokl.varlen_backend as varlen_backend  # noqa: E402,F401
from skyrl.backends.skyrl_train.zerokl.gptmodel_vllm import (  # noqa: E402
    register_gptmodel_to_vllm, VLLM_MODEL_NAME, find_inprocess_gptmodel)
from skyrl.backends.skyrl_train.zerokl import apply_vllm_zerokl_env  # noqa: E402
from vllm import LLM  # noqa: E402

apply_vllm_zerokl_env()
register_gptmodel_to_vllm()
print("=== building engine (low gpu_mem to leave room for 2nd model) ===", flush=True)
llm = LLM(model=MODEL, hf_overrides={"architectures": [VLLM_MODEL_NAME]}, attention_backend="CUSTOM",
          dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.30, max_model_len=1024,
          enable_prefix_caching=False, enable_chunked_prefill=False, trust_remote_code=True)
wrapper = find_inprocess_gptmodel(llm)
eng = dict(wrapper.gpt.named_parameters())
print(f"engine gpt params={len(eng)}", flush=True)

print("=== building 2nd bridge GPTModel (attention_backend=local) ===", flush=True)
for _v in ("NVTE_FUSED_ATTN", "NVTE_FLASH_ATTN", "NVTE_UNFUSED_ATTN"):
    os.environ.pop(_v, None)
from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import local_layer_spec
from megatron.core.transformer.enums import AttnBackend
from transformers import AutoConfig
b = AutoBridge.from_hf_pretrained(MODEL, trust_remote_code=True)
mp = b.to_megatron_provider(load_weights=True)
mp.tensor_model_parallel_size = 1
mp.pipeline_model_parallel_size = 1
mp.expert_model_parallel_size = 1
mp.expert_tensor_parallel_size = 1
mp.pipeline_dtype = torch.bfloat16
mp.apply_rope_fusion = False
mp.gradient_accumulation_fusion = False
mp.attention_backend = AttnBackend.local          # <-- the NVTE-crash fix
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
trn = dict(gpt2.named_parameters())
print(f"2nd gpt params={len(trn)}", flush=True)

# param-by-param compare
rows = []
for n, p in trn.items():
    if n not in eng:
        rows.append((float("inf"), n, "MISSING_IN_ENGINE")); continue
    with torch.no_grad():
        d = (eng[n].float() - p.float().to(eng[n].device)).abs().max().item()
    rows.append((d, n, ""))
rows.sort(reverse=True)
ndiff = sum(1 for d, _, _ in rows if d > 0)
print(f"\n[WEIGHTS] params with any diff: {ndiff}/{len(trn)}", flush=True)
print("[WEIGHTS] top-12 differing params (max|diff|, name):", flush=True)
for d, n, note in rows[:12]:
    print(f"    {d:.4e}  {n}  {note}", flush=True)
also_eng_only = [n for n in eng if n not in trn]
print(f"[WEIGHTS] params in engine but not 2nd: {also_eng_only[:8]}", flush=True)
print("RESULT:", "WEIGHTS DIFFER (bridge non-deterministic / random-init params)" if ndiff > 0 else
      "weights bitwise-identical -> residual is the distributed native SYNC, not the load", flush=True)
