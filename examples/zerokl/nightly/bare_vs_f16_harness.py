"""Decisive test: ENGINE forward (bare GPTModel, what gptmodel_vllm runs: self.gpt = gpt[0].module)
vs TRAINER forward (Float16Module-wrapped GPTModel, what megatron_worker runs through
forward_backward_func). SAME GPTModel/weights, SAME varlen attention. If the bare and Float16Module
forwards produce different logits (esp. different dtype / fp32-vs-bf16 residual), that is the run's
residual (~0.2 on sharp tokens). Prints logits dtype + per-token logprob diff.
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
import torch.nn.functional as F  # noqa: E402
import skyrl.backends.skyrl_train.zerokl.varlen_backend as varlen_backend  # noqa: E402,F401
from skyrl.backends.skyrl_train.zerokl.gptmodel_vllm import (  # noqa: E402
    register_gptmodel_to_vllm, VLLM_MODEL_NAME, find_inprocess_gptmodel)
from skyrl.backends.skyrl_train.zerokl import apply_vllm_zerokl_env  # noqa: E402
from skyrl.backends.skyrl_train.zerokl.megatron_varlen_attn import (  # noqa: E402
    swap_trainer_core_attention_varlen, enable_trainer_batch_invariant)
from vllm import LLM  # noqa: E402

apply_vllm_zerokl_env()
register_gptmodel_to_vllm()
print("=== building engine ===", flush=True)
llm = LLM(model=MODEL, hf_overrides={"architectures": [VLLM_MODEL_NAME]}, attention_backend="CUSTOM",
          dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.45, max_model_len=1024,
          enable_prefix_caching=False, enable_chunked_prefill=False, trust_remote_code=True)
wrapper = find_inprocess_gptmodel(llm)
bare = wrapper.gpt                       # engine path (gpt[0].module)
wrapped = wrapper._gpt_list[0]           # trainer path (the Float16Module, if bf16-wrapped)
print(f"bare type={type(bare).__name__}  wrapped[0] type={type(wrapped).__name__}  "
      f"wrapped is Float16Module-ish={hasattr(wrapped, 'module') and wrapped.module is bare}", flush=True)

enable_trainer_batch_invariant()
swap_trainer_core_attention_varlen(bare)   # shared GPTModel -> affects both bare and wrapped

dev = bare.embedding.word_embeddings.weight.device
L = 200
torch.manual_seed(0)
ids = torch.randint(0, 100000, (L,), device=dev)
pos = torch.arange(L, device=dev)
wrapper._rope.set_positions(pos)

with torch.no_grad():
    out_bare = bare(input_ids=ids.unsqueeze(0), position_ids=pos.unsqueeze(0), attention_mask=None)
    wrapper._rope.set_positions(pos)
    out_wrap = wrapped(input_ids=ids.unsqueeze(0), position_ids=pos.unsqueeze(0), attention_mask=None)
ob = out_bare.reshape(-1, out_bare.shape[-1]) if out_bare.dim() == 3 else out_bare
ow = out_wrap.reshape(-1, out_wrap.shape[-1]) if out_wrap.dim() == 3 else out_wrap
print(f"\n[DTYPE] bare(engine) logits dtype={ob.dtype}  wrapped(trainer) logits dtype={ow.dtype}", flush=True)
logit_d = (ob.float() - ow.float()).abs()
print(f"[LOGITS] max|bare - wrapped| = {float(logit_d.max()):.3e}  mean={float(logit_d.mean()):.3e}", flush=True)

# per-token logprob the metric way (engine: (x-amax)-log(sum exp) fp32; trainer same formula fp32)
def lp(logits):
    x = logits.float()
    x = x - torch.amax(x, dim=-1, keepdim=True)
    z = x - x.exp().sum(-1, keepdim=True).float().log()
    nxt = torch.cat([ids[1:], ids[-1:]])
    return z.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
d = (lp(ob) - lp(ow)).abs()
print(f"[LOGPROB] max|diff|={float(d.max()):.4e} mean={float(d.mean()):.4e} #>0.05={int((d>0.05).sum())}/{L}", flush=True)
print("RESULT:", "BARE vs FLOAT16MODULE DIVERGE (precision path is the cause)" if float(d.max()) > 1e-3
      else "bare == wrapped (precision not the cause)", flush=True)
