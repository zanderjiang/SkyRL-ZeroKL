"""Isolate the engine-vs-trainer forward divergence: ATTENTION-MODULE path + weights (RoPE ruled out).

Engine attention = MegatronCoreAttnToVLLM (vLLM Attention layer: KV-cache write + paged read).
Trainer attention = TorchVarlenCoreAttn (direct varlen_attn). Proven equal on RANDOM q/k/v only.
Here we test them on REAL activations through the full model:
  A. capture engine logits for one sequence via vLLM (forward hook on the wrapper).
  B. swap wrapper.gpt.core_attention -> TorchVarlenCoreAttn, run the SAME sequence standalone.
  compare per-token logprobs. If they diverge (~0.2 on some tokens) -> attention-module path is the cause.
Then (if attention matches) compare weights to a 2nd bridge-built GPTModel (NVTE-crash fixed).
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
from skyrl.backends.skyrl_train.zerokl.megatron_varlen_attn import (  # noqa: E402
    swap_trainer_core_attention_varlen, enable_trainer_batch_invariant)
from vllm import LLM, SamplingParams  # noqa: E402

apply_vllm_zerokl_env()
register_gptmodel_to_vllm()
print("=== building engine ===", flush=True)
llm = LLM(model=MODEL, hf_overrides={"architectures": [VLLM_MODEL_NAME]}, attention_backend="CUSTOM",
          dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.45, max_model_len=2048,
          enable_prefix_caching=False, enable_chunked_prefill=False, trust_remote_code=True)
wrapper = find_inprocess_gptmodel(llm)
assert wrapper is not None and hasattr(wrapper, "gpt")
tok = llm.get_tokenizer()

# a fixed sequence (prompt + a few tokens), score its prompt_logprobs to drive a full-seq prefill.
prompt = ("Solve step by step: a train travels at 60 mph for 2.5 hours, then 40 mph for 1.5 hours. "
          "Explain how to find the total distance and discuss factors affecting it. Answer carefully.")
ids = tok(prompt, add_special_tokens=False).input_ids[:200]
L = len(ids)

# ---- A. engine logits via vLLM (hook the wrapper's forward output = logits [tokens, vocab]) ----
_cap = {}
def _hook(_m, _inp, _out):
    _cap["logits"] = _out.detach().float().cpu()
h = wrapper.register_forward_hook(_hook)
llm.generate([{"prompt_token_ids": ids}], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
h.remove()
eng_logits = _cap["logits"]            # [L (+1), vocab]
eng_logits = eng_logits[:L]
print(f"[A] engine logits captured: {tuple(eng_logits.shape)}", flush=True)

# ---- B. trainer attention: swap core_attention -> TorchVarlenCoreAttn, run standalone ----
enable_trainer_batch_invariant()
swap_trainer_core_attention_varlen(wrapper.gpt)
with torch.no_grad():
    pos = torch.arange(L, device=wrapper.gpt.embedding.word_embeddings.weight.device)
    wrapper._rope.set_positions(pos)   # RoPE proven == stock; index by 0..L-1
    ids_t = torch.tensor(ids, device=pos.device).unsqueeze(0)
    out = wrapper.gpt(input_ids=ids_t, position_ids=pos.unsqueeze(0), attention_mask=None)
    if out.dim() == 3:
        out = out.reshape(-1, out.shape[-1])
    trn_logits = out.detach().float().cpu()[:L]
print(f"[B] trainer-attn logits: {tuple(trn_logits.shape)}", flush=True)

# ---- compare per-token logprob of the actual next token (the metric's quantity) ----
import torch.nn.functional as F
def per_tok_lp(logits):
    lp = F.log_softmax(logits.float(), dim=-1)
    nxt = torch.tensor(ids[1:] + [ids[-1]])
    return lp.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
elp = per_tok_lp(eng_logits); tlp = per_tok_lp(trn_logits)
d = (elp - tlp).abs()
logit_d = (eng_logits - trn_logits).abs()
print(f"\n[ATTN-MODULE] max|engine_logit - trainer_logit| = {float(logit_d.max()):.3e}", flush=True)
print(f"[ATTN-MODULE] per-token logprob diff: max={float(d.max()):.4e} mean={float(d.mean()):.4e} "
      f"#>0.05={int((d>0.05).sum())}/{L}", flush=True)
_bad = (d > 0.01).nonzero(as_tuple=True)[0].tolist()[:12]
print(f"[ATTN-MODULE] bad token positions (>0.01): {_bad}", flush=True)
ATTN_CAUSE = float(d.max()) > 1e-3
print("RESULT:", "ATTENTION-MODULE path is the cause" if ATTN_CAUSE else
      "attention module bitwise -> check weights", flush=True)
