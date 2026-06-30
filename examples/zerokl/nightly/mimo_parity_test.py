"""Real MiMo-7B local-spec GPTModel in vLLM-1.0: coherent generation + bitwise decode==prefill."""
import os, sys
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
N = int(os.environ.get("PARITY_NTOK", "256"))
BACKEND = os.environ.get("PARITY_BACKEND", "CUSTOM")
MODEL = "/mnt/local_storage/models/MiMo-7B-RL"

if BACKEND == "CUSTOM":
    import varlen_backend  # noqa: F401
import torch  # noqa: E402
import vllm.envs as vllm_envs  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from mimo_megatron_vllm import register_mimo_to_vllm, CONFIG_FORMAT  # noqa: E402

print(f"=== MiMo local-spec in vLLM-1.0 | backend={BACKEND} BI={vllm_envs.VLLM_BATCH_INVARIANT} N={N} ===", flush=True)
print("torch", torch.__version__, "| vllm", __import__("vllm").__version__, flush=True)
register_mimo_to_vllm()

# no-op dummy loader so our build-time loaded weights are not overwritten
import vllm.model_executor.model_loader.weight_utils as _wu
import vllm.model_executor.model_loader.dummy_loader as _dl
_wu.initialize_dummy_weights = lambda *a, **k: None
_dl.initialize_dummy_weights = lambda *a, **k: None

llm = LLM(model=MODEL, config_format=CONFIG_FORMAT, dtype="bfloat16", enforce_eager=True,
          gpu_memory_utilization=0.55, max_model_len=2048, enable_prefix_caching=False,
          enable_chunked_prefill=False, load_format="dummy", trust_remote_code=True,
          **({"attention_backend": BACKEND} if BACKEND else {}))

tok = llm.get_tokenizer()
# ---- (A) COHERENT GENERATION check ----
prompt = "The capital of France is"
pids = tok(prompt, add_special_tokens=False).input_ids
g = llm.generate([{"prompt_token_ids": pids}],
                 SamplingParams(temperature=0.0, max_tokens=40, logprobs=0))[0]
gen_text = tok.decode(g.outputs[0].token_ids)
print("\n[COHERENT-GEN] prompt:", repr(prompt), flush=True)
print("[COHERENT-GEN] continuation:", repr(gen_text), flush=True)

# ---- (B) BITWISE decode==prefill at N tokens ----
prompt2 = ("Solve step by step: a train travels at 60 mph for 2.5 hours, then 40 mph for 1.5 "
           "hours. Explain how to find the total distance and discuss factors affecting it.")
p2 = tok(prompt2, add_special_tokens=False).input_ids[:64]
out = llm.generate([{"prompt_token_ids": p2}],
                   SamplingParams(temperature=0.0, max_tokens=N, logprobs=0, ignore_eos=True))[0]
comp = out.outputs[0]; gen_ids = list(comp.token_ids); n = len(gen_ids)
decode_lps = [comp.logprobs[i][gen_ids[i]].logprob for i in range(n)]
full = list(p2) + gen_ids
out2 = llm.generate([{"prompt_token_ids": full}],
                    SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))[0]
pl = out2.prompt_logprobs; P = len(p2)
prefill_lps = [pl[P + i][full[P + i]].logprob for i in range(n)]
diffs = [abs(decode_lps[i] - prefill_lps[i]) for i in range(n)]
maxd = max(diffs); argmax = diffs.index(maxd); exact0 = sum(1 for d in diffs if d == 0.0)
print(f"\n{'idx':>5} {'decode_lp':>18} {'prefill_lp':>18} {'|d|':>13}")
for i in [0, 1, 7, 31, 63, 127, min(191, n-1), n-1]:
    if i < n:
        print(f"{i:>5} {decode_lps[i]:>18.12f} {prefill_lps[i]:>18.12f} {diffs[i]:>13.3e}")
print(f"\nbackend={BACKEND} BI={vllm_envs.VLLM_BATCH_INVARIANT}", flush=True)
print(f"MAX |decode - prefill| over {n} tokens = {maxd:.6e} (idx {argmax})", flush=True)
print(f"tokens EXACT 0.0: {exact0}/{n}", flush=True)
print("RESULT:", "BITWISE-IDENTICAL (max==0)" if maxd == 0.0 else f"DRIFT max={maxd:.3e}", flush=True)
