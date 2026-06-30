"""Test: does the PRODUCTION megatron-bridge build a LOCAL-spec (no-TE) MiMo GPTModel WITH
correct HF weights? If yes -> save state_dict for the nightly venv. Safe (build+check only)."""
import os, warnings; warnings.filterwarnings("ignore")
import torch
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29601")
os.environ.setdefault("RANK","0"); os.environ.setdefault("WORLD_SIZE","1"); os.environ.setdefault("LOCAL_RANK","0")
import torch.distributed as dist
dist.init_process_group(backend="nccl", world_size=1, rank=0)
from megatron.core import parallel_state, tensor_parallel
parallel_state.initialize_model_parallel(1,1)
tensor_parallel.model_parallel_cuda_manual_seed(0)
from megatron.bridge import AutoBridge
M="/mnt/local_storage/models/MiMo-7B-RL"
b=AutoBridge.from_hf_pretrained(M, trust_remote_code=True)
p=b.to_megatron_provider(load_weights=True)
print("default transformer_impl:", getattr(p,"transformer_impl",None))
p.transformer_impl="local"
if getattr(p,"mtp_num_layers",None): p.mtp_num_layers=None
p.tensor_model_parallel_size=1; p.pipeline_model_parallel_size=1
p.pipeline_dtype=torch.bfloat16; p.apply_rope_fusion=False
p.gradient_accumulation_fusion=False
p.masked_softmax_fusion=False
p.bias_activation_fusion=False
p.bias_dropout_fusion=False
# rope base workaround (transformers v5)
_hf=b.hf_pretrained.config if hasattr(b,'hf_pretrained') else None
try:
    rp=getattr(_hf,"rope_parameters",None) or getattr(_hf,"rope_scaling",None)
    if isinstance(rp,dict) and rp.get("rope_theta"): p.rotary_base=rp["rope_theta"]
except Exception: pass
p.finalize()
m=p.provide_distributed_model(wrap_with_ddp=False)
gpt=m[0].module if hasattr(m[0],"module") else m[0]
names=[n for n,_ in gpt.named_parameters()]
print("local-spec GPTModel params:", len(names))
print("sample names:", names[:6])
print("has q_layernorm?", any("q_layernorm" in n for n in names), "| input_layernorm?", any("input_layernorm" in n for n in names), "| linear_qkv?", any("linear_qkv" in n for n in names))
# weight sanity: norm of a big weight
w=dict(gpt.named_parameters())
emb=w.get("embedding.word_embeddings.weight"); out=w.get("output_layer.weight")
print("embedding norm:", float(emb.float().norm()) if emb is not None else None, "| output norm:", float(out.float().norm()) if out is not None else None)
# quick forward sanity (is it coherent? check top token for a trivial input)
gpt.eval()
ids=torch.tensor([[1,2,3,4,5]],device="cuda")
pos=torch.arange(5,device="cuda").unsqueeze(0)
with torch.no_grad():
    lg=gpt(input_ids=ids, position_ids=pos, attention_mask=None)
print("forward logits shape:", tuple(lg.shape), "finite:", torch.isfinite(lg).all().item())
# save state_dict for nightly transfer (bf16, cpu)
import torch as _t
_raw=gpt.state_dict()
sd={k:v.detach().to(torch.bfloat16).cpu() for k,v in _raw.items() if isinstance(v,_t.Tensor)}
# dump full param/buffer inventory for the nightly remap
with open("/mnt/local_storage/mimo_local_keys.txt","w") as f:
    for k,v in _raw.items():
        shp = tuple(v.shape) if isinstance(v,_t.Tensor) else type(v).__name__
        f.write(k + "\t" + str(shp) + "\n")
print("biases present:", [k for k in sd if k.endswith(".bias")][:6])
torch.save(sd, "/mnt/local_storage/mimo_local_sd.pt")
print("SAVED /mnt/local_storage/mimo_local_sd.pt  (", len(sd), "tensors )")
print("PROD-LOCAL-BUILD OK")
