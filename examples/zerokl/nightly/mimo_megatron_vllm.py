"""REAL MiMo-7B local-spec Megatron GPTModel served INSIDE vLLM-1.0 (nightly stack).

Builds the MiMo-7B GPTModel with the LOCAL spec (no TransformerEngine), loads the
production-converted weights (/mnt/local_storage/mimo_local_sd.pt, megatron-native layout but
with TE-FUSED norm names) by remapping the fused-norm tensors onto the unfused local-spec param
names, swaps attention to vLLM paged Attention, and registers into vLLM-1.0.

Goal: COHERENT generation (weights correct) + BITWISE decode==prefill (varlen num_splits=1).
"""
from __future__ import annotations
import itertools, logging, os
import torch
from torch import nn

logger = logging.getLogger("mimo_megatron_vllm")

VLLM_MODEL_NAME = "MegatronMiMoForCausalLM"
CONFIG_FORMAT = "megatron_mimo"
SD_PATH = os.environ.get("MIMO_LOCAL_SD", "/mnt/local_storage/mimo_local_sd.pt")

MIMO = dict(
    vocab_size=151680, hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
    head_dim=128, num_hidden_layers=36, intermediate_size=11008,
    max_position_embeddings=int(os.environ.get("MIMO_MAXLEN", "2048")),
    rope_theta=640000, rms_norm_eps=1e-5,
)


def mimo_hf_config_dict():
    t = MIMO
    return {
        "architectures": [VLLM_MODEL_NAME], "model_type": CONFIG_FORMAT,
        "vocab_size": t["vocab_size"], "hidden_size": t["hidden_size"],
        "num_attention_heads": t["num_attention_heads"], "num_key_value_heads": t["num_key_value_heads"],
        "head_dim": t["head_dim"], "max_position_embeddings": t["max_position_embeddings"],
        "num_hidden_layers": t["num_hidden_layers"], "intermediate_size": t["intermediate_size"],
        "rope_theta": t["rope_theta"], "rms_norm_eps": t["rms_norm_eps"],
        "tie_word_embeddings": False, "torch_dtype": "bfloat16", "bos_token_id": 0, "eos_token_id": 1,
    }


class MegatronCoreAttnToVLLM(nn.Module):
    _layer_counter = itertools.count()
    def __init__(self, *, num_heads, num_kv_heads, head_dim, scale):
        super().__init__()
        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.attention import Attention
        vllm_config = get_current_vllm_config()
        cache_config = getattr(vllm_config, "cache_config", None)
        layer_id = next(MegatronCoreAttnToVLLM._layer_counter)
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.vllm_attn = Attention(
            num_heads=num_heads, head_size=head_dim, scale=scale, num_kv_heads=num_kv_heads,
            cache_config=cache_config, quant_config=None,
            prefix=f"decoder.layers.{layer_id}.self_attention.core_attention",
        )
    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        sq, b = query.shape[0], query.shape[1]
        q = query.reshape(sq * b, self.num_heads * self.head_dim).contiguous()
        k = key.reshape(sq * b, self.num_kv_heads * self.head_dim).contiguous()
        v = value.reshape(sq * b, self.num_kv_heads * self.head_dim).contiguous()
        out = self.vllm_attn(q, k, v)
        return out.reshape(sq, b, self.num_heads * self.head_dim)


def swap_core_attention(gpt, *, num_heads, num_kv_heads, head_dim, scale):
    inner = gpt.module if hasattr(gpt, "module") else gpt
    n = 0
    for layer in inner.decoder.layers:
        sa = getattr(layer, "self_attention", None)
        if sa is None:
            continue
        sa.core_attention = MegatronCoreAttnToVLLM(num_heads=num_heads, num_kv_heads=num_kv_heads,
                                                   head_dim=head_dim, scale=scale)
        n += 1
    logger.info("[mimo] swapped core_attention on %d layers", n)
    return n


class _PositionIndexedRoPE(nn.Module):
    def __init__(self, orig, max_pos):
        super().__init__()
        self._orig = orig
        with torch.no_grad():
            self._emb_full = orig(max_pos)
        self._positions = None
    def set_positions(self, positions):
        self._positions = positions
    def forward(self, max_seq_len, *a, **k):
        if self._positions is not None:
            return self._emb_full.to(self._positions.device)[self._positions]
        return self._orig(max_seq_len, *a, **k)
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig, name)


def _remap_fused_to_local(src_sd, target_names):
    """Map production (TE-fused norm) state_dict keys onto the unfused local-spec param names.
    Fused -> local: linear_qkv.layer_norm_weight -> input_layernorm.weight;
                    mlp.linear_fc1.layer_norm_weight -> pre_mlp_layernorm.weight.
    All other names are identical. Returns (mapped_sd, missing, unexpected)."""
    tset = set(target_names)
    out = {}
    for k, v in src_sd.items():
        nk = k
        if k.endswith("self_attention.linear_qkv.layer_norm_weight"):
            nk = k.replace("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight")
        elif k.endswith("mlp.linear_fc1.layer_norm_weight"):
            nk = k.replace("mlp.linear_fc1.layer_norm_weight", "pre_mlp_layernorm.weight")
        if nk in tset:
            out[nk] = v
    missing = sorted(tset - set(out))
    unexpected = sorted(set(src_sd) - {k for k in src_sd})  # placeholder
    return out, missing


def build_mimo_gptmodel(device, dtype=torch.bfloat16):
    import torch.distributed as dist
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    if not dist.is_initialized():
        for k, v in (("MASTER_ADDR","127.0.0.1"),("MASTER_PORT","29588"),("RANK","0"),("WORLD_SIZE","1"),("LOCAL_RANK","0")):
            os.environ.setdefault(k, v)
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)
    tensor_parallel.model_parallel_cuda_manual_seed(0)
    t = MIMO; H = t["hidden_size"]
    config = TransformerConfig(
        num_layers=t["num_hidden_layers"], hidden_size=H,
        num_attention_heads=t["num_attention_heads"], num_query_groups=t["num_key_value_heads"],
        kv_channels=t["head_dim"], ffn_hidden_size=t["intermediate_size"],
        use_cpu_initialization=False, pipeline_dtype=dtype, bf16=(dtype==torch.bfloat16),
        params_dtype=dtype, tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        sequence_parallel=False, add_bias_linear=False, add_qkv_bias=True,
        gated_linear_unit=True, normalization="RMSNorm", apply_rope_fusion=False,
        gradient_accumulation_fusion=False, layernorm_epsilon=t["rms_norm_eps"],
    )
    layer_spec = get_gpt_layer_local_spec(num_experts=None, moe_grouped_gemm=False,
                                          qk_layernorm=False, normalization="RMSNorm")
    model = GPTModel(config=config, transformer_layer_spec=layer_spec, vocab_size=t["vocab_size"],
                     max_sequence_length=t["max_position_embeddings"], pre_process=True,
                     post_process=True, position_embedding_type="rope",
                     rotary_base=int(t["rope_theta"])).to(device).to(dtype)
    # load production-converted weights with the fused->local remap
    src = torch.load(SD_PATH, map_location="cpu")
    tgt_names = [n for n, _ in model.named_parameters()]
    mapped, missing = _remap_fused_to_local(src, tgt_names)
    bufs = dict(model.named_buffers())
    sd_full = {**{n: p for n, p in model.state_dict().items()}}  # start from current (for buffers)
    for k, v in mapped.items():
        sd_full[k] = v.to(dtype)
    incompat = model.load_state_dict(sd_full, strict=False)
    print(f"[MIMO-WRAP] loaded weights: matched={len(mapped)} target_params={len(tgt_names)} "
          f"missing={len(missing)} (e.g. {missing[:4]}) load_missing={len(incompat.missing_keys)} "
          f"unexpected={len(incompat.unexpected_keys)}", flush=True)
    model.eval()
    return model, config


class MegatronMiMoVLLMWrapper(nn.Module):
    is_text_generation_model = True
    supports_pp = False
    supports_multimodal = False
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        device = vllm_config.device_config.device
        dtype = vllm_config.model_config.dtype
        self.gpt, cfg = build_mimo_gptmodel(device, dtype=dtype)
        head_dim = cfg.kv_channels
        swap_core_attention(self.gpt, num_heads=cfg.num_attention_heads,
                            num_kv_heads=cfg.num_query_groups, head_dim=head_dim, scale=head_dim ** -0.5)
        max_pos = int(getattr(vllm_config.model_config, "max_model_len", MIMO["max_position_embeddings"]))
        self._rope = _PositionIndexedRoPE(self.gpt.rotary_pos_emb, max_pos)
        self.gpt.rotary_pos_emb = self._rope
        with torch.no_grad():
            _w = next((p for n, p in self.gpt.named_parameters() if "embedding" in n), None)
            _wn = float(_w.float().norm()) if _w is not None else -1.0
        print(f"[MIMO-WRAP] built; embedding_norm={_wn:.3f} (expect ~359.957)", flush=True)
    def embed_input_ids(self, input_ids):
        return self.gpt.embedding(input_ids=input_ids.unsqueeze(0), position_ids=None)
    def get_input_embeddings(self, input_ids):
        return self.embed_input_ids(input_ids)
    def forward(self, input_ids=None, positions=None, inputs_embeds=None, **kwargs):
        tokens = input_ids.unsqueeze(0); pos = positions.unsqueeze(0)
        self._rope.set_positions(positions.reshape(-1))
        out = self.gpt(input_ids=tokens, position_ids=pos, attention_mask=None)
        if out.dim() == 3:
            out = out.reshape(-1, out.shape[-1])
        return out
    def compute_logits(self, hidden_states, sampling_metadata=None):
        return hidden_states
    def load_weights(self, weights_iter):
        all_names = {"gpt." + n for n, _ in self.gpt.named_parameters()}
        for _ in weights_iter:
            pass
        return all_names


def register_mimo_to_vllm():
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.config import register_config_parser
    from vllm.transformers_utils.config_parser_base import ConfigParserBase
    from transformers import PretrainedConfig
    ModelRegistry.register_model(VLLM_MODEL_NAME, MegatronMiMoVLLMWrapper)
    @register_config_parser(CONFIG_FORMAT)
    class MiMoConfigParser(ConfigParserBase):
        def parse(self, model, trust_remote_code, revision=None, code_revision=None, **kwargs):
            d = mimo_hf_config_dict()
            return d, PretrainedConfig.from_dict(d)
    return VLLM_MODEL_NAME, CONFIG_FORMAT
