"""Minimal standalone vLLM attention backend that uses PyTorch-native varlen FlashAttention
(torch.nn.attention.varlen.varlen_attn_out) with num_splits=1, giving BITWISE
decode==prefill. Adapted from TorchTitan's
torchtitan/experiments/rl/models/attention.py, with all torchtitan-internal imports
removed so it stands alone on vllm-nightly + torch-nightly.

Import this module BEFORE constructing the vLLM engine and set
    VLLM_ATTENTION_BACKEND=CUSTOM
    VLLM_ENABLE_V1_MULTIPROCESSING=0
so the @register_backend(CUSTOM) registration is visible to the (in-process) engine.
"""
import logging
import os
from typing import Any

import torch
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
)
from torch.nn.attention.varlen import AuxRequest

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

logger = logging.getLogger(__name__)

# Force num_splits=1 for the varlen kernel so it is query-length-invariant
# (=> bitwise decode==prefill at ALL lengths). Default ON; set
# VARLEN_FORCE_NUM_SPLITS_1=0 to disable.
_FORCE_NUM_SPLITS_1 = os.environ.get("VARLEN_FORCE_NUM_SPLITS_1", "1") == "1"


def _has_sm90() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


@register_backend(AttentionBackendEnum.CUSTOM)
class PyTorchVarlenAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return PyTorchVarlenAttentionImpl

    @staticmethod
    def get_builder_cls():
        class PyTorchVarlenAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
            _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

        return PyTorchVarlenAttentionMetadataBuilder


class PyTorchVarlenAttentionImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.out_transform = None  # no epilogue
        self.enable_gqa = self.num_heads > self.num_kv_heads
        if _has_sm90():
            if current_flash_attention_impl() != "FA3":
                activate_flash_attention_impl("FA3")
        else:
            logger.warning("FA3 not available (requires SM 9.0+), falling back to FA2.")

    @eager_break_during_capture
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if not getattr(PyTorchVarlenAttentionImpl, "_FWD_LOGGED", False):
            PyTorchVarlenAttentionImpl._FWD_LOGGED = True
            print("[varlen_backend] PyTorchVarlenAttentionImpl.forward IS EXECUTING "
                  "(torch.nn.attention.varlen.varlen_attn_out, num_splits="
                  f"{1 if _FORCE_NUM_SPLITS_1 else 'auto'})", flush=True)

        # Re-read live per-layer metadata + kv_cache from forward context
        attn_metadata, _, kv_cache, _ = get_attention_context(layer.layer_name)

        if attn_metadata is None:
            return output.fill_(0)

        attn_type = self.attn_type
        num_actual_tokens = attn_metadata.num_actual_tokens

        assert attn_type not in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ), "Encoder-only attention not supported yet."

        key_cache, value_cache = kv_cache.unbind(1)

        assert not self.kv_cache_dtype.startswith("fp8"), "FP8 KV cache not supported."
        assert not attn_metadata.use_cascade, "Cascade not supported yet."

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        assert self.dcp_world_size == 1, "DCP not supported yet."

        if not attn_metadata.causal:
            raise RuntimeError("Non-causal attention not supported yet.")

        if self.sliding_window == (-1, -1):
            sliding_window_size = (-1, 0)
        else:
            sliding_window_size = self.sliding_window

        assert self.alibi_slopes is None, "Alibi slopes not supported yet."

        if current_flash_attention_impl() == "FA3":
            cu_seqlens_k = None
        else:
            num_seqs = seqused_k.shape[0]
            cu_seqlens_k = torch.zeros(
                num_seqs + 1, dtype=torch.int32, device=query.device
            )
            cu_seqlens_k[1:] = torch.cumsum(seqused_k, dim=0)

        extra_kwargs: dict[str, Any] = {}
        fa_impl = current_flash_attention_impl()
        # Force num_splits=1 => bitwise decode==prefill. Always force when
        # _FORCE_NUM_SPLITS_1, plus always for FA2/None (NaN workaround).
        if fa_impl in (None, "FA2") or _FORCE_NUM_SPLITS_1:
            extra_kwargs["num_splits"] = 1

        if self.enable_gqa:
            extra_kwargs["enable_gqa"] = True

        result = torch.nn.attention.varlen.varlen_attn_out(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            scale=self.scale,
            window_size=sliding_window_size,
            block_table=block_table,
            seqused_k=seqused_k,
            **extra_kwargs,
        )
        return result
