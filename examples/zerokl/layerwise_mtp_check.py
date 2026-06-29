"""Layer-wise bitwise check: does disabling MTP perturb the MAIN next-token logits?

Context (see ZEROKL_DIFF_HANDOFF.md): the full SkyRL pipeline shows
``rollout_train_logprobs_abs_diff_mean ~= 0.0104`` for MiMo-7B, while the single-process
standalone (``examples/zerokl/dapo_zerokl.py``) is bitwise (~1e-6). The one structural
difference found so far: the pipeline trainer disables MTP
(``megatron_worker.py:408  provider.mtp_num_layers = None``) so its GPTModel has 255 params,
while the engine (``gptmodel_vllm.py``) keeps MTP -> 266 params. The standalone trainer also
keeps MTP, hence its parity.

This script builds the SAME MiMo GPTModel TWO ways in ONE process -- MTP-on (engine/standalone
layout) and MTP-off (pipeline-trainer layout) -- loads identical HF weights into both, and runs
the SAME input through both with the zero-KL kernels active. It then reports, per decoder layer,
the max/mean abs diff of the hidden states, plus the final-logits / per-token-logprob diff.

Interpretation:
  * hidden states + logits bitwise-identical  -> MTP-disable is harmless; the 0.0104 lives in
    the SkyRL forward WRAPPER or cross-process sync, NOT in MTP. (exonerate MTP)
  * hidden states diverge at some layer / logits differ ~0.01  -> MTP-disable changes the main
    forward; that IS the regression. The first diverging layer localizes WHY.

Usage:
  CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    /home/ray/skyrl-zerokl-venv/.venv/bin/python -m examples.zerokl.layerwise_mtp_check \
    --model /mnt/local_storage/models/MiMo-7B-RL
"""
import argparse
import os
import sys

os.environ.setdefault("HF_HOME", "/mnt/local_storage/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["VLLM_USE_AOT_COMPILE"] = "0"
os.environ["NVTE_FLASH_ATTN"] = "1"
os.environ["NVTE_FUSED_ATTN"] = "0"
os.environ.setdefault("NCCL_ALGO", "allreduce:tree")
os.environ.setdefault("NCCL_MIN_NCHANNELS", "1")
os.environ.setdefault("NCCL_MAX_NCHANNELS", "1")
os.environ["SKYRL_ZERO_KL"] = "1"

import torch  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/mnt/local_storage/models/MiMo-7B-RL")
    p.add_argument("--seqlen", type=int, default=128)
    p.add_argument("--n_prompt", type=int, default=16)
    return p.parse_args()


def build_gptmodel(model_path, *, disable_mtp: bool):
    """Replicate the bridge build used by both the engine (gptmodel_vllm) and the trainer
    (megatron_worker). The ONLY toggle is mtp_num_layers (pipeline trainer sets it None)."""
    from megatron.bridge import AutoBridge
    from megatron.core.transformer.enums import AttnBackend

    b = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
    mp = b.to_megatron_provider(load_weights=True)
    mp.tensor_model_parallel_size = 1
    mp.pipeline_model_parallel_size = 1
    mp.expert_model_parallel_size = 1
    mp.expert_tensor_parallel_size = 1
    mp.pipeline_dtype = torch.bfloat16
    mp.apply_rope_fusion = False
    mp.attention_backend = AttnBackend.flash
    mp.gradient_accumulation_fusion = False
    # MiMo rope_theta workaround (transformers v5 moves it into rope_parameters)
    hf = b.hf_pretrained.config if hasattr(b, "hf_pretrained") else None
    rp = None
    if hf is not None:
        rp = getattr(hf, "rope_parameters", None) or getattr(hf, "rope_scaling", None)
        if isinstance(rp, dict) and rp.get("rope_theta"):
            mp.rotary_base = rp["rope_theta"]
        elif getattr(hf, "rope_theta", None):
            mp.rotary_base = hf.rope_theta
    mtp_before = getattr(mp, "mtp_num_layers", None)
    if disable_mtp and getattr(mp, "mtp_num_layers", None):
        mp.mtp_num_layers = None  # exactly what megatron_worker.py:408 does
    mp.finalize()
    gpt_list = mp.provide_distributed_model(wrap_with_ddp=False)
    gpt = gpt_list[0].module if hasattr(gpt_list[0], "module") else gpt_list[0]
    print(f"[LW] build disable_mtp={disable_mtp}: mtp_num_layers {mtp_before} -> "
          f"{getattr(mp, 'mtp_num_layers', None)}, rotary_base={getattr(mp, 'rotary_base', '?')}, "
          f"params={sum(1 for _ in gpt.named_parameters())}", flush=True)
    return gpt


def capture_layerwise(gpt, input_ids, position_ids, scoring_mode):
    """Run gpt forward, capturing each decoder layer's output hidden states + final logits."""
    caps = {}
    handles = []

    def mk_hook(i):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            caps[i] = h.detach().float().clone()
        return hook

    for i, layer in enumerate(gpt.decoder.layers):
        handles.append(layer.register_forward_hook(mk_hook(i)))
    # final layernorm output (pre-logits hidden), if present
    if getattr(gpt.decoder, "final_layernorm", None) is not None:
        def fn_hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            caps["final_ln"] = h.detach().float().clone()
        handles.append(gpt.decoder.final_layernorm.register_forward_hook(fn_hook))
    try:
        with torch.no_grad(), scoring_mode():
            logits = gpt(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
    finally:
        for h in handles:
            h.remove()
    return caps, logits.detach().float()


def main():
    args = parse_args()
    sys.path.insert(0, "/home/ray/default/SkyRL-ZeroKL")
    from transformers import AutoTokenizer
    from skyrl.backends.skyrl_train.zerokl import apply_megatron_zerokl_patches, scoring_mode

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ---- build BOTH models (MTP-on like engine/standalone; MTP-off like pipeline trainer) ----
    print("[LW] building MTP-ON model (engine/standalone layout) ...", flush=True)
    gpt_on = build_gptmodel(args.model, disable_mtp=False)
    print("[LW] building MTP-OFF model (pipeline-trainer layout) ...", flush=True)
    gpt_off = build_gptmodel(args.model, disable_mtp=True)
    apply_megatron_zerokl_patches()  # global zero-KL kernels (fp32 rope, vops norm, batch-invariant)

    # ---- param-name symmetric difference (the 11) ----
    names_on = set(n for n, _ in gpt_on.named_parameters())
    names_off = set(n for n, _ in gpt_off.named_parameters())
    only_on = sorted(names_on - names_off)
    only_off = sorted(names_off - names_on)
    print(f"\n[LW] params: MTP-on={len(names_on)} MTP-off={len(names_off)} "
          f"(on-only={len(only_on)}, off-only={len(only_off)})")
    print(f"[LW] MTP-on-ONLY params ({len(only_on)}):")
    for n in only_on:
        print(f"        {n}")
    if only_off:
        print(f"[LW] MTP-off-ONLY params ({len(only_off)}): {only_off}")

    # ---- verify SHARED params are bitwise-identical at init (both bridge-load same HF ckpt) ----
    pon = dict(gpt_on.named_parameters())
    poff = dict(gpt_off.named_parameters())
    shared = sorted(names_on & names_off)
    max_w = 0.0
    n_diff = 0
    for n in shared:
        d = (pon[n].float() - poff[n].float()).abs().max().item()
        if d > 0:
            n_diff += 1
            max_w = max(max_w, d)
    print(f"\n[LW] shared params bitwise-equal: {len(shared) - n_diff}/{len(shared)} "
          f"(max weight diff over shared = {max_w:.3e})")

    # ---- identical input through both ----
    torch.manual_seed(0)
    text = ("Solve the problem step by step. What is the integral of x squared from zero to "
            "three, and explain each step carefully so a student can follow along.")
    ids = tok(text, add_special_tokens=False).input_ids[: args.seqlen]
    input_ids = torch.tensor([ids], device="cuda")
    position_ids = torch.arange(len(ids), device="cuda").unsqueeze(0)
    print(f"\n[LW] forward on {len(ids)} tokens ...", flush=True)

    caps_on, logits_on = capture_layerwise(gpt_on, input_ids, position_ids, scoring_mode)
    caps_off, logits_off = capture_layerwise(gpt_off, input_ids, position_ids, scoring_mode)

    # ---- per-layer hidden-state diff ----
    print("\n[LW] ===== per-layer hidden-state |MTP_on - MTP_off| =====")
    n_layers = len(gpt_on.decoder.layers)
    first_div = None
    for i in range(n_layers):
        a, b = caps_on.get(i), caps_off.get(i)
        if a is None or b is None:
            continue
        dmax = (a - b).abs().max().item()
        dmean = (a - b).abs().mean().item()
        flag = ""
        if dmax > 0 and first_div is None:
            first_div = i
            flag = "  <-- FIRST DIVERGENCE"
        if i < 4 or i % 8 == 0 or dmax > 0 or i == n_layers - 1:
            print(f"[LW] layer {i:>2}: max={dmax:.3e} mean={dmean:.3e}{flag}")
    if "final_ln" in caps_on and "final_ln" in caps_off:
        a, b = caps_on["final_ln"], caps_off["final_ln"]
        print(f"[LW] final_ln : max={(a-b).abs().max().item():.3e} mean={(a-b).abs().mean().item():.3e}")

    # ---- final logits / logprob diff ----
    lg_on = logits_on.reshape(-1, logits_on.shape[-1])[:, : len(tok)]
    lg_off = logits_off.reshape(-1, logits_off.shape[-1])[:, : len(tok)]
    lp_on = torch.log_softmax(lg_on, dim=-1)
    lp_off = torch.log_softmax(lg_off, dim=-1)
    resp = torch.tensor(ids, device="cuda")
    idx = torch.arange(0, len(ids) - 1, device="cuda")
    tok_lp_on = lp_on[idx].gather(1, resp[1:, None]).squeeze(1)
    tok_lp_off = lp_off[idx].gather(1, resp[1:, None]).squeeze(1)
    dlp = (tok_lp_on - tok_lp_off).abs()
    print(f"\n[LW] logits |on-off| max={(lg_on-lg_off).abs().max().item():.3e}")
    print(f"[LW] per-token logprob |on-off|: mean={dlp.mean().item():.5f} max={dlp.max().item():.5f} "
          f"(this is the analog of rollout_train_logprobs_abs_diff)")
    if first_div is None:
        print("\n[LW] VERDICT: hidden states bitwise-identical across ALL layers -> MTP-disable does "
              "NOT change the main forward. The 0.0104 is in the wrapper / sync, not MTP.")
    else:
        print(f"\n[LW] VERDICT: hidden states first diverge at decoder layer {first_div} -> disabling "
              "MTP perturbs the main forward. MTP-disable is implicated in the 0.0104.")


if __name__ == "__main__":
    main()
