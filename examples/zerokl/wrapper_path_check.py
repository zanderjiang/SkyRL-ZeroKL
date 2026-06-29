"""Bisect the SkyRL forward WRAPPER: does remove_left_padding + recover_left_padding +
from_parallel_logits_to_logprobs (the pipeline path) match a direct unpadded forward?

Established so far (layerwise_mtp_check.py): the trainer-layout GPTModel (MTP-off) produces
BITWISE-identical logits to the engine-layout (MTP-on) model, which is itself bitwise-identical
to the vLLM engine (standalone parity). So the model forward + weights are NOT the cause of the
0.0104 rollout_train_logprobs_abs_diff. The remaining single-process suspect is the SkyRL
forward wrapper that the standalone bypasses.

This builds ONE trainer-layout GPTModel and, for each test sequence, computes the response
logprobs TWO ways:
  (A) DIRECT  -- like the standalone: model(unpadded_ids, arange, None) -> log_softmax -> gather
  (B) WRAPPER -- like the pipeline forward_step/loss_func: build a left/right-padded [1, S]
                 microbatch -> remove_left_padding -> model -> recover_left_padding ->
                 from_parallel_logits_to_logprobs -> take the last num_actions slice
and compares them token-by-token. If (A) != (B), the wrapper is the regression and the
per-token dump localizes it. If (A) == (B) bitwise, the wrapper is exonerated and the 0.0104 is
a MULTI-PROCESS effect (weight sync / cumem sleep-wake / engine), to be chased in the live run.

Usage:
  CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    /home/ray/skyrl-zerokl-venv/bin/python -m examples.zerokl.wrapper_path_check \
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
    p.add_argument("--prompt_len", type=int, default=24)
    p.add_argument("--resp_len", type=int, default=40)
    p.add_argument("--left_pad", type=int, default=11)   # left padding (prompt side)
    p.add_argument("--right_pad", type=int, default=7)   # right padding (response side)
    return p.parse_args()


def build_trainer_gptmodel(model_path):
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
    hf = b.hf_pretrained.config if hasattr(b, "hf_pretrained") else None
    if hf is not None:
        rp = getattr(hf, "rope_parameters", None) or getattr(hf, "rope_scaling", None)
        if isinstance(rp, dict) and rp.get("rope_theta"):
            mp.rotary_base = rp["rope_theta"]
        elif getattr(hf, "rope_theta", None):
            mp.rotary_base = hf.rope_theta
    if getattr(mp, "mtp_num_layers", None):
        mp.mtp_num_layers = None  # pipeline-trainer layout
    mp.finalize()
    gpt_list = mp.provide_distributed_model(wrap_with_ddp=False)
    gpt = gpt_list[0].module if hasattr(gpt_list[0], "module") else gpt_list[0]
    return gpt


def direct_logprobs(gpt, full_ids, num_actions, vocab, scoring_mode):
    """Standalone-style: unpadded forward -> log_softmax -> gather response slice."""
    inp = torch.tensor([full_ids], device="cuda")
    pos = torch.arange(len(full_ids), device="cuda").unsqueeze(0)
    with torch.no_grad(), scoring_mode():
        logits = gpt(input_ids=inp, position_ids=pos, attention_mask=None)[0].float()[:, :vocab]
    lp = torch.log_softmax(logits, dim=-1)
    tgt = torch.tensor(full_ids, device="cuda")
    idx = torch.arange(len(full_ids) - 1, device="cuda")
    tok_lp = lp[idx].gather(1, tgt[1:, None]).squeeze(1)   # logp of token t+1, length L-1
    return tok_lp[-num_actions:]


def wrapper_logprobs(gpt, full_ids, num_actions, left_pad, right_pad, scoring_mode):
    """Pipeline-style: build a left+right padded [1,S] microbatch and run the EXACT wrapper path.

    Layout mirrors SkyRL RL rollouts: [left_pad][prompt+response][right_pad]. The pipeline takes
    action_log_probs = from_parallel(...)[:, -num_actions_window:] with num_actions_window =
    max_response_len (= resp_len + right_pad here); the real response tokens sit at the FRONT of
    that window (loss_mask selects them). We return exactly those `num_actions` real-response
    logprobs so the comparison to the direct forward is mask-aware (apples-to-apples)."""
    from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
        remove_left_padding, recover_left_padding)
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        from_parallel_logits_to_logprobs)
    import megatron.core.parallel_state as mpu

    L = len(full_ids)
    S = left_pad + L + right_pad
    pad_id = 0
    seq = torch.full((1, S), pad_id, dtype=torch.long, device="cuda")
    seq[0, left_pad:left_pad + L] = torch.tensor(full_ids, device="cuda")
    attn = torch.zeros((1, S), dtype=torch.long, device="cuda")
    attn[0, left_pad:left_pad + L] = 1
    # position_ids as SkyRL builds them for padded batches: cumsum(mask)-1, clamped at 0.
    pos = (attn.cumsum(dim=1) - 1).clamp(min=0)

    attn_bool = attn.to(bool)
    new_seq, new_attn, new_pos = remove_left_padding(seq, attn_bool, pos, pre_process=True)
    # zero-KL passes attention_mask=None (pure causal flash)
    with torch.no_grad(), scoring_mode():
        out = gpt(input_ids=new_seq, position_ids=new_pos, attention_mask=None)
    out = recover_left_padding(out, new_attn, attn_bool, S, post_process=True)
    tp_grp = mpu.get_tensor_model_parallel_group()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    token_logprobs = from_parallel_logits_to_logprobs(
        out, seq,
        vocab_start_index=tp_rank * out.shape[-1],
        vocab_end_index=(tp_rank + 1) * out.shape[-1],
        tp_group=tp_grp, inference_only=True, cp_group=None, chunk_size=None,
    )  # [1, S-1]
    window = num_actions + right_pad           # the padded response window (= max_response_len)
    w = token_logprobs[0, -window:]            # [response][right_pad]
    return w[:num_actions]                     # real response logprobs (front of window)


def main():
    args = parse_args()
    sys.path.insert(0, "/home/ray/default/SkyRL-ZeroKL")
    from transformers import AutoTokenizer
    from skyrl.backends.skyrl_train.zerokl import apply_megatron_zerokl_patches, scoring_mode

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab = len(tok)
    print("[WP] building trainer-layout GPTModel (MTP-off) ...", flush=True)
    gpt = build_trainer_gptmodel(args.model)
    apply_megatron_zerokl_patches()

    texts = [
        "Compute the derivative of f(x) = x^3 - 2x + 1 and evaluate it at x = 2, showing every "
        "single step in full detail so that a beginning calculus student who has never seen the "
        "power rule before can follow the reasoning from start to finish without confusion.",
        "Explain in clear and simple terms why the sky appears blue during the middle of the day "
        "but turns shades of red and orange at sunset, covering Rayleigh scattering, the path "
        "length of sunlight through the atmosphere, and how our eyes perceive the resulting colors.",
        "A train leaves the station at three in the afternoon traveling at sixty miles per hour. "
        "Another train leaves the same station one hour later traveling at eighty miles per hour "
        "in the same direction. Determine exactly when and where the second train catches the first.",
    ]
    print(f"[WP] vocab(len tok)={vocab}; left_pad={args.left_pad} right_pad={args.right_pad}\n", flush=True)

    all_dmax = 0.0
    all_dmean = []
    for ti, text in enumerate(texts):
        ids = tok(text, add_special_tokens=False).input_ids
        full = ids[: args.prompt_len + args.resp_len]
        if len(full) < args.prompt_len + args.resp_len:
            print(f"[WP] seq {ti}: SKIP (only {len(ids)} toks < {args.prompt_len + args.resp_len})")
            continue
        num_actions = args.resp_len
        d = direct_logprobs(gpt, full, num_actions, vocab, scoring_mode)
        w = wrapper_logprobs(gpt, full, num_actions, args.left_pad, args.right_pad, scoring_mode)
        diff = (d - w).abs()
        all_dmax = max(all_dmax, diff.max().item())
        all_dmean.append(diff.mean().item())
        worst = int(diff.argmax())
        print(f"[WP] seq {ti}: num_actions={num_actions} "
              f"|direct-wrapper| mean={diff.mean().item():.3e} max={diff.max().item():.3e}")
        print(f"       worst tok @action#{worst}: direct={d[worst].item():.5f} "
              f"wrapper={w[worst].item():.5f} (token_id={full[args.prompt_len + worst] if args.prompt_len+worst < len(full) else '?'})")
        # show first few side by side
        k = min(6, num_actions)
        print(f"       first {k}: direct ={[round(x,4) for x in d[:k].tolist()]}")
        print(f"       first {k}: wrapper={[round(x,4) for x in w[:k].tolist()]}")

    import numpy as np
    print(f"\n[WP] OVERALL |direct - wrapper|: mean={np.mean(all_dmean):.3e} max={all_dmax:.3e}")
    if all_dmax == 0.0:
        print("[WP] VERDICT: wrapper path is BITWISE-identical to direct forward -> the SkyRL "
              "forward wrapper is NOT the cause. The 0.0104 is a MULTI-PROCESS effect (weight "
              "sync / cumem sleep-wake / engine). Chase it in the live pipeline.")
    else:
        print("[WP] VERDICT: wrapper path DIFFERS from direct forward -> the SkyRL forward wrapper "
              "introduces the drift. Localize via the per-token dump above.")


if __name__ == "__main__":
    main()
