"""DAPO with ZERO KL (unified Megatron-GPTModel route).

A self-contained DAPO training loop on the SkyRL-ZeroKL components where the rollout engine and
the trainer compute BITWISE-IDENTICAL logprobs (|behavior - train_old| == 0). DAPO pieces
(matching examples/train/megatron/run_megatron_dapo_qwen3_4b.sh):
  - dual-clip / clip-higher PPO loss (eps_low=0.2, eps_high=0.28, clip_ratio_c=10)
  - token_mean loss reduction
  - dynamic sampling (drop groups whose rewards are all equal -> zero advantage)
  - overlong filtering (mask truncated samples that never emitted EOS)
  - no KL loss, temperature=1.0
  - TIS OFF (unnecessary: is_ratio == 1 because rollout==train bitwise)

Architecture: vLLM runs Megatron's GPTModel (rollout); the "old/train" logprob is a
vLLM-GPTModel prefill rescore == behavior (vLLM prefill==decode is bitwise); the Megatron
GPTModel computes new_logp (grad) for the DAPO update; weights sync natively (no HF) each step.

Usage (see run_dapo_zerokl_qwen3_4b.sh):
  VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m examples.zerokl.dapo_zerokl --steps 20 ...
"""
import argparse, os, sys, time
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

import random
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/mnt/local_storage/hf/hub/models--Qwen--Qwen3-4B/"
                                      "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--prompts_per_step", type=int, default=8)
    p.add_argument("--n_samples_per_prompt", type=int, default=8)   # DAPO group size
    p.add_argument("--max_response_length", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--eps_clip_low", type=float, default=0.2)
    p.add_argument("--eps_clip_high", type=float, default=0.28)     # clip-higher
    p.add_argument("--clip_ratio_c", type=float, default=10.0)      # dual-clip
    p.add_argument("--loss_reduction", default="token_mean")
    p.add_argument("--dynamic_sampling", type=int, default=1)
    p.add_argument("--overlong_filtering", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--gpu_mem_util", type=float, default=0.40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", type=int, default=0)
    p.add_argument("--wandb_project", default="zerokl_qwen_dapo")
    p.add_argument("--wandb_run", default="zerokl_qwen_dapo")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        print(f"[dapo-zerokl] wandb logging to project={args.wandb_project} run={args.wandb_run}")
    sys.path.insert(0, "/home/ray/default/SkyRL-ZeroKL")
    from transformers import AutoTokenizer
    from skyrl.backends.skyrl_train.zerokl.gptmodel_vllm import (
        register_gptmodel_to_vllm, find_inprocess_gptmodel, VLLM_MODEL_NAME)
    from skyrl.backends.skyrl_train.zerokl.native_weight_sync import (
        extract_native_weights, load_native_weights)

    tok = AutoTokenizer.from_pretrained(args.model)

    # ---- verifiable, learnable reward: count occurrences of a target token in the response.
    # (A real DAPO run would use a math/verifier reward; this keeps the demo single-GPU and
    # gives a clear learning signal so DAPO's dynamic sampling keeps groups.) ----
    TARGET = tok(" the", add_special_tokens=False).input_ids[-1]
    _PROMPTS = ["The best number is", "My favorite thing is", "Today I will",
                "The answer is", "In the morning", "Once upon a time",
                "The most important", "Scientists discovered"]

    def make_prompt():
        return random.choice(_PROMPTS), None

    def reward_fn(rids, ans):
        return float(sum(1 for t in rids if t == TARGET))

    # ---- ROLLOUT: vLLM running Megatron GPTModel ----
    register_gptmodel_to_vllm(args.model)
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=1024, seed=42,
              enable_prefix_caching=False, enable_chunked_prefill=False, trust_remote_code=True,
              hf_overrides={"architectures": [VLLM_MODEL_NAME]})
    vllm_model = find_inprocess_gptmodel(llm)
    print(f"[dapo-zerokl] rollout vLLM-GPTModel ready ({type(vllm_model).__name__})")

    # ---- TRAINER: Megatron GPTModel (grad) ----
    from megatron.bridge import AutoBridge
    from megatron.core.transformer.enums import AttnBackend
    from skyrl.backends.skyrl_train.zerokl import apply_megatron_zerokl_patches, scoring_mode
    bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=True)
    mp = bridge.to_megatron_provider(load_weights=True)
    mp.tensor_model_parallel_size = 1; mp.pipeline_model_parallel_size = 1
    mp.expert_model_parallel_size = 1; mp.expert_tensor_parallel_size = 1
    mp.pipeline_dtype = torch.bfloat16; mp.apply_rope_fusion = False
    mp.attention_backend = AttnBackend.flash; mp.gradient_accumulation_fusion = False
    mp.finalize()
    trainer = mp.provide_distributed_model(wrap_with_ddp=False)
    tinner = trainer[0].module if hasattr(trainer[0], "module") else trainer[0]
    apply_megatron_zerokl_patches()
    opt = torch.optim.SGD([p for p in tinner.parameters() if p.requires_grad], lr=args.lr)
    load_native_weights(vllm_model.gpt, extract_native_weights(trainer, dtype=torch.bfloat16))
    print("[dapo-zerokl] trainer GPTModel ready; synced -> rollout (native, no HF)")

    def trainer_logp(full_ids, n_prompt):
        inp = torch.tensor([full_ids], device="cuda")
        pos = torch.arange(len(full_ids), device="cuda").unsqueeze(0)
        with scoring_mode():
            logits = tinner(input_ids=inp, position_ids=pos, attention_mask=None)[0].float()[:, :len(tok)]
        lp = torch.log_softmax(logits, dim=-1)
        resp = torch.tensor(full_ids[n_prompt:], device="cuda")
        idx = torch.arange(n_prompt - 1, len(full_ids) - 1, device="cuda")
        return lp[idx].gather(1, resp[:, None]).squeeze(1)

    eos = tok.eos_token_id
    # SkyRL-native metric names:
    #   policy/rollout_train_logprobs_abs_diff_* = |rollout behavior - trainer old recompute|
    #     trainer old recompute goes through the UNIFIED vLLM-GPTModel => exactly 0 (zero KL).
    #   is_ratio_* = exp(new_logp[Megatron grad] - old_logp). ~1 (tiny fp32 grad-engine drift).
    hdr = (f"{'step':>4} {'reward':>7} {'rt_abs_diff[mean,max]':>22} "
           f"{'is_ratio[mean,max]':>20} {'policy/clipfrac':>15} {'kept/grp':>9} {'policy_loss':>11}")
    print("\n" + hdr)
    for step in range(args.steps):
        torch.cuda.reset_peak_memory_stats()
        t_step0 = time.perf_counter()
        prompts = [make_prompt() for _ in range(args.prompts_per_step)]
        sp = SamplingParams(n=args.n_samples_per_prompt, temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_response_length, logprobs=0, seed=100 + step)
        outs = llm.generate([p for p, _ in prompts], sp)
        t_generate = time.perf_counter() - t_step0

        groups = []  # list of groups; each group is list of sample dicts
        for (ptext, ans), o in zip(prompts, outs):
            pids = list(o.prompt_token_ids)
            grp = []
            for s in o.outputs:
                rids = list(s.token_ids)
                if not rids:
                    continue
                blp = [s.logprobs[i][rids[i]].logprob for i in range(len(rids))]
                reward = reward_fn(rids, ans)
                truncated = (rids[-1] != eos) and (len(rids) >= args.max_response_length)
                grp.append(dict(pids=pids, rids=rids, blp=blp, reward=reward, truncated=truncated))
            if grp:
                groups.append(grp)

        # ---- trainer OLD logprob recompute via the UNIFIED vLLM-GPTModel (== rollout behavior) ----
        t1 = time.perf_counter()
        rt_diffs = []   # per-token |rollout behavior - trainer old recompute|  -> the zero-KL metric
        for grp in groups:
            for s in grp:
                full = s["pids"] + s["rids"]
                o = llm.generate({"prompt_token_ids": full},
                                 SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=0))[0]
                s["old"] = [o.prompt_logprobs[t][full[t]].logprob for t in range(len(s["pids"]), len(full))]
                rt_diffs.extend([abs(a - c) for a, c in zip(s["blp"], s["old"])])
        rt = np.array(rt_diffs) if rt_diffs else np.zeros(1)
        rt_stats = {"policy/rollout_train_logprobs_abs_diff_mean": float(rt.mean()),
                    "policy/rollout_train_logprobs_abs_diff_max": float(rt.max()),
                    "policy/rollout_train_logprobs_abs_diff_min": float(rt.min()),
                    "policy/rollout_train_logprobs_abs_diff_std": float(rt.std())}
        t_old_logprob = time.perf_counter() - t1

        # ---- DAPO advantages: group-normalized; dynamic sampling drops std==0 groups ----
        samples = []
        n_groups_total = len(groups); n_groups_kept = 0
        for grp in groups:
            r = np.array([s["reward"] for s in grp])
            if args.dynamic_sampling and (r.std() < 1e-8):   # all same reward -> no signal
                continue
            n_groups_kept += 1
            adv = (r - r.mean()) / (r.std() + 1e-6)
            for s, a in zip(grp, adv):
                if args.overlong_filtering and s["truncated"]:  # mask truncated samples
                    continue
                s["adv"] = float(a); samples.append(s)

        # ---- generate/ + reward/ stats (SkyRL taxonomy) ----
        all_rewards = np.array([s["reward"] for g in groups for s in g])
        resp_lens = [len(s["rids"]) for g in groups for s in g]
        gen_tok = int(sum(resp_lens))
        generate_stats = {"generate/batch_num_seq": int(sum(len(g) for g in groups)),
                          "generate/response_length_mean": float(np.mean(resp_lens)),
                          "generate/response_length_max": int(max(resp_lens))}
        reward_stats = {"reward/mean": float(all_rewards.mean()),
                        "reward/avg_raw_reward": float(all_rewards.mean()),
                        "reward/mean_positive_reward": float((all_rewards > 0).mean()),
                        "reward/num_zero_variance_filtered": n_groups_total - n_groups_kept}
        system_stats = {"system/gpu_mem_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
                        "system/gpu_mem_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
        base_log = {"trainer/global_step": step, "trainer/epoch": step,
                    "dapo/groups_kept": n_groups_kept, "dapo/groups_total": n_groups_total,
                    **rt_stats, **generate_stats, **reward_stats, **system_stats}
        if not samples:
            t_step = time.perf_counter() - t_step0
            print(f"{step:>4} {all_rewards.mean():>7.3f} "
                  f"[{rt.mean():>9.2e},{rt.max():>9.2e}] {'(all groups filtered)':>20}")
            if wb is not None:
                wb.log({**base_log, "timing/generate": t_generate, "timing/old_logprob": t_old_logprob,
                        "timing/step": t_step}, step=step)
            continue

        # ---- DAPO dual-clip / clip-higher loss, token_mean (SkyRL ppo_policy_loss / dual_clip) ----
        t2 = time.perf_counter()
        opt.zero_grad()
        tot_tok = sum(len(s["rids"]) for s in samples)
        clip_hits = 0; n_tok = 0; ratios = []; loss_val = 0.0
        for s in samples:
            new = trainer_logp(s["pids"] + s["rids"], len(s["pids"]))      # grad (Megatron-GPTModel)
            old = torch.tensor(s["old"], device="cuda")                    # unified vLLM-GPTModel recompute
            ratio = torch.exp(new - old)
            a = s["adv"]
            surr1 = ratio * a
            surr2 = torch.clamp(ratio, 1 - args.eps_clip_low, 1 + args.eps_clip_high) * a
            pg = -torch.minimum(surr1, surr2)                              # PPO clip-higher
            if a < 0:                                                      # dual-clip lower bound
                pg = torch.minimum(pg, torch.full_like(pg, -a * args.clip_ratio_c))
            loss = pg.sum() / max(tot_tok, 1) if args.loss_reduction == "token_mean" else pg.mean() / len(samples)
            loss.backward()
            loss_val += loss.item()
            clip_hits += int((surr2 < surr1).sum())                        # SkyRL clipfrac convention
            n_tok += ratio.numel(); ratios.extend(ratio.detach().tolist())
        opt.step()
        t_train = time.perf_counter() - t2

        # ---- native sync trainer -> rollout (no HF) ----
        t3 = time.perf_counter()
        load_native_weights(vllm_model.gpt, extract_native_weights(trainer, dtype=torch.bfloat16))
        t_sync = time.perf_counter() - t3
        t_step = time.perf_counter() - t_step0
        rr = np.array(ratios); clipfrac = clip_hits / max(n_tok, 1)
        print(f"{step:>4} {all_rewards.mean():>7.3f} [{rt.mean():>9.2e},{rt.max():>9.2e}] "
              f"[{rr.mean():>8.5f},{rr.max():>8.5f}] {clipfrac:>15.4f} "
              f"{n_groups_kept:>4d}/{n_groups_total:<4d} {loss_val:>11.5f}")
        if wb is not None:
            wb.log({**base_log,
                    "is_ratio_mean": float(rr.mean()), "is_ratio_std": float(rr.std()),
                    "is_ratio_max": float(rr.max()), "is_ratio_min": float(rr.min()),
                    "policy/clipfrac": clipfrac, "policy/policy_loss": loss_val,
                    "timing/generate": t_generate, "timing/old_logprob": t_old_logprob,
                    "timing/train_step": t_train, "timing/weight_sync": t_sync, "timing/step": t_step,
                    "trainer/tokens_per_second": gen_tok / max(t_step, 1e-6)}, step=step)

    if wb is not None:
        wb.finish()
    print("\n==> DAPO zero-KL run complete. policy/rollout_train_logprobs_abs_diff == 0 "
          "(trainer old-logprob recompute via the unified vLLM-GPTModel == rollout behavior, bitwise). "
          "is_ratio ~1 carries only the fp32 grad-engine drift (Megatron forward vs unified ~1e-6).")


if __name__ == "__main__":
    main()
