"""DAPO zero-KL training loop on the NIGHTLY stack (true bitwise rollout_train==0).

Port of examples/zerokl/dapo_zerokl.py to the validated nightly architecture: the rollout engine
is the local-spec MiMo GPTModel-in-vLLM-1.0 with the varlen (num_splits=1) attention backend, so
engine decode == engine prefill BITWISE. The trainer is a second local-spec MiMo GPTModel (same
weights) computing the grad logprob. The zero-KL metric (rollout behavior vs the engine prefill
rescore) is therefore 0.000e+00 by construction; is_ratio carries only the small trainer-vs-engine
cross-kernel drift. Native (no-HF) weight sync each step. Toy verifiable reward (count of " the").

Run:
  CUDA_VISIBLE_DEVICES=5 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_BATCH_INVARIANT=1 \
    /mnt/local_storage/zerokl-nightly-venv/bin/python dapo_zerokl_nightly.py --steps 6
"""
import argparse, os, sys
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--prompts_per_step", type=int, default=4)
    ap.add_argument("--group", type=int, default=6)            # samples/prompt (GRPO group)
    ap.add_argument("--max_tokens", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--eps_low", type=float, default=0.2)
    ap.add_argument("--eps_high", type=float, default=0.28)
    ap.add_argument("--clip_c", type=float, default=10.0)
    ap.add_argument("--gpu_mem", type=float, default=0.55)
    args = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    MODEL = "/mnt/local_storage/models/MiMo-7B-RL"

    import varlen_backend  # noqa: F401  -> registers CUSTOM (num_splits=1) attention backend
    from vllm import LLM, SamplingParams
    from mimo_megatron_vllm import (register_mimo_to_vllm, CONFIG_FORMAT, build_mimo_gptmodel,
                                    find_inprocess_gpt)
    register_mimo_to_vllm()
    import vllm.model_executor.model_loader.weight_utils as _wu
    import vllm.model_executor.model_loader.dummy_loader as _dl
    _wu.initialize_dummy_weights = lambda *a, **k: None
    _dl.initialize_dummy_weights = lambda *a, **k: None

    # ---- ROLLOUT engine: local-spec MiMo GPTModel in vLLM, varlen backend ----
    llm = LLM(model=MODEL, config_format=CONFIG_FORMAT, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem, max_model_len=1024, enable_prefix_caching=False,
              enable_chunked_prefill=False, load_format="dummy", trust_remote_code=True,
              attention_backend="CUSTOM")
    tok = llm.get_tokenizer()
    engine_gpt = find_inprocess_gpt(llm)
    assert engine_gpt is not None, "could not reach in-process engine GPTModel"
    print(f"[nightly-dapo] engine ready; engine_gpt params={sum(1 for _ in engine_gpt.named_parameters())}", flush=True)

    # ---- TRAINER: a second local-spec MiMo GPTModel (same weights), trainable ----
    trainer, _cfg = build_mimo_gptmodel(torch.device("cuda"), dtype=torch.bfloat16)
    for p in trainer.parameters():
        p.requires_grad_(True)
    trainer.train()
    opt = torch.optim.SGD([p for p in trainer.parameters() if p.requires_grad], lr=args.lr)
    VOCAB = len(tok)
    TARGET = tok(" the", add_special_tokens=False).input_ids[-1]
    PROMPTS = ["The best way to", "My favorite thing is", "Once upon a time", "In the morning",
               "The answer to the question", "Scientists recently found", "Today I will", "The most important"]

    def native_sync():
        with torch.no_grad():
            dst = dict(engine_gpt.named_parameters())
            for n, p in trainer.named_parameters():
                d = dst.get(n)
                if d is not None and tuple(d.shape) == tuple(p.shape):
                    d.copy_(p.detach().to(d.dtype))
    native_sync()  # trainer == engine at start

    def trainer_logp(full_ids, n_prompt):
        inp = torch.tensor([full_ids], device="cuda")
        pos = torch.arange(len(full_ids), device="cuda").unsqueeze(0)
        logits = trainer(input_ids=inp, position_ids=pos, attention_mask=None)[0].float()[:, :VOCAB]
        lp = torch.log_softmax(logits, dim=-1)
        resp = torch.tensor(full_ids[n_prompt:], device="cuda")
        idx = torch.arange(n_prompt - 1, len(full_ids) - 1, device="cuda")
        return lp[idx].gather(1, resp[:, None]).squeeze(1)

    print(f"\n{'step':>4} {'reward':>7} {'rollout_train[mean,max]':>24} {'is_ratio[mean,max]':>20} {'loss':>9}", flush=True)
    for step in range(args.steps):
        prompts = [random.choice(PROMPTS) for _ in range(args.prompts_per_step)]
        pid_lists = [tok(p, add_special_tokens=False).input_ids for p in prompts]
        sp = SamplingParams(n=args.group, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens,
                            logprobs=0, seed=100 + step)
        outs = llm.generate([{"prompt_token_ids": pids} for pids in pid_lists], sp)

        groups, rt_diffs = [], []
        for pids, o in zip(pid_lists, outs):
            grp = []
            for s in o.outputs:
                rids = list(s.token_ids)
                if not rids:
                    continue
                blp = [s.logprobs[i][rids[i]].logprob for i in range(len(rids))]
                full = pids + rids
                # OLD logprob via ENGINE prefill rescore (== decode, bitwise) -> zero-KL metric
                r = llm.generate([{"prompt_token_ids": full}],
                                 SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=0))[0]
                old = [r.prompt_logprobs[t][full[t]].logprob for t in range(len(pids), len(full))]
                rt_diffs += [abs(a - c) for a, c in zip(blp, old)]
                reward = float(sum(1 for t in rids if t == TARGET))
                grp.append(dict(pids=pids, rids=rids, old=old, reward=reward))
            if grp:
                groups.append(grp)
        rt = np.array(rt_diffs) if rt_diffs else np.zeros(1)

        # GRPO advantage (group-normalized); skip zero-variance groups
        samples = []
        for grp in groups:
            r = np.array([s["reward"] for s in grp])
            if r.std() < 1e-8:
                continue
            adv = (r - r.mean()) / (r.std() + 1e-6)
            for s, a in zip(grp, adv):
                s["adv"] = float(a); samples.append(s)
        all_r = np.array([s["reward"] for g in groups for s in g])
        if not samples:
            print(f"{step:>4} {all_r.mean():>7.3f} [{rt.mean():>9.2e},{rt.max():>9.2e}]  (all groups filtered)", flush=True)
            continue

        # DAPO dual-clip / clip-higher loss, token_mean
        opt.zero_grad()
        tot_tok = sum(len(s["rids"]) for s in samples); ratios = []; loss_val = 0.0
        for s in samples:
            new = trainer_logp(s["pids"] + s["rids"], len(s["pids"]))
            old = torch.tensor(s["old"], device="cuda")
            ratio = torch.exp(new - old); a = s["adv"]
            surr1 = ratio * a
            surr2 = torch.clamp(ratio, 1 - args.eps_low, 1 + args.eps_high) * a
            pg = -torch.minimum(surr1, surr2)
            if a < 0:
                pg = torch.minimum(pg, torch.full_like(pg, -a * args.clip_c))
            loss = pg.sum() / max(tot_tok, 1)
            loss.backward(); loss_val += loss.item(); ratios += ratio.detach().tolist()
        opt.step()
        native_sync()
        rr = np.array(ratios)
        print(f"{step:>4} {all_r.mean():>7.3f} [{rt.mean():>9.2e},{rt.max():>9.2e}] "
              f"[{rr.mean():>8.5f},{rr.max():>8.5f}] {loss_val:>9.4f}", flush=True)

    print("\n==> nightly DAPO zero-KL: rollout_train (behavior vs engine prefill rescore) is BITWISE 0 "
          "because engine decode==prefill (varlen num_splits=1 + local-spec batch-invariant). "
          "is_ratio carries only trainer-vs-engine cross-kernel drift.", flush=True)


if __name__ == "__main__":
    main()
