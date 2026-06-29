"""Self-test: verify the zero-KL Megatron patches produce BITWISE-identical results to
the vLLM rollout kernels, by exercising the REAL patched Megatron functions (not a
re-implementation). Run inside the SkyRL-ZeroKL venv:

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m skyrl.backends.skyrl_train.zerokl._selftest
"""
import torch

torch.manual_seed(0)
dev = "cuda"


def _ndiff(a, b):
    return int(((a.view(torch.int16).int() - b.view(torch.int16).int()).abs() > 0).sum())


def main():
    import vllm._custom_ops as vops
    from skyrl.backends.skyrl_train.zerokl import (
        apply_megatron_zerokl_patches,
        zerokl_patch_status,
    )

    # Apply ALL megatron-side zero-KL patches (enables BIK, routes norm->vops, fp32 rope)
    apply_megatron_zerokl_patches()
    print("patch status:", zerokl_patch_status())

    from megatron.core.models.common.embeddings import rope_utils
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

    ok = True

    # ---- GEMM (already identical; reconfirm through Megatron's BIK matmul) ----
    a = torch.randn(2048, 2560, device=dev, dtype=torch.bfloat16) * 0.1
    w = torch.randn(2560, 2560, device=dev, dtype=torch.bfloat16) * 0.1
    g_meg = bik.matmul_persistent(a, w)
    from vllm.model_executor.layers.batch_invariant import matmul_persistent as v_mm
    g_v = v_mm(a, w)
    e = torch.equal(g_meg, g_v); ok &= e
    print(f"[GEMM]    Megatron BIK == vLLM BIK : bitwise={e}")

    # ---- RoPE: patched Megatron _apply_rotary_pos_emb_bshd == vLLM CUDA ----
    S, nH, D, theta = 1024, 8, 128, 1_000_000.0
    pos = torch.arange(S, device=dev)
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=dev, dtype=torch.float32) / D))
    freqs = torch.outer(pos.float(), inv)
    emb = torch.cat((freqs, freqs), -1)[:, None, None, :]
    q = torch.randn(S, nH, D, device=dev, dtype=torch.float32)
    cache = torch.cat((freqs.cos(), freqs.sin()), -1).to(torch.bfloat16)
    qv = q.to(torch.bfloat16).reshape(S, nH * D).clone(); kv = qv.clone()
    vops.rotary_embedding(pos, qv, kv, D, cache, True)
    tgt_rope = qv.reshape(S, nH, D)
    t = q.to(torch.bfloat16)[:, None, :, :].reshape(S, 1, nH, D)
    out_meg = rope_utils._apply_rotary_pos_emb_bshd(t, emb, rotary_interleaved=False).reshape(S, nH, D)
    e = torch.equal(out_meg, tgt_rope); ok &= e
    print(f"[RoPE]    patched Megatron == vLLM CUDA : bitwise={e}  ndiff={_ndiff(out_meg, tgt_rope)}/{out_meg.numel()}")

    # ---- RMSNorm main (over hidden): patched norm == vLLM fused_add_rms_norm, many draws ----
    H, eps = 2560, 1e-6
    tot = 0
    for s in range(5):
        torch.manual_seed(100 + s)
        x = torch.randn(4096, H, device=dev, dtype=torch.bfloat16)
        resid = torch.randn(4096, H, device=dev, dtype=torch.bfloat16)
        wn = torch.randn(H, device=dev, dtype=torch.bfloat16)
        xv = x.clone(); rv = resid.clone(); vops.fused_add_rms_norm(xv, rv, wn, eps)
        radd = x + resid  # separate bf16 add (bitwise == rv)
        out_norm = bik.BatchInvariantRMSNormFn.apply(radd, wn, eps, False)
        tot += _ndiff(out_norm, xv)
    e = (tot == 0); ok &= e
    print(f"[RMSNorm] main(hidden) patched == vLLM fused over 5 draws : bitwise={e}  total_ndiff={tot}")

    # ---- RMSNorm q/k head (over head_dim): patched norm == vLLM Triton no-residual ----
    from vllm.model_executor.layers.batch_invariant import rms_norm as vllm_tri_rms
    torch.manual_seed(7)
    h = torch.randn(2048, 128, device=dev, dtype=torch.bfloat16)
    wh = torch.randn(128, device=dev, dtype=torch.bfloat16)
    oh = bik.BatchInvariantRMSNormFn.apply(h, wh, eps, False)
    ref = vllm_tri_rms(h, wh, eps=eps)
    e = torch.equal(oh, ref); ok &= e
    print(f"[RMSNorm] q/k head(128) patched == vLLM Triton : bitwise={e}  ndiff={_ndiff(oh, ref)}/{oh.numel()}")

    # ---- gradient still flows through patched norm (training safety) ----
    xg = torch.randn(64, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
    yg = bik.BatchInvariantRMSNormFn.apply(xg, wn, eps, False)
    yg.float().pow(2).mean().backward()
    grad_ok = xg.grad is not None and torch.isfinite(xg.grad.float()).all().item()
    print(f"[grad]    backward through patched norm finite={grad_ok}")
    ok &= grad_ok

    print("\n==>", "ALL BITWISE CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
