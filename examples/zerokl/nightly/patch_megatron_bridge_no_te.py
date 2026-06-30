"""Make megatron-bridge import-safe on the no-TE (zero-KL nightly) stack.

megatron-bridge eagerly imports its whole model zoo from `megatron.bridge.__init__`
(diffusion models + all conversion bridges). A few modules hard-import TransformerEngine
at module load with NO try/except, so `from megatron.bridge import AutoBridge` crashes on a
venv where TE is intentionally absent (the no-TE local-spec stack that gives bitwise zero-KL).

This script idempotently wraps those 3 unguarded imports in try/except, substituting a LOCAL
placeholder when TE is missing. Crucially it does NOT inject a global `transformer_engine`
module into sys.modules — so megatron-core's own `import transformer_engine` still fails and
its `HAVE_TE=False` graceful fallback (torch SDPA/RMSNorm/AdamW) stays engaged. The placeholder
classes are only ever used as base classes / isinstance targets on the LoRA+TE path, which is
unused in zero-KL DAPO.

Run once against the zero-KL nightly venv after installing megatron-bridge:
    /mnt/local_storage/zerokl-nightly-venv/bin/python patch_megatron_bridge_no_te.py
"""
import os
import sys

MARKER = "# [zerokl-no-te-guard]"

# Guarded replacement for `import transformer_engine.pytorch as te` (used as base class).
TE_PYTORCH_GUARD = (
    "try:\n"
    "    import transformer_engine.pytorch as te  " + MARKER + "\n"
    "except ModuleNotFoundError:  " + MARKER + "\n"
    "    import types as _zk_types  " + MARKER + "\n"
    "    _zk_ph = type('_TEUnavailable', (), {})  " + MARKER + "\n"
    "    te = _zk_types.SimpleNamespace(Linear=_zk_ph, LayerNormLinear=_zk_ph, "
    "ops=_zk_types.SimpleNamespace(Sequential=_zk_ph))  " + MARKER + "\n"
)

# Guarded replacement for `import transformer_engine_torch as tex` (used only at runtime).
TEX_GUARD = (
    "try:\n"
    "    import transformer_engine_torch as tex  " + MARKER + "\n"
    "except ModuleNotFoundError:  " + MARKER + "\n"
    "    tex = None  " + MARKER + "\n"
)

PATCHES = [
    ("peft/lora_layers.py", "import transformer_engine.pytorch as te", TE_PYTORCH_GUARD),
    ("peft/lora.py", "import transformer_engine.pytorch as te", TE_PYTORCH_GUARD),
    ("diffusion/models/wan/utils.py", "import transformer_engine_torch as tex", TEX_GUARD),
]


def find_bridge_root():
    # Must NOT `import megatron.bridge` — its __init__ is exactly what crashes on the no-TE
    # venv (that is what we are fixing). Resolve the path via the loader spec instead, which
    # does not execute the package __init__.
    import importlib.util

    spec = importlib.util.find_spec("megatron.bridge")
    if spec is None or not spec.origin:
        raise RuntimeError("megatron.bridge not found on this interpreter")
    return os.path.dirname(spec.origin)


def main():
    root = find_bridge_root()
    print(f"[patch] megatron-bridge root: {root}")
    changed = 0
    for rel, needle, replacement in PATCHES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"[patch] SKIP (absent): {rel}")
            continue
        src = open(path).read()
        if MARKER in src:
            print(f"[patch] already patched: {rel}")
            continue
        # Replace the first exact, line-anchored, unindented occurrence only.
        line = needle + "\n"
        if line not in src:
            print(f"[patch] WARN: needle not found verbatim in {rel}; leaving untouched")
            continue
        src = src.replace(line, replacement, 1)
        open(path, "w").write(src)
        print(f"[patch] guarded TE import in {rel}")
        changed += 1
    print(f"[patch] done; {changed} file(s) changed")
    # Verify the import now works.
    try:
        from megatron.bridge import AutoBridge  # noqa: F401

        print("[patch] VERIFY: `from megatron.bridge import AutoBridge` OK")
    except Exception as e:  # pragma: no cover
        print(f"[patch] VERIFY FAILED: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
