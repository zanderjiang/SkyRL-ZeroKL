"""In-process no-TransformerEngine guard for the zero-KL nightly stack.

megatron-bridge eagerly imports its model zoo from `megatron.bridge.__init__`, and three modules
(peft/lora_layers, peft/lora, diffusion/wan/utils) hard-import TransformerEngine at module load with
NO try/except. On the zero-KL nightly stack TE is intentionally absent, so `from megatron.bridge
import AutoBridge` crashes. The Anyscale Ray actor hook builds a FRESH uv env per run, so a one-off
site-package edit doesn't survive. This guard lives in the (path-installed) skyrl package, so it runs
in EVERY env -- driver and actor.

IMPORTANT: it does NOT register a stub `transformer_engine` in sys.modules. A stub would make
`import transformer_engine` SUCCEED, flipping megatron-core/-bridge HAVE_TE checks to True and
triggering deeper TE imports (e.g. transformer_engine.pytorch.float8_tensor) that then fail. Instead
it GENUINELY keeps TE absent (so HAVE_TE stays False and the local/torch fallbacks engage) and only
neutralises the three unguarded module-level imports by editing those megatron-bridge files in place
-- the same proven edit as examples/zerokl/nightly/patch_megatron_bridge_no_te.py, applied atomically
and idempotently. The driver imports skyrl (and thus runs this) before spawning actors, so the shared
env is patched before any actor imports megatron.bridge; the MARKER check makes re-entry a no-op.

Call install_no_te_guard() BEFORE the first `import megatron.bridge`. No-op unless
SKYRL_ZEROKL_LOCAL_SPEC=1, and a no-op if real TE is importable.
"""
import importlib.util
import os
import sys

_INSTALLED = False
MARKER = "# [zerokl-no-te-guard]"

_TE_PYTORCH_GUARD = (
    "try:\n"
    "    import transformer_engine.pytorch as te  " + MARKER + "\n"
    "except ModuleNotFoundError:  " + MARKER + "\n"
    "    import types as _zk_types  " + MARKER + "\n"
    "    _zk_ph = type('_TEUnavailable', (), {})  " + MARKER + "\n"
    "    te = _zk_types.SimpleNamespace(Linear=_zk_ph, LayerNormLinear=_zk_ph, "
    "ops=_zk_types.SimpleNamespace(Sequential=_zk_ph))  " + MARKER + "\n"
)
_TEX_GUARD = (
    "try:\n"
    "    import transformer_engine_torch as tex  " + MARKER + "\n"
    "except ModuleNotFoundError:  " + MARKER + "\n"
    "    tex = None  " + MARKER + "\n"
)
_PATCHES = (
    ("peft/lora_layers.py", "import transformer_engine.pytorch as te", _TE_PYTORCH_GUARD),
    ("peft/lora.py", "import transformer_engine.pytorch as te", _TE_PYTORCH_GUARD),
    ("diffusion/models/wan/utils.py", "import transformer_engine_torch as tex", _TEX_GUARD),
)


def _bridge_root():
    # Resolve the path WITHOUT importing megatron.bridge (its __init__ is what crashes).
    spec = importlib.util.find_spec("megatron.bridge")
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(spec.origin)


def install_no_te_guard() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get("SKYRL_ZEROKL_LOCAL_SPEC") != "1":
        return False
    try:
        import transformer_engine  # noqa: F401  -- real TE present, nothing to do
        return False
    except ImportError:
        pass

    root = _bridge_root()
    if root is None:
        return False
    changed = 0
    for rel, needle, replacement in _PATCHES:
        path = os.path.join(root, rel)
        try:
            src = open(path).read()
        except OSError:
            continue
        if MARKER in src:
            continue
        line = needle + "\n"
        if line not in src:
            continue
        new_src = src.replace(line, replacement, 1)
        # Atomic write so a concurrent importer never sees a torn file.
        tmp = f"{path}.zk.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            fh.write(new_src)
        os.replace(tmp, path)
        changed += 1
    _INSTALLED = True
    if changed:
        print(f"[zerokl] no-TE guard: patched {changed} megatron-bridge module(s) for genuine TE absence",
              flush=True)
    return True
