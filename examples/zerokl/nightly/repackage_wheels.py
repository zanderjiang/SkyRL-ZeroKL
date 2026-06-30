"""Reconstruct .whl files for packages that are installed in the proven zero-KL venv but have been
garbage-collected from public nightly indexes (torch 2.14.0.dev20260620+cu130, vllm
1.0.0.dev20260620+cu130, triton 3.7.1+git...). The Anyscale Ray uv hook re-resolves actor envs from
pyproject, so these must be resolvable; hosting reconstructed wheels in a local find-links dir makes
`uv --extra zerokl` resolve to the EXACT validated binaries with no public-index dependency.

A wheel is a zip whose members are exactly the files listed in {dist}.dist-info/RECORD, at paths
relative to site-packages, named {name}-{version}-{tag}.whl (tag from dist-info/WHEEL). We rebuild
from RECORD (the canonical manifest) so the result is byte-faithful to the original install.

Run on the proven venv:
    /mnt/local_storage/zerokl-nightly-venv/bin/python repackage_wheels.py
Output: /mnt/local_storage/zerokl-wheels/*.whl
"""
import csv
import importlib.metadata as md
import os
import sys
import zipfile

OUT_DIR = os.environ.get("ZEROKL_WHEELS_DIR", "/mnt/local_storage/zerokl-wheels")
PKGS = ["torch", "vllm", "triton"]


def wheel_filename(dist):
    name = dist.metadata["Name"].replace("-", "_")
    version = dist.version
    # Tag(s) from the WHEEL metadata; a wheel may declare multiple compressed tags (tag1.tag2.tag3).
    tag = None
    for f in dist.files or []:
        if str(f).endswith(".dist-info/WHEEL"):
            for line in open(dist.locate_file(f), "r", errors="replace").read().splitlines():
                if line.startswith("Tag:"):
                    tag = line.split("Tag:", 1)[1].strip()
                    break
    if tag is None:
        raise RuntimeError(f"no Tag in WHEEL for {name}")
    return f"{name}-{version}-{tag}.whl"


def record_paths(dist):
    """Canonical file list from dist-info/RECORD (relative to site-packages)."""
    record_rel = None
    for f in dist.files or []:
        if str(f).endswith(".dist-info/RECORD"):
            record_rel = f
            break
    if record_rel is None:
        raise RuntimeError(f"no RECORD for {dist.metadata['Name']}")
    rows = []
    with open(dist.locate_file(record_rel), newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            rows.append(row[0])  # first column = path relative to install root
    return rows


def build(pkg):
    dist = md.distribution(pkg)
    site = dist.locate_file("")  # site-packages root
    site = os.path.abspath(str(site))
    whl = os.path.join(OUT_DIR, wheel_filename(dist))
    paths = record_paths(dist)
    n_ok, n_miss = 0, 0
    missing = []
    with zipfile.ZipFile(whl, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for rel in paths:
            src = os.path.join(site, rel)
            if not os.path.isfile(src):
                n_miss += 1
                if len(missing) < 5:
                    missing.append(rel)
                continue
            z.write(src, arcname=rel)
            n_ok += 1
    size = os.path.getsize(whl)
    print(f"[repackage] {pkg}=={dist.version}: wrote {os.path.basename(whl)} "
          f"({size/1e6:.0f}MB, {n_ok} files, {n_miss} missing){' e.g. '+str(missing) if missing else ''}",
          flush=True)
    return whl


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    built = []
    for p in PKGS:
        try:
            built.append(build(p))
        except Exception as e:
            print(f"[repackage] {p}: FAILED {type(e).__name__}: {e}", flush=True)
            sys.exit(1)
    print(f"[repackage] done -> {OUT_DIR}", flush=True)
    for b in built:
        print("  ", os.path.basename(b))


if __name__ == "__main__":
    main()
