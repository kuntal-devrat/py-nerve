#!/usr/bin/env python3
"""Package the compiled MNN (built once with full SIMD) as a reusable prebuilt.

Why this exists
---------------
ocr-rs normally compiles MNN from source with cmake on every build (~15-30 min).
The vendored patch (vendor/ocr-rs) supports `OCR_RS_MNN_PREBUILT_DIR`: if it
points at a directory containing `include/` and `lib/`, the build links against
that precompiled MNN and skips cmake entirely.

This script produces that directory. Run it once per (OS, architecture) on a
machine that has the toolchain, upload `dist/mnn-prebuilt/<triple>/` as a build
artifact / GitHub release asset, and point CI builds at it:

    OCR_RS_MNN_PREBUILT_DIR=dist/mnn-prebuilt/<triple> maturin build --release

The resulting wheel still bundles MNN statically — end users never compile
anything, and neither does the wheel build.

Usage:
    python scripts/build_mnn_prebuilt.py          # build (if needed) + package
    python scripts/build_mnn_prebuilt.py --only-package   # reuse existing target/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def host_triple() -> str:
    """Best-effort rustc host triple (e.g. x86_64-pc-windows-msvc)."""
    try:
        out = subprocess.run(
            ["rustc", "-vV"], capture_output=True, text=True, check=True
        ).stdout
        for line in out.splitlines():
            if line.startswith("host: "):
                return line.split("host: ", 1)[1].strip()
    except Exception:
        pass
    # Fallback mapping
    import platform

    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(
        machine, machine
    )
    if sys.platform == "win32":
        return f"{arch}-pc-windows-msvc"
    if sys.platform == "darwin":
        return f"{arch}-apple-darwin"
    return f"{arch}-unknown-linux-gnu"


def find_ocr_build_dir() -> Path | None:
    """Locate the ocr-rs cmake output that contains a compiled MNN."""
    build_root = ROOT / "target" / "release" / "build"
    if not build_root.is_dir():
        return None
    candidates = []
    for entry in build_root.glob("ocr-rs-*/out"):
        if (entry / "include" / "MNN").is_dir() and (
            (entry / "lib" / "MNN.lib").exists() or (entry / "lib" / "libMNN.a").exists()
        ):
            candidates.append(entry)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-package", action="store_true", help="Skip the build step")
    args = parser.parse_args()

    triple = host_triple()
    out_dir = ROOT / "dist" / "mnn-prebuilt" / triple

    if not args.only_package:
        print(f"[1/3] Building ocr-rs (MNN from source, SIMD-enabled) for {triple} ...")
        subprocess.run(
            ["cargo", "build", "--release"], cwd=ROOT, check=True
        )

    print("[2/3] Locating compiled MNN ...")
    ocr_out = find_ocr_build_dir()
    if ocr_out is None:
        sys.exit(
            "Could not find a compiled MNN under target/release/build/ocr-rs-*/out. "
            "Run without --only-package so the build runs first."
        )
    print(f"      found: {ocr_out}")

    print(f"[3/3] Packaging prebuilt to {out_dir} ...")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(ocr_out / "include", out_dir / "include")
    shutil.copytree(ocr_out / "lib", out_dir / "lib")
    size = sum(f.stat().st_size for f in (out_dir / "lib").rglob("*") if f.is_file())
    print(f"      done ({size / 1e6:.1f} MB).")
    print()
    print("Build wheels / the extension against it with:")
    print(f"  OCR_RS_MNN_PREBUILT_DIR={out_dir} maturin build --release")
    print()
    print("Upload this directory (e.g. as a GitHub release asset named")
    print(f"  mnn-prebuilt-{triple}.tar.gz) and let CI download it before building.")


if __name__ == "__main__":
    main()
