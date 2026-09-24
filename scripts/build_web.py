"""Assemble the static in-browser demo in web/ (and optionally deploy it).

    uv run python scripts/build_web.py                  # vendor libraries into web/vendor
    uv run python -m http.server -d web 8000            # then open http://localhost:8000
    uv run --extra app python scripts/build_web.py --deploy   # upload to a free static Space

Prerequisites, in order:
    uv run --with onnx --with onnxruntime --with onnxscript python scripts/export_onnx.py
    uv run python scripts/build_web_pool.py
    cd web && npm install && node test/golden.mjs      # must pass before deploying

The browser loads onnxruntime-web and RDKit.js from web/vendor, copied from the
exact npm packages the golden test ran against -- not from a CDN -- so visitors
run the versions that were verified.

Deploys as a PRIVATE static Space by default: COCONUT is open data, but the
model was trained on competition data, so keep it private until the
competition ends, then run with --public.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
NM = WEB / "node_modules"
VENDOR = {
    NM / "onnxruntime-web/dist/ort.wasm.min.js": "ort.wasm.min.js",
    NM / "onnxruntime-web/dist/ort-wasm-simd-threaded.mjs": "ort-wasm-simd-threaded.mjs",
    NM / "onnxruntime-web/dist/ort-wasm-simd-threaded.wasm": "ort-wasm-simd-threaded.wasm",
    NM / "@rdkit/rdkit/dist/RDKit_minimal.js": "RDKit_minimal.js",
    NM / "@rdkit/rdkit/dist/RDKit_minimal.wasm": "RDKit_minimal.wasm",
}
# What the Space serves -- nothing else from web/ (no node_modules, no tests).
SITE = ["index.html", "core.js", "meta.json", "model.onnx", "pool", "vendor"]
SPACE_NAME = "casmi26-spectrum-to-structure"

README = """\
---
title: Spectrum to Structure
emoji: 🧪
colorFrom: indigo
colorTo: green
sdk: static
pinned: false
---
"""


def vendor() -> None:
    out = WEB / "vendor"
    out.mkdir(exist_ok=True)
    for src, name in VENDOR.items():
        if not src.exists():
            raise SystemExit(f"missing {src} -- run `npm install` in web/")
        shutil.copy2(src, out / name)
    for f in SITE:
        if not (WEB / f).exists():
            raise SystemExit(f"missing web/{f} -- see the prerequisites in this script")
    size = sum(p.stat().st_size for f in SITE for p in
               ([WEB / f] if (WEB / f).is_file() else (WEB / f).rglob("*")) if p.is_file())
    print(f"site assembled in web/ ({size / 1e6:.0f} MB)")


def deploy(public: bool) -> None:
    from huggingface_hub import HfApi
    token = next(l.split("=", 1)[1].strip() for l in (ROOT / ".env").read_text().splitlines()
                 if l.startswith("HF_TOKEN=") and l.split("=", 1)[1].strip())
    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/{SPACE_NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="static",
                    private=not public, exist_ok=True)
    api.upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space")
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=WEB,
                      allow_patterns=[f"{f}" if (WEB / f).is_file() else f"{f}/**" for f in SITE],
                      commit_message="Deploy in-browser demo")
    print(f"deployed ({'public' if public else 'private'}): https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    vendor()
    if args.deploy:
        deploy(args.public)
