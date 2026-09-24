"""Deploy the Gradio demo to a Hugging Face Space.

    uv run --extra app python scripts/deploy_space.py            # private (default)
    uv run --extra app python scripts/deploy_space.py --public   # after the competition

Assembles exactly what the Space needs from the repo -- the app, the four
casmi modules it imports, the model, the candidate pool, requirements and the
Space config -- into a temp folder, then uploads it. Nothing is copied by hand,
so the Space cannot drift from the repo.

PRIVATE BY DEFAULT: the candidate pool contains ~275k structures taken from the
competition's train.parquet, and competition rules generally forbid
redistributing competition data outside Kaggle while it runs.

The token is read from .env (HF_TOKEN) and never printed.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_NAME = "casmi26-molecule-id"
MODEL = ROOT / "data" / "fpnet" / "fpnet_v4c.pt"
POOL = ROOT / "data" / "pool.parquet"
MODULES = ["fpnet.py", "pool.py", "analog.py"]   # what app.py imports, plus analog
                                                 # (fpnet.candidate_bits uses it)

# Pinned to what was tested locally. CPU-only torch: a free Space has no GPU,
# and the CUDA build is ~2 GB larger for nothing.
REQUIREMENTS = """\
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.14.0
rdkit==2026.3.3
numpy==2.4.6
pandas==3.0.6
pyarrow==25.0.1
"""

README = """\
---
title: CASMI 2026 molecule ID
emoji: 🧪
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
---

Paste a tandem mass spectrum; a transformer predicts which substructures the
molecule contains and ranks candidate structures of matching mass.

Built for the Enveda CASMI 2026 Kaggle competition.
"""


def read_token() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("HF_TOKEN=") and line.split("=", 1)[1].strip():
            return line.split("=", 1)[1].strip()
    raise SystemExit("HF_TOKEN missing from .env")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true")
    # Assemble into this folder and stop -- to test the exact Space layout
    # locally before uploading anything.
    ap.add_argument("--dry-run", type=Path, default=None)
    args = ap.parse_args()

    for f in (MODEL, POOL):
        if not f.exists():
            raise SystemExit(f"missing {f}")

    with tempfile.TemporaryDirectory() as t:
        d = args.dry_run or Path(t)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "app" / "app.py", d / "app.py")
        (d / "casmi").mkdir()
        (d / "casmi" / "__init__.py").write_text("")
        for m in MODULES:
            shutil.copy2(ROOT / "src" / "casmi" / m, d / "casmi" / m)
        shutil.copy2(MODEL, d / MODEL.name)
        shutil.copy2(POOL, d / POOL.name)
        (d / "requirements.txt").write_text(REQUIREMENTS)
        (d / "README.md").write_text(README)

        if args.dry_run:
            print(f"assembled in {d} (dry run, nothing uploaded)")
            return
        api = HfApi(token=read_token())
        repo_id = f"{api.whoami()['name']}/{SPACE_NAME}"
        api.create_repo(repo_id, repo_type="space", space_sdk="gradio",
                        private=not args.public, exist_ok=True)
        # app.py resolves paths from the repo layout by default; on the Space
        # the files sit next to it, so point it there.
        api.add_space_variable(repo_id, "MODEL_PATH", MODEL.name)
        api.add_space_variable(repo_id, "POOL_PATH", POOL.name)
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=d,
                          commit_message="Deploy CASMI 2026 demo")

    vis = "public" if args.public else "private"
    print(f"deployed ({vis}): https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
