#!/usr/bin/env python3
"""
upload_taco_zips.py — Upload TACO Allocentric camera zip files to HuggingFace
Uploads one at a time, deletes local file after confirmed upload to free disk.

Usage:
    HF_TOKEN=hf_xxx conda run -n base python tools/upload_taco_zips.py

Files uploaded to: UCBProject/V2AP-TACOData (dataset)
Remote path:       Allocentric/{filename}
"""

import os, sys, time
from pathlib import Path
from huggingface_hub import HfApi, upload_file

# ── Config ────────────────────────────────────────────────────────────────────
TOKEN      = os.environ.get("HF_TOKEN") or input("HF_TOKEN: ").strip()
REPO_ID    = "UCBProject/V2AP-TACOData"
REPO_TYPE  = "dataset"
REMOTE_DIR = "Allocentric"  # subfolder inside HF repo
DOWNLOAD_DIR = Path(os.environ.get("TACO_DOWNLOAD_DIR", str(Path.home() / "Downloads")))

# Files to upload (in order: smallest first to free space quickly)
ZIPS = sorted([
    "21218078.zip",
    "22070938.zip",
    "22139905.zip",
    "22139908.zip",
    "22139909.zip",
    "22139910.zip",
    "22139911.zip",
    "22139913.zip",
    "22139916.zip",
    "22139946.zip",
], key=lambda f: (DOWNLOAD_DIR / f).stat().st_size if (DOWNLOAD_DIR / f).exists() else 0)

api = HfApi(token=TOKEN)

# ── Ensure repo exists ────────────────────────────────────────────────────────
try:
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True)
    print(f"✅ Repo ready: {REPO_ID}")
except Exception as e:
    print(f"⚠️  Repo create: {e}")

# ── Check what's already uploaded ────────────────────────────────────────────
try:
    existing = {
        f.rfilename for f in api.list_repo_tree(
            repo_id=REPO_ID, repo_type=REPO_TYPE, path_in_repo=REMOTE_DIR
        )
    }
except Exception:
    existing = set()

print(f"\nAlready on HF: {len(existing)} files")
print(f"To upload:     {len(ZIPS)} files\n")

# ── Upload loop ───────────────────────────────────────────────────────────────
for i, fname in enumerate(ZIPS, 1):
    local_path = DOWNLOAD_DIR / fname
    remote_path = f"{REMOTE_DIR}/{fname}"

    if not local_path.exists():
        print(f"[{i}/{len(ZIPS)}] ⚠️  NOT FOUND locally: {fname}, skipping")
        continue

    size_gb = local_path.stat().st_size / 1e9

    if remote_path in existing:
        print(f"[{i}/{len(ZIPS)}] ⏭  Already on HF: {fname} ({size_gb:.1f} GB)")
        # Safe to delete if already uploaded
        answer = input(f"    Delete local copy? [y/N]: ").strip().lower()
        if answer == 'y':
            local_path.unlink()
            print(f"    🗑  Deleted {fname}")
        continue

    print(f"\n[{i}/{len(ZIPS)}] ▶  Uploading {fname} ({size_gb:.1f} GB) ...")
    t0 = time.time()

    try:
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=TOKEN,
            commit_message=f"Upload TACO Allocentric cam {fname}",
        )
        elapsed = (time.time() - t0) / 60
        speed   = size_gb / (elapsed / 60) if elapsed > 0 else 0
        print(f"    ✅ Done in {elapsed:.1f} min ({speed:.1f} GB/h)")

        # Auto-delete after confirmed upload to free disk space
        local_path.unlink()
        print(f"    🗑  Deleted local {fname} (disk freed)")

    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted at {fname}. File NOT deleted.")
        sys.exit(1)
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        print(f"    Local file kept. Retrying next run.")

print(f"\n✅ All done! Repo: https://huggingface.co/datasets/{REPO_ID}")
