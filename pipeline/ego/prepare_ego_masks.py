#!/usr/bin/env python3
"""
prepare_ego_masks.py — Auto-generate FoundationPose init masks for egocentric datasets
======================================================================================
Creates one representative frame + heuristic mask per task category.

Output structure (one mask per task category, NOT per sequence):
  obj_recon_input/egocentric/{task}/
    0.png         ← binary mask (255 = object, 0 = background)
    image.png     ← RGB frame used for mask

Datasets supported:
  egodex  — EgoDex (Apple Vision Pro, MP4 videos)
  taco    — TACO Egocentric (GoPro, PNG frames)

Usage:
  conda activate depth-pro
  python data/prepare_ego_masks.py --dataset egodex
  python data/prepare_ego_masks.py --dataset taco
  python data/prepare_ego_masks.py --dataset both    # run both
  python data/prepare_ego_masks.py --dataset egodex --task add_remove_lid  # single task

After running, upload to HuggingFace:
  python data/prepare_ego_masks.py --upload
"""

import os, sys, cv2, argparse
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# ── Paths ─────────────────────────────────────────────────────────────────────
EGO_RAW   = os.path.join(config.DATA_HUB, "RawData", "EgoRawData")
OUT_BASE  = os.path.join(config.DATA_HUB, "ProcessedData", "obj_recon_input", "egocentric")
EGODEX_ROOT = os.path.join(EGO_RAW, "egodex", "test")
TACO_ROOT   = os.path.join(EGO_RAW, "taco", "Egocentric_RGB_Videos")


# ── Frame extractors ──────────────────────────────────────────────────────────

def get_egodex_frame(task_dir: str) -> Optional[np.ndarray]:
    """Extract one representative frame from an EgoDex task (MP4 videos)."""
    episodes = sorted([f for f in os.listdir(task_dir) if f.endswith(".mp4")])
    if not episodes:
        return None

    # Try first episode, grab middle frame
    for ep in episodes[:3]:
        cap = cv2.VideoCapture(os.path.join(task_dir, ep))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 3)  # 1/3 into the video
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def get_taco_frame(task_dir: str) -> Optional[np.ndarray]:
    """Extract one representative frame from a TACO Ego task (PNG frames)."""
    episodes = sorted([d for d in os.listdir(task_dir)
                       if os.path.isdir(os.path.join(task_dir, d))])
    if not episodes:
        return None

    for ep in episodes[:3]:
        ep_dir = os.path.join(task_dir, ep)
        frames = sorted([f for f in os.listdir(ep_dir)
                         if f.endswith((".png", ".jpg"))])
        if not frames:
            continue
        # pick middle frame
        frame_path = os.path.join(ep_dir, frames[len(frames) // 3])
        img = cv2.imread(frame_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


# ── Mask generation ───────────────────────────────────────────────────────────

def generate_center_mask(rgb: np.ndarray, margin: float = 0.25) -> np.ndarray:
    """
    Generate a centered elliptical mask covering the central region.
    This is a conservative but reliable heuristic for egocentric views where
    the object-of-interest tends to be at the center.

    For better quality, replace with SAM2 interactive segmentation.
    """
    H, W = rgb.shape[:2]

    # Detect brightest central region (common for egocentric hand-object interaction)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Restrict to center crop
    r0, r1 = int(H * margin), int(H * (1 - margin))
    c0, c1 = int(W * margin), int(W * (1 - margin))
    mask_full = np.zeros((H, W), dtype=np.uint8)
    mask_full[r0:r1, c0:c1] = bright[r0:r1, c0:c1]

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN,  kernel)

    # Fallback: if mask is too small, use center ellipse
    if mask_full.sum() < (H * W * 0.02 * 255):
        mask_full = np.zeros((H, W), dtype=np.uint8)
        cy, cx = H // 2, W // 2
        ry, rx = int(H * 0.25), int(W * 0.25)
        Y, X = np.ogrid[:H, :W]
        mask_full[((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 <= 1] = 255

    return mask_full


# ── Per-dataset processing ────────────────────────────────────────────────────

def process_dataset(root: str, get_frame_fn, dataset_label: str, task_filter=None):
    tasks = sorted(os.listdir(root))
    if task_filter:
        tasks = [t for t in tasks if task_filter in t]

    done = skipped = failed = 0

    for task in tqdm(tasks, desc=f"Masks/{dataset_label}"):
        task_dir = os.path.join(root, task)
        if not os.path.isdir(task_dir):
            continue

        out_dir = os.path.join(OUT_BASE, task)
        mask_path = os.path.join(out_dir, "0.png")

        if os.path.exists(mask_path):
            tqdm.write(f"  ⏭  {task}: already has mask, skipping")
            skipped += 1
            continue

        rgb = get_frame_fn(task_dir)
        if rgb is None:
            tqdm.write(f"  ❌ {task}: could not extract frame")
            failed += 1
            continue

        mask = generate_center_mask(rgb)

        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(mask_path,
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),  # save as image.png (rgb)
                    )
        # Actually save properly:
        cv2.imwrite(os.path.join(out_dir, "image.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask)  # 0.png = binary mask

        pct = int(mask.mean() / 255 * 100)
        tqdm.write(f"  ✅ {task}: mask_area={pct}%")
        done += 1

    print(f"\n{dataset_label}: ✅ Done={done}  ⏭ Skipped={skipped}  ❌ Failed={failed}")
    return done, skipped, failed


# ── Upload to HuggingFace ─────────────────────────────────────────────────────

def upload_to_hf():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ pip install huggingface_hub")
        return

    api = HfApi()
    repo_id = "UCBProject/EgoDataMask"

    print(f"Creating/uploading to {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=OUT_BASE,
        path_in_repo="egocentric",
        commit_message="upload: EgoDex + TACO Ego object masks for FoundationPose",
    )
    print(f"✅ Uploaded {OUT_BASE} → {repo_id}/egocentric/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="both",
                        choices=["egodex", "taco", "both"],
                        help="Which egocentric dataset to process")
    parser.add_argument("--task", default=None,
                        help="Filter to a single task name substring")
    parser.add_argument("--upload", action="store_true",
                        help="Upload completed masks to HuggingFace UCBProject/EgoDataMask")
    args = parser.parse_args()

    os.makedirs(OUT_BASE, exist_ok=True)

    if args.upload:
        upload_to_hf()
        return

    if args.dataset in ("egodex", "both"):
        print(f"\n{'='*60}")
        print(f" EgoDex  ({EGODEX_ROOT})")
        print(f" Output  ({OUT_BASE})")
        print(f"{'='*60}")
        process_dataset(EGODEX_ROOT, get_egodex_frame, "EgoDex", args.task)

    if args.dataset in ("taco", "both"):
        print(f"\n{'='*60}")
        print(f" TACO Ego  ({TACO_ROOT})")
        print(f" Output    ({OUT_BASE})")
        print(f"{'='*60}")
        process_dataset(TACO_ROOT, get_taco_frame, "TACO_Ego", args.task)

    print(f"\n✅ All done. Output in: {OUT_BASE}")
    print(f"To upload: python data/prepare_ego_masks.py --upload")


if __name__ == "__main__":
    main()
