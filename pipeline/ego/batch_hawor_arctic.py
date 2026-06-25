"""
Batch HaWoR inference on ARCTIC egocentric videos.
Processes each sequence through HaWoR's full pipeline and caches MANO output.

Usage:
  conda activate hawor
  python -m data.batch_hawor_arctic

Output:
  For each sequence, saves to hawor_arctic_cache/:
    {subject}__{sequence}.npz containing:
      - right_verts: (N, 778, 3) right hand vertices (world space)
      - left_verts: (N, 778, 3) left hand vertices (world space)
      - R_w2c: (N, 3, 3) world-to-camera rotation
      - t_w2c: (N, 3) world-to-camera translation
      - img_focal: float, estimated focal length
      - pred_trans/rot/hand_pose/betas: raw MANO params
"""

import os
import sys
import json
import numpy as np
import torch
from glob import glob
from natsort import natsorted
from tqdm import tqdm

# Project config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from data.megasam_utils import load_megasam_K, load_megasam_K_fx

sys.path.insert(0, config.HAWOR_DIR)
sys.path.insert(0, os.path.join(config.HAWOR_DIR, 'thirdparty', 'DROID-SLAM', 'droid_slam'))

from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam

ARCTIC_ROOT = config.ARCTIC_ROOT
VIDEO_DIR = os.path.join(ARCTIC_ROOT, "arctic_egocam_videos")
CACHE_DIR = config.HAWOR_CACHE
os.makedirs(CACHE_DIR, exist_ok=True)

# Sequences where DROID-SLAM scale estimation permanently fails
# (no valid hand-keyframe pairs found → "not enough values to unpack").
# These are data-level limitations; skip them rather than error each run.
SLAM_SKIP_LIST = {
    ("s09", "scissors_use_01"),
    ("s10", "scissors_use_02"),
}


def process_sequence(video_path, subject, seq_name):
    """Run full HaWoR pipeline on one ARCTIC egocam video."""
    cache_key = f"{subject}__{seq_name}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npz")

    if os.path.exists(cache_path):
        return "skip"

    if (subject, seq_name) in SLAM_SKIP_LIST:
        print(f"  ⚠️  Skipping {subject}/{seq_name} (known SLAM failure)")
        return "skip"

    try:
        # Create args namespace
        class Args:
            pass
        args = Args()
        args.video_path = video_path
        args.input_type = 'file'
        # Use MegaSAM focal length if available; falls back to HaWoR self-estimate (None).
        megasam_K = load_megasam_K(seq_name)
        args.img_focal = load_megasam_K_fx(seq_name)
        if args.img_focal is not None:
            print(f"  [MegaSAM] Using focal={args.img_focal:.1f}px for {seq_name}")
        else:
            print(f"  [MegaSAM] No intrinsics found for {seq_name}, using HaWoR self-estimate")
        args.checkpoint = os.path.join(config.HAWOR_DIR, 'weights/hawor/checkpoints/hawor.ckpt')
        args.infiller_weight = os.path.join(config.HAWOR_DIR, 'weights/hawor/checkpoints/infiller.pt')
        args.vis_mode = 'cam'

        # ALL HaWoR pipeline steps use relative paths (weights, MANO models, etc.)
        # that only resolve when CWD == HAWOR_DIR.  Wrap everything in one chdir.
        _orig_cwd = os.getcwd()
        os.chdir(config.HAWOR_DIR)
        try:
            # Step 1: Detect and track hands
            start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

            # Step 2: HaWoR motion estimation
            frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

            # Step 3: SLAM
            slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
            if not os.path.exists(slam_path):
                # Fallback: find any existing SLAM file (end_idx may differ from video length)
                existing_slams = natsorted(glob(os.path.join(seq_folder, "SLAM/hawor_slam_w_scale_*.npz")))
                if existing_slams:
                    slam_path = existing_slams[-1]
                    print(f"  [SLAM] Using existing file: {os.path.basename(slam_path)}")
                else:
                    hawor_slam(args, start_idx, end_idx)
            R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

            # Step 4: Infiller
            pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
                args, start_idx, end_idx, frame_chunks_all
            )

            N = pred_trans.shape[1]

            # Step 5: Get MANO vertices
            pred_glob_r = run_mano(
                pred_trans[1:2, :N], pred_rot[1:2, :N],
                pred_hand_pose[1:2, :N], betas=pred_betas[1:2, :N]
            )
            right_verts = pred_glob_r['vertices'][0].cpu().numpy()  # (N, 778, 3)

            pred_glob_l = run_mano_left(
                pred_trans[0:1, :N], pred_rot[0:1, :N],
                pred_hand_pose[0:1, :N], betas=pred_betas[0:1, :N]
            )
            left_verts = pred_glob_l['vertices'][0].cpu().numpy()  # (N, 778, 3)
        finally:
            os.chdir(_orig_cwd)

        # Apply R_x coordinate transform (same as demo.py)
        R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        right_verts = np.einsum('ij,tnj->tni', R_x, right_verts)
        left_verts = np.einsum('ij,tnj->tni', R_x, left_verts)

        R_c2w_np = R_c2w_sla_all.cpu().numpy()
        t_c2w_np = t_c2w_sla_all.cpu().numpy()
        R_c2w_np = np.einsum('ij,njk->nik', R_x, R_c2w_np)
        t_c2w_np = np.einsum('ij,nj->ni', R_x, t_c2w_np)
        R_w2c_np = np.transpose(R_c2w_np, (0, 2, 1))
        t_w2c_np = -np.einsum('nij,nj->ni', R_w2c_np, t_c2w_np)

        # Save cache
        save_kwargs = dict(
            right_verts=right_verts,
            left_verts=left_verts,
            R_w2c=R_w2c_np[:N],
            t_w2c=t_w2c_np[:N],
            img_focal=img_focal,
            pred_trans=pred_trans.cpu().numpy(),
            pred_rot=pred_rot.cpu().numpy(),
            pred_hand_pose=pred_hand_pose.cpu().numpy(),
            pred_betas=pred_betas.cpu().numpy(),
        )
        # Also persist MegaSAM K so downstream scripts can reproject consistently
        if megasam_K is not None:
            save_kwargs['megasam_K'] = megasam_K
        np.savez_compressed(cache_path, **save_kwargs)
        return "ok"

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return "error"


def main():
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ Video directory not found: {VIDEO_DIR}")
        print("   Run arctic_prepare_videos.py first to create egocam videos.")
        return

    # Collect all videos
    videos = []
    for subj in sorted(os.listdir(VIDEO_DIR)):
        subj_dir = os.path.join(VIDEO_DIR, subj)
        if not os.path.isdir(subj_dir):
            continue
        for vf in sorted(os.listdir(subj_dir)):
            if vf.endswith('.mp4'):
                seq_name = vf.replace('.mp4', '')
                videos.append((subj, seq_name, os.path.join(subj_dir, vf)))

    print(f"📊 ARCTIC sequences: {len(videos)}")
    cached = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.npz')])
    print(f"📦 Already cached: {cached}")

    ok, skip, err = 0, 0, 0
    for subj, seq_name, vpath in tqdm(videos, desc="HaWoR"):
        tqdm.write(f"  {subj}/{seq_name}")
        status = process_sequence(vpath, subj, seq_name)
        if status == "ok":
            ok += 1
        elif status == "skip":
            skip += 1
        else:
            err += 1

    print(f"\n{'='*60}")
    print(f"✅ Done! Processed={ok}, Skipped={skip}, Errors={err}")
    print(f"📁 Cache: {CACHE_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
