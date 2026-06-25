#!/usr/bin/env python3
"""
Convert MegaSAM cam_c2w outputs → HaWoR SLAM .npz format.

MegaSAM gives us metric camera poses for 60 sampled frames.
hawor_slam expects poses for EVERY video frame.
This script:
  1. Loads cam_c2w.npy from batch_megasam output
  2. SLERP-interpolates to match total video frame count
  3. Saves hawor-compatible SLAM .npz

Usage (run BEFORE demo.py):
  conda activate hawor
  cd $HAWOR_DIR

  # EgoDex
  python ../../V2AP/tools/megasam_to_hawor_slam.py \\
    --cam_c2w $PROJ/data_hub/ProcessedData/egocentric_depth/egodex/assemble_disassemble_tiles/1/cam_c2w.npy \\
    --video_path $PROJ/data_hub/RawData/EgoRawData/egodex/test/assemble_disassemble_tiles/1.mp4 \\
    --img_focal 681.6 \\
    --out_dir $PROJ/data_hub/RawData/EgoRawData/egodex/test/assemble_disassemble_tiles/1/SLAM

  # PH2D
  python ../../V2AP/tools/megasam_to_hawor_slam.py \\
    --cam_c2w $PROJ/data_hub/ProcessedData/egocentric_depth/ph2d_avp/1407-picking_orange_tj_2025-03-12_16-42-19/processed_episode_3/cam_c2w.npy \\
    --video_path $PROJ/data_hub/RawData/ThirdPersonRawData/ph2d/1407-picking_orange_tj_2025-03-12_16-42-19/processed_episode_3.mp4 \\
    --img_focal 271.2 \\
    --out_dir $PROJ/data_hub/RawData/ThirdPersonRawData/ph2d/1407-picking_orange_tj_2025-03-12_16-42-19/processed_episode_3/SLAM
"""

import argparse, os, sys
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp


def mat_to_quat_wxyz(R):
    """(N,3,3) → (N,4) quaternions in [w,x,y,z] order."""
    rot = Rotation.from_matrix(R)
    q_xyzw = rot.as_quat()          # scipy returns xyzw
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]  # → wxyz
    return q_wxyz


def slerp_poses(cam_c2w, src_frames, total_frames):
    """
    Interpolate cam_c2w (N_src, 4, 4) sampled at src_frames to all total_frames.
    Frames outside the src range get the nearest boundary pose.
    Returns cam_c2w_interp (total_frames, 4, 4).
    """
    R_src = cam_c2w[:, :3, :3]   # (N_src, 3, 3)
    t_src = cam_c2w[:, :3, 3]    # (N_src, 3)

    all_frames  = np.arange(total_frames, dtype=float)
    t_min, t_max = float(src_frames[0]), float(src_frames[-1])

    # Clamp to valid SLERP range (out-of-range frames hold boundary rotation)
    clamped = np.clip(all_frames, t_min, t_max)
    rot_obj  = Rotation.from_matrix(R_src)
    slerp_fn = Slerp(src_frames, rot_obj)
    R_interp = slerp_fn(clamped).as_matrix()   # (total_frames, 3, 3)

    # Translation: np.interp already extrapolates with boundary values
    t_interp = np.zeros((total_frames, 3))
    for d in range(3):
        t_interp[:, d] = np.interp(all_frames, src_frames, t_src[:, d])

    out = np.eye(4)[None].repeat(total_frames, 0)
    out[:, :3, :3] = R_interp
    out[:, :3,  3] = t_interp
    return out


def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_c2w",   required=True,
                    help="Path to cam_c2w.npy from batch_megasam output (N,4,4)")
    ap.add_argument("--video_path", required=True,
                    help="Original video (.mp4) to determine total frame count")
    ap.add_argument("--img_focal", type=float, required=True,
                    help="Calibrated focal length at original video resolution")
    ap.add_argument("--out_dir",    required=True,
                    help="Directory to save hawor_slam_w_scale_*.npz in")
    ap.add_argument("--start_idx",  type=int, default=None,
                    help="HaWoR start frame index (auto-detect if not given)")
    ap.add_argument("--end_idx",    type=int, default=None,
                    help="HaWoR end frame index (auto-detect if not given)")
    ap.add_argument("--max_megasam_frames", type=int, default=60,
                    help="MAX_FRAMES used in batch_megasam.py (default 60)")
    args = ap.parse_args()

    # ── Load MegaSAM output ───────────────────────────────────────────────────
    cam_c2w = np.load(args.cam_c2w)   # (N_mega, 4, 4)
    N_mega  = len(cam_c2w)
    print(f"  MegaSAM cam_c2w: {N_mega} frames from {args.cam_c2w}")

    # ── Total video frames ────────────────────────────────────────────────────
    total_frames = get_video_frame_count(args.video_path)
    print(f"  Video total frames: {total_frames}  ({args.video_path})")

    # ── Frame index mapping: which original frames did MegaSAM sample? ────────
    step       = max(1, total_frames // args.max_megasam_frames)
    src_frames = np.array([i * step for i in range(N_mega)], dtype=float)
    src_frames = np.clip(src_frames, 0, total_frames - 1)
    print(f"  MegaSAM sampled frames: step={step}, range [{int(src_frames[0])}, {int(src_frames[-1])}]")

    # ── Determine start/end for HaWoR SLAM file name ─────────────────────────
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx   = args.end_idx   if args.end_idx   is not None else total_frames  # HaWoR default: open-ended
    print(f"  HaWoR frame range: {start_idx} → {end_idx}")

    cam_c2w_interp = slerp_poses(cam_c2w, src_frames, total_frames)
    print(f"  Interpolated poses: {total_frames} frames  shape={cam_c2w_interp.shape}")
    # Slice to the requested [start_idx, end_idx] window
    cam_c2w_interp = cam_c2w_interp[start_idx : end_idx + 1]
    n_out = len(cam_c2w_interp)
    print(f"  Sliced to HaWoR window: {n_out} frames")

    # ── Convert to hawor traj format [tx, ty, tz, qx, qy, qz, qw] ───────────
    R_c2w = cam_c2w_interp[:, :3, :3]   # (N, 3, 3)
    t_c2w = cam_c2w_interp[:, :3,  3]   # (N, 3)
    q_wxyz = mat_to_quat_wxyz(R_c2w)    # (N, 4) wxyz
    # hawor load_slam_cam reads: pred_camq[:, [3,0,1,2]] = [w,x,y,z]→[x,y,z,w]
    # so store in [x,y,z,w] order (index 3=w)
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]   # (N, 4) xyzw
    traj   = np.concatenate([t_c2w, q_xyzw], axis=1)   # (N, 7)

    # ── Dummy tstamp / disps ──────────────────────────────────────────────────
    tstamp = np.arange(n_out, dtype=np.int32)
    disps  = np.ones((n_out, 1), dtype=np.float32) * 0.5  # dummy

    # ── Write .npz ────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir,
                            f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    W_orig = int(cv2.VideoCapture(args.video_path).get(cv2.CAP_PROP_FRAME_WIDTH))
    H_orig = int(cv2.VideoCapture(args.video_path).get(cv2.CAP_PROP_FRAME_HEIGHT))

    np.savez(out_path,
             tstamp=tstamp,
             disps=disps,
             traj=traj.astype(np.float32),
             img_focal=np.float32(args.img_focal),
             img_center=np.array([W_orig / 2.0, H_orig / 2.0], dtype=np.float32),
             scale=np.float32(1.0))   # MegaSAM output is already metric

    print(f"\n  ✅ Saved → {out_path}")
    print(f"     traj shape  : {traj.shape}")
    print(f"     scale       : 1.0 (MegaSAM metric)")
    print(f"\nNow run HaWoR (SLAM file pre-computed, will be skipped):")
    print(f"  conda activate hawor")
    print(f"  cd $HAWOR_DIR")
    print(f"  python demo.py --video_path {args.video_path} --img_focal {args.img_focal} --vis_mode world")


if __name__ == "__main__":
    main()
