"""
run_hawor_seq.py  -  Standalone per-sequence HaWoR runner (no visualisation).
Mirrors demo.py logic exactly; called as a subprocess by batch_hawor.py.

Usage (from within hawor conda env):
    cd /path/to/third_party/hawor
    python run_hawor_seq.py \
        --video_path /abs/path/to/video.mp4 \
        --out_npz   /abs/path/to/output.npz \
        [--checkpoint  weights/hawor/checkpoints/hawor.ckpt] \
        [--infiller_weight weights/hawor/checkpoints/infiller.pt] \
        [--img_focal FLOAT] \
        [--force_rerun]

Output npz keys
---------------
right_verts  : (T, 778, 3)  float32  – right hand vertices in world-coord
left_verts   : (T, 778, 3)  float32  – left  hand vertices in world-coord
R_w2c        : (T, 3, 3)    float32  – camera rotation  world→cam
t_w2c        : (T, 3)       float32  – camera translation world→cam
R_c2w        : (T, 3, 3)    float32  – camera rotation  cam→world
t_c2w        : (T, 3)       float32  – camera translation cam→world
img_focal    : scalar       float32  – focal length (pixels)
pred_trans   : (2, T, 3)    float32  – MANO root translation  [0=left,1=right]
pred_rot     : (2, T, 3)    float32  – MANO root rotation (axis-angle)
pred_hand_pose:(2,T,45)     float32  – MANO hand pose (axis-angle, 15 joints)
pred_betas   : (2, T, 10)   float32  – MANO shape coefficients
pred_valid   : (2, T)       float32  – validity mask per hand per frame
"""
import argparse
import os
import sys
import traceback
from glob import glob
from natsort import natsorted

# ── path setup (must be done before any HaWoR imports) ────────────────────────
# HAWOR_DIR is resolved from --hawor_dir arg or falls back to __file__ location
# (batch_hawor.py always passes --hawor_dir explicitly)
_THIS_FILE = os.path.dirname(os.path.abspath(__file__))

# Preliminary parse to get --hawor_dir before full arg parsing
_hawor_dir_default = _THIS_FILE  # fallback: file was copied into hawor dir
for _i, _a in enumerate(sys.argv):
    if _a == "--hawor_dir" and _i + 1 < len(sys.argv):
        _hawor_dir_default = sys.argv[_i + 1]
        break

HAWOR_DIR = _hawor_dir_default
sys.path.insert(0, HAWOR_DIR)
sys.path.insert(0, os.path.join(HAWOR_DIR, "thirdparty", "DROID-SLAM"))
sys.path.insert(0, os.path.join(HAWOR_DIR, "thirdparty", "DROID-SLAM", "droid_slam"))

# Always work from HAWOR_DIR so relative weight paths (./weights/…) resolve
os.chdir(HAWOR_DIR)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run HaWoR on a single video and save MANO + camera data to .npz")
    p.add_argument("--video_path",       required=True,
                   help="Absolute path to the input .mp4 file")
    p.add_argument("--out_npz",          required=True,
                   help="Absolute path for the output .npz file")
    p.add_argument("--hawor_dir",        default=HAWOR_DIR,
                   help="Absolute path to third_party/hawor (set by batch_hawor.py)")
    p.add_argument("--checkpoint",
                   default=os.path.join(HAWOR_DIR, "weights", "hawor", "checkpoints", "hawor.ckpt"))
    p.add_argument("--infiller_weight",
                   default=os.path.join(HAWOR_DIR, "weights", "hawor", "checkpoints", "infiller.pt"))
    p.add_argument("--img_focal",        type=float, default=None,
                   help="Override focal length (pixels); auto-estimated if omitted")
    p.add_argument("--force_rerun",      action="store_true",
                   help="Delete cached intermediates and re-run from scratch")
    return p.parse_args()


def main():
    args = parse_args()

    import numpy as np
    import torch

    # ── HaWoR pipeline imports ─────────────────────────────────────────────────
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
    from scripts.scripts_test_video.hawor_slam import hawor_slam
    from hawor.utils.process import run_mano, run_mano_left
    from lib.eval_utils.custom_utils import load_slam_cam

    # ── Step 1: detect & track ─────────────────────────────────────────────────
    print(f"[hawor_seq] detect_track: {args.video_path}", flush=True)
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

    # ── Step 2: motion estimation ──────────────────────────────────────────────
    print(f"[hawor_seq] motion estimation  frames {start_idx}–{end_idx}", flush=True)
    frame_chunks_all, img_focal = hawor_motion_estimation(
        args, start_idx, end_idx, seq_folder)

    # ── Step 3: SLAM ──────────────────────────────────────────────────────────
    slam_path = os.path.join(
        seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        existing = natsorted(
            glob(os.path.join(seq_folder, "SLAM/hawor_slam_w_scale_*.npz")))
        if existing:
            slam_path = existing[-1]
        else:
            print("[hawor_seq] running SLAM …", flush=True)
            hawor_slam(args, start_idx, end_idx)
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    # ── Step 4: infiller ──────────────────────────────────────────────────────
    print("[hawor_seq] infiller …", flush=True)
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        args, start_idx, end_idx, frame_chunks_all)

    # ── Step 5: MANO vertices (mirrors demo.py exactly) ───────────────────────
    N = pred_trans.shape[1]   # number of frames

    # right hand
    pred_glob_r = run_mano(
        pred_trans[1:2, :N], pred_rot[1:2, :N],
        pred_hand_pose[1:2, :N], betas=pred_betas[1:2, :N])
    right_verts = pred_glob_r["vertices"][0]          # (N, 778, 3) torch

    # left hand
    pred_glob_l = run_mano_left(
        pred_trans[0:1, :N], pred_rot[0:1, :N],
        pred_hand_pose[0:1, :N], betas=pred_betas[0:1, :N])
    left_verts = pred_glob_l["vertices"][0]           # (N, 778, 3) torch

    # coordinate flip: HaWoR internal → standard visual coords (same as demo.py)
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    R_c2w = torch.einsum("ij,njk->nik", R_x, R_c2w_sla_all)
    t_c2w = torch.einsum("ij,nj->ni",  R_x, t_c2w_sla_all)
    R_w2c = R_c2w.transpose(-1, -2)
    t_w2c = -torch.einsum("bij,bj->bi", R_w2c, t_c2w)
    right_verts = torch.einsum("ij,tnj->tni", R_x, right_verts.cpu())
    left_verts  = torch.einsum("ij,tnj->tni", R_x, left_verts.cpu())

    # ── Step 6: save ──────────────────────────────────────────────────────────
    out_npz = args.out_npz
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)

    pv = pred_valid if isinstance(pred_valid, np.ndarray) else pred_valid.numpy()

    np.savez_compressed(
        out_npz,
        right_verts   = right_verts.cpu().numpy().astype(np.float32),   # (T,778,3)
        left_verts    = left_verts.cpu().numpy().astype(np.float32),    # (T,778,3)
        R_w2c         = R_w2c[:N].cpu().numpy().astype(np.float32),     # (T,3,3)
        t_w2c         = t_w2c[:N].cpu().numpy().astype(np.float32),     # (T,3)
        R_c2w         = R_c2w[:N].cpu().numpy().astype(np.float32),     # (T,3,3)
        t_c2w         = t_c2w[:N].cpu().numpy().astype(np.float32),     # (T,3)
        img_focal     = np.float32(img_focal),
        pred_trans    = pred_trans.cpu().numpy().astype(np.float32),    # (2,T,3)
        pred_rot      = pred_rot.cpu().numpy().astype(np.float32),      # (2,T,3)
        pred_hand_pose= pred_hand_pose.cpu().numpy().astype(np.float32),# (2,T,45)
        pred_betas    = pred_betas.cpu().numpy().astype(np.float32),    # (2,T,10)
        pred_valid    = pv.astype(np.float32),                          # (2,T)
    )
    print(f"[hawor_seq] saved → {out_npz}", flush=True)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
