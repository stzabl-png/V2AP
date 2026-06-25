#!/usr/bin/env python3
"""
Batch MegaSAM inference — egocentric depth for PH2D AVP + EgoDex.

Pipeline (models loaded ONCE, reused across all episodes):
  Step 1  Extract frames from HDF5/MP4 → temp PNG dir
  Step 2  DepthAnything  → per-frame relative disparity (in-memory)
  Step 3  UniDepth       → per-frame metric depth + scale alignment
  Step 4  DROID-SLAM     → temporally-consistent metric depth + camera poses
  Step 5  Save outputs   → standardised .npz format

Output layout:
  <DATA_HUB>/ProcessedData/egocentric_depth/
    ph2d_avp/{task}/{episode}/
      depth.npz        depths (N,H,W) float32 [metres]
      cam_c2w.npy      camera-to-world (N,4,4)
      K.npy            intrinsic matrix (3,3)
      motion_prob.npy  dynamic-object probability (N,H,W)
      meta.json        {seq_id, n_frames, hw, motion_score, calibrated_fx}
    egodex/{task}/{episode}/
      (same structure)

Usage:
  conda activate mega_sam
  cd mega-sam                     # from repo root; populated by git clone --recursive
  python ../data/batch_megasam.py --dataset both --workers 1

  # Resume interrupted run
  python ../data/batch_megasam.py --resume

  # Only high-motion episodes (skip SLAM on static cam sequences)
  python ../data/batch_megasam.py --min-motion 8.0

  # Process a slice (for parallel multi-GPU runs)
  python ../data/batch_megasam.py --dataset egodex --start 0 --end 500
"""

import os, sys, json, argparse, shutil, glob, time
import numpy as np
import torch, cv2
from tqdm import tqdm
from natsort import natsorted
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEGASAM_DIR  = os.path.join(PROJECT_DIR, "thirdparty", "megasam")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(MEGASAM_DIR, "UniDepth"))
sys.path.insert(0, os.path.join(MEGASAM_DIR, "base", "droid_slam"))
sys.path.insert(0, os.path.join(MEGASAM_DIR, "Depth-Anything"))

import config as proj_cfg

DATA_ROOT      = os.path.join(proj_cfg.DATA_HUB, "RawData", "ThirdPersonRawData")
EGO_ROOT       = os.path.join(proj_cfg.DATA_HUB, "RawData", "EgoRawData")
PH2D_ROOT      = os.path.join(DATA_ROOT, "ph2d")
PH2D_META      = os.path.join(DATA_ROOT, "ph2d", "ph2d_metadata.json")
EGODEX_ROOT    = os.path.join(EGO_ROOT,  "egodex", "test")
TACO_EGO_ROOT  = os.path.join(EGO_ROOT,  "taco",   "Egocentric_RGB_Videos")
HOI4D_ROOT     = os.path.join(EGO_ROOT,  "hoi4d",  "HOI4D_release")
OUT_BASE    = os.path.join(proj_cfg.DATA_HUB, "ProcessedData", "egocentric_depth")
WORK_DIR    = os.path.join(proj_cfg.DATA_HUB, "ProcessedData", "egocentric_depth", "_workdir")   # temp frames + DA/UD outputs

# MegaSAM systematically underestimates depth by ~20% on egocentric HOI4D-style cameras.
# Scale all depth outputs up by this factor for downstream alignment accuracy.
MEGASAM_DEPTH_SCALE = 1.20

DA_CKPT     = os.path.join(MEGASAM_DIR, "Depth-Anything", "checkpoints",
                            "depth_anything_vitl14.pth")
MEGA_CKPT   = os.path.join(MEGASAM_DIR, "checkpoints", "megasam_final.pth")
UNIDEPTH_ID = "lpiccinelli/unidepth-v2-vitl14"
UNIDEPTH_REV = "1d0d3c52f60b5164629d279bb9a7546458e6dcc4"

# ── calibrated focal lengths (SLAM opt-intr auto-calib, x1.10 corrected) ──────
# Run data/calibrate_dataset_fx.py to regenerate for new datasets.
CALIB_FX = {
    # px @640px wide  (all frames resized so long-side = LONG_DIM = 640)
    "ph2d_avp": 271.2,   # px @640px wide (median over 8 seqs)
    "egodex":   249.4,   # SLAM opt-intr auto-calib x1.10 (was 227.8)
    "taco":     455.3,   # SLAM opt-intr auto-calib x1.10
}
# Generic fallback: assume 60deg FoV -> fx = W / (2*tan(30deg)) = W*sqrt(3)/2
CALIB_FX_FALLBACK_RATIO = 0.866  # relative to frame width

LONG_DIM   = 640    # resize long-side to this
MAX_FRAMES = 60     # frames per episode fed to DROID-SLAM


# ══════════════════════════════════════════════════════════════════════════════
# Episode enumeration
# ══════════════════════════════════════════════════════════════════════════════

def enumerate_ph2d_avp():
    """Returns list of dicts {seq_id, cam_key, hdf5_path}."""
    with open(PH2D_META) as f:
        meta = json.load(f)
    eps = []
    for task_name, attrs in meta["per_task_attributes"].items():
        if attrs.get("embodiment_type") != "human_avp":
            continue
        task_dir = os.path.join(PH2D_ROOT, task_name)
        for hdf5 in natsorted(glob.glob(os.path.join(task_dir, "*.hdf5"))):
            ep_name = os.path.splitext(os.path.basename(hdf5))[0]
            eps.append({
                "seq_id":   f"ph2d_avp/{task_name}/{ep_name}",
                "cam_key":  "ph2d_avp",
                "hdf5":     hdf5,
                "mp4":      None,
            })
    return eps


def enumerate_egodex():
    """Returns list of dicts {seq_id, cam_key, mp4_path, hdf5_path}."""
    eps = []
    for task in natsorted(os.listdir(EGODEX_ROOT)):
        task_dir = os.path.join(EGODEX_ROOT, task)
        if not os.path.isdir(task_dir):
            continue
        for mp4 in natsorted(glob.glob(os.path.join(task_dir, "*.mp4"))):
            stem = os.path.splitext(os.path.basename(mp4))[0]
            hdf5 = os.path.join(task_dir, stem + ".hdf5")
            if not os.path.exists(hdf5):
                continue
            eps.append({
                "seq_id":  f"egodex/{task}/{stem}",
                "cam_key": "egodex",
                "mp4":     mp4,
                "hdf5":    hdf5,
            })
    return eps


def enumerate_taco_ego():
    """Returns list of dicts for TACO Egocentric_RGB_Videos."""
    eps = []
    if not os.path.isdir(TACO_EGO_ROOT):
        return eps
    for triplet in natsorted(os.listdir(TACO_EGO_ROOT)):
        triplet_dir = os.path.join(TACO_EGO_ROOT, triplet)
        if not os.path.isdir(triplet_dir):
            continue
        for seq in natsorted(os.listdir(triplet_dir)):
            mp4 = os.path.join(triplet_dir, seq, "color.mp4")
            if os.path.exists(mp4):
                eps.append({
                    "seq_id":  f"taco/{triplet}/{seq}",
                    "cam_key": "taco",
                    "mp4":     mp4,
                    "hdf5":    None,
                })
    return eps


def enumerate_hoi4d():
    """Returns list of dicts for HOI4D align_rgb sequences."""
    eps = []
    if not os.path.isdir(HOI4D_ROOT):
        return eps
    for mp4 in natsorted(glob.glob(
            os.path.join(HOI4D_ROOT, "**", "image.mp4"), recursive=True)):
        rel    = os.path.relpath(mp4, HOI4D_ROOT)   # e.g. ZY.../H1/.../align_rgb/image.mp4
        seq_id = "hoi4d/" + rel.replace("/align_rgb/image.mp4", "").replace("/", "_")
        eps.append({
            "seq_id":  seq_id,
            "cam_key": "hoi4d",
            "mp4":     mp4,
            "hdf5":    None,
        })
    return eps


# ══════════════════════════════════════════════════════════════════════════════
# Frame extraction (→ temp PNG dir)
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames_to_dir(ep, frame_dir, max_frames=MAX_FRAMES):
    """Extract frames and return (n_frames, width, height)."""
    os.makedirs(frame_dir, exist_ok=True)
    if ep["hdf5"] and ep["cam_key"] == "ph2d_avp":
        return _extract_hdf5(ep["hdf5"], frame_dir, max_frames)
    else:
        return _extract_mp4(ep["mp4"], frame_dir, max_frames)


def _extract_hdf5(hdf5_path, frame_dir, max_frames):
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        raw = f["observation.image.left"][:]
    n = min(len(raw), max_frames)
    H_out = W_out = None
    for i in range(n):
        buf = np.frombuffer(raw[i].tobytes(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img, H_out, W_out = _resize_long(img)
        cv2.imwrite(os.path.join(frame_dir, f"frame_{i:04d}.png"), img)
    return n, W_out, H_out


def _extract_mp4(mp4_path, frame_dir, max_frames):
    cap   = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // max_frames)
    idxs  = list(range(0, total, step))[:max_frames]
    H_out = W_out = None
    for out_i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame, H_out, W_out = _resize_long(frame)
        cv2.imwrite(os.path.join(frame_dir, f"frame_{out_i:04d}.png"), frame)
    cap.release()
    n = len(glob.glob(os.path.join(frame_dir, "*.png")))
    return n, W_out, H_out


def _resize_long(img, long_dim=LONG_DIM):
    h, w = img.shape[:2]
    if w >= h:
        nw, nh = long_dim, int(round(long_dim * h / w))
    else:
        nh, nw = long_dim, int(round(long_dim * w / h))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img, nh, nw


# ══════════════════════════════════════════════════════════════════════════════
# Model loaders (called once)
# ══════════════════════════════════════════════════════════════════════════════

def load_depth_anything():
    from depth_anything.dpt import DPT_DINOv2
    from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
    from torchvision.transforms import Compose
    model = DPT_DINOv2(encoder="vitl", features=256,
                        out_channels=[256, 512, 1024, 1024],
                        localhub=False).cuda()
    model.load_state_dict(torch.load(DA_CKPT, map_location="cpu", weights_only=False), strict=True)
    model.eval()
    transform = Compose([
        Resize(width=768, height=768, resize_target=False,
               keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method="upper_bound",
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    print("  ✅ DepthAnything loaded")
    return model, transform


def load_unidepth():
    from unidepth.models import UniDepthV2
    model = UniDepthV2.from_pretrained(UNIDEPTH_ID, revision=UNIDEPTH_REV)
    model = model.to("cuda").eval()
    print("  ✅ UniDepthV2 loaded")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: DepthAnything
# ══════════════════════════════════════════════════════════════════════════════

def run_da_on_dir(da_model, da_transform, frame_dir, da_dir):
    """Run DA on all PNGs in frame_dir; save .npy to da_dir."""
    import torch.nn.functional as F
    os.makedirs(da_dir, exist_ok=True)
    pngs = natsorted(glob.glob(os.path.join(frame_dir, "*.png")))
    disp_list = []
    for p in pngs:
        raw = cv2.imread(p)[..., :3]
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) / 255.0
        h, w = img.shape[:2]
        t = da_transform({"image": img})["image"]
        t = torch.from_numpy(t).unsqueeze(0).cuda()
        with torch.no_grad():
            disp = da_model(t)
        disp = F.interpolate(disp[None], (h, w), mode="bilinear",
                             align_corners=False)[0, 0]
        d = disp.cpu().float().numpy()
        fname = os.path.splitext(os.path.basename(p))[0]
        np.save(os.path.join(da_dir, fname + ".npy"), d)
        disp_list.append(d)
    return disp_list


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: UniDepth (metric scale)
# ══════════════════════════════════════════════════════════════════════════════

def run_unidepth_on_dir(ud_model, frame_dir, ud_dir, calibrated_fx):
    """
    Run UniDepth; override the estimated fx with calibrated_fx when saving fov.
    Saves .npz (depth, fov) per frame.  Returns list of metric depth arrays.
    """
    os.makedirs(ud_dir, exist_ok=True)
    pngs  = natsorted(glob.glob(os.path.join(frame_dir, "*.png")))
    depth_list = []
    for p in pngs:
        rgb = np.array(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
        t   = torch.from_numpy(rgb).permute(2, 0, 1).cuda()
        with torch.no_grad():
            pred = ud_model.infer(t)
        depth = pred["depth"][0, 0].cpu().float().numpy()
        # Use calibrated fx instead of UniDepth's own estimate for FOV field
        fov_calibrated = float(
            np.degrees(2 * np.arctan(depth.shape[1] / (2 * calibrated_fx)))
        )
        fname = os.path.splitext(os.path.basename(p))[0]
        np.savez(os.path.join(ud_dir, fname + ".npz"),
                 depth=np.float32(depth), fov=fov_calibrated)
        depth_list.append(depth)
    return depth_list


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: DROID-SLAM
# ══════════════════════════════════════════════════════════════════════════════

def run_droid(frame_dir, da_dir, ud_dir, seq_id, calibrated_fx,
              opt_intr=False):
    """
    In-process DROID-SLAM (no subprocess).
    Returns (traj_est, depth_est, motion_prob, K_opt, cam_c2w)
    """
    import torch.nn.functional as F
    from droid import Droid
    from lietorch import SE3

    image_list = natsorted(glob.glob(os.path.join(frame_dir, "*.png")))
    da_paths   = natsorted(glob.glob(os.path.join(da_dir, "*.npy")))
    ud_paths   = natsorted(glob.glob(os.path.join(ud_dir, "*.npz")))

    if not image_list or not da_paths or not ud_paths:
        raise RuntimeError(f"Missing inputs for {seq_id}")

    img_0  = cv2.imread(image_list[0])
    h0, w0 = img_0.shape[:2]

    # Build K from calibrated fx (not from UniDepth's FOV estimate)
    ff = calibrated_fx
    K  = np.array([[ff, 0, w0/2.0],
                   [0,  ff, h0/2.0],
                   [0,  0,  1.0  ]], dtype=np.float64)

    # Align DepthAnything disparity to UniDepth metric depth
    scales, shifts, mono_disp_list = [], [], []
    fovs = []
    for da_p, ud_p in zip(da_paths, ud_paths):
        da_disp = np.float32(np.load(da_p))
        ud_data = np.load(ud_p)
        metric_depth = ud_data["depth"]
        fovs.append(float(ud_data["fov"]))
        da_disp = cv2.resize(da_disp,
                             (metric_depth.shape[1], metric_depth.shape[0]),
                             interpolation=cv2.INTER_NEAREST_EXACT)
        gt_disp    = 1.0 / (metric_depth + 1e-8)
        valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
        gt_disp[valid_mask] = 1e-2
        sky_ratio  = np.sum(da_disp < 0.01) / da_disp.size
        if sky_ratio > 0.5:
            m = da_disp > 0.01
            scale = np.median((gt_disp[m]-np.median(gt_disp[m])+1e-8) /
                              (da_disp[m]-np.median(da_disp[m])+1e-8))
            shift = np.median(gt_disp[m] - scale * da_disp[m])
        else:
            gt_ms  = gt_disp - np.median(gt_disp) + 1e-8
            da_ms  = da_disp - np.median(da_disp) + 1e-8
            scale  = np.median(gt_ms / da_ms)
            shift  = np.median(gt_disp - scale * da_disp)
        scales.append(scale); shifts.append(shift)
        mono_disp_list.append(da_disp)

    ss = np.array(scales) * np.array(shifts)
    med_idx = np.argmin(np.abs(ss - np.median(ss)))
    a_scale, a_shift = scales[med_idx], shifts[med_idx]
    norm_scale = (np.percentile(a_scale * np.array(mono_disp_list) + a_shift, 98)
                  / 2.0)
    aligns = (a_scale, a_shift, norm_scale)

    # ── image generator ───────────────────────────────────────────────────────
    def image_stream(image_list, mono_disp_list, K, aligns):
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        for t, img_file in enumerate(image_list):
            image = cv2.imread(img_file)
            mono_disp = mono_disp_list[t]
            depth = np.clip(
                1.0 / ((1.0/aligns[2]) * (aligns[0]*mono_disp + aligns[1])),
                1e-4, 1e4)
            depth[depth < 1e-2] = 0.0
            h0_, w0_ = image.shape[:2]
            h1 = int(h0_ * np.sqrt((384*512)/(h0_*w0_)))
            w1 = int(w0_ * np.sqrt((384*512)/(h0_*w0_)))
            image = cv2.resize(image, (w1, h1), cv2.INTER_AREA)
            image = image[:h1-h1%8, :w1-w1%8]
            image_t = torch.as_tensor(image).permute(2,0,1)
            depth_t = torch.as_tensor(depth)
            depth_t = F.interpolate(depth_t[None, None], (h1, w1),
                                    mode="nearest-exact").squeeze()
            depth_t = depth_t[:h1-h1%8, :w1-w1%8]
            mask_t  = torch.ones_like(depth_t)
            intr = torch.as_tensor([fx, fy, cx, cy])
            intr[0::2] *= w1 / w0_
            intr[1::2] *= h1 / h0_
            yield t, image_t[None], depth_t, intr, mask_t

    # ── DROID-SLAM tracking ───────────────────────────────────────────────────
    class _Args:
        weights         = MEGA_CKPT
        buffer          = 512
        image_size      = [h0, w0]
        disable_vis     = True
        beta            = 0.3
        filter_thresh   = 2.0
        warmup          = 8
        keyframe_thresh = 2.0
        frontend_thresh = 12.0
        frontend_window = 25
        frontend_radius = 2
        frontend_nms    = 1
        stereo          = False
        depth           = True
        upsample        = False
        backend_thresh  = 16.0
        backend_radius  = 2
        backend_nms     = 3
        scene_name      = seq_id.replace("/", "_")

    args_obj  = _Args()
    rgb_list  = []
    depth_list_raw = []
    droid = None

    stream = image_stream(image_list, mono_disp_list, K, aligns)
    for t, image, depth, intrinsics, mask in tqdm(stream,
                                                   total=len(image_list),
                                                   desc=f"  DROID {seq_id[-30:]}",
                                                   leave=False):
        rgb_list.append(image[0])
        depth_list_raw.append(depth)
        if t == 0:
            args_obj.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args_obj)
        droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)

    droid.track_final(t, image, depth, intrinsics=intrinsics, mask=mask)

    traj_est, depth_est, motion_prob = droid.terminate(
        image_stream(image_list, mono_disp_list, K, aligns),
        _opt_intr=opt_intr,     # False = trust calibrated fx
        full_ba=True,
        scene_name=args_obj.scene_name,
    )

    # ── Extract outputs ───────────────────────────────────────────────────────
    n_frames   = traj_est.shape[0]
    intr_all   = droid.video.intrinsics[:n_frames].cpu().numpy() * 8.0  # (N,4)
    K_opt      = np.eye(3)
    K_opt[0,0] = intr_all[0, 0]; K_opt[1,1] = intr_all[0, 1]
    K_opt[0,2] = intr_all[0, 2]; K_opt[1,2] = intr_all[0, 3]

    # Camera-to-world poses
    poses_th = torch.as_tensor(traj_est, device="cpu")
    cam_c2w  = SE3(poses_th).inv().matrix().numpy()   # (N,4,4)

    # Depth maps (float32, metres)
    depths_m = np.float32(1.0 / (depth_est + 1e-6))  # depth_est is disparity

    return depths_m, motion_prob, K_opt, cam_c2w


# ══════════════════════════════════════════════════════════════════════════════
# Output saver
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(ep, depths, motion_prob, K, cam_c2w, n_frames, hw,
                 motion_score, calibrated_fx):
    """Save all outputs in standardised format under OUT_BASE."""
    out_dir = os.path.join(OUT_BASE, ep["seq_id"])
    os.makedirs(out_dir, exist_ok=True)

    scaled_depths = depths[:n_frames].astype(np.float32) * MEGASAM_DEPTH_SCALE
    # Apply scale to camera translations too (rotations are invariant to scale)
    scaled_c2w = cam_c2w[:n_frames].copy().astype(np.float32)
    scaled_c2w[:, :3, 3] *= MEGASAM_DEPTH_SCALE

    np.savez_compressed(os.path.join(out_dir, "depth.npz"),
                        depths=scaled_depths)
    np.save(os.path.join(out_dir, "cam_c2w.npy"), scaled_c2w)
    np.save(os.path.join(out_dir, "K.npy"), K.astype(np.float32))
    np.save(os.path.join(out_dir, "motion_prob.npy"),
            motion_prob[:n_frames].astype(np.float32))

    meta = {
        "seq_id":           ep["seq_id"],
        "cam_key":          ep["cam_key"],
        "n_frames":         int(n_frames),
        "hw":               list(hw),
        "motion_score":     float(motion_score) if motion_score else None,
        "calibrated_fx":    float(calibrated_fx),
        "K":                K.tolist(),
        "depth_scale":      MEGASAM_DEPTH_SCALE,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def is_done(ep):
    out_dir = os.path.join(OUT_BASE, ep["seq_id"])
    return (os.path.exists(os.path.join(out_dir, "depth.npz")) and
            os.path.exists(os.path.join(out_dir, "meta.json")))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch MegaSAM depth inference for egocentric datasets."
    )
    parser.add_argument("--dataset",
                        choices=["ph2d_avp", "egodex", "taco", "hoi4d", "all_ego", "both"],
                        default="all_ego",
                        help="Dataset to process. 'all_ego' runs egodex+taco+hoi4d.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip episodes that already have depth.npz output")
    parser.add_argument("--min-motion", type=float, default=0.0,
                        help="Skip episodes with motion score below this "
                             "(0 = process all; recommend 3.0 for SLAM quality)")
    parser.add_argument("--start", type=int, default=0,
                        help="Episode index start (for multi-GPU slicing)")
    parser.add_argument("--end", type=int, default=None,
                        help="Episode index end (exclusive)")
    parser.add_argument("--opt-intr", action="store_true",
                        help="Allow DROID-SLAM to refine focal length (slower)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List episodes without processing")
    parser.add_argument("--seq-ids", default=None,
                        help="Comma-separated seq_ids, e.g. "
                             "hoi4d/ZY.../... or egodex/task/ep")
    args = parser.parse_args()

    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    # ── Collect episodes ───────────────────────────────────────────────────────
    if args.seq_ids:
        # Accept comma-separated or space-separated list
        seq_id_list = [s.strip() for s in args.seq_ids.replace(",", " ").split()]
        # Determine which enumerators are needed based on seq_id prefixes
        needed = set()
        for sid in seq_id_list:
            prefix = sid.split("/")[0]
            needed.add(prefix)

        all_eps = {}
        if "ph2d_avp" in needed:
            for ep in enumerate_ph2d_avp():
                all_eps[ep["seq_id"]] = ep
        if "egodex" in needed:
            for ep in enumerate_egodex():
                all_eps[ep["seq_id"]] = ep
        if "taco_ego" in needed or "taco" in needed:
            for ep in enumerate_taco_ego():
                all_eps[ep["seq_id"]] = ep
        if "hoi4d" in needed:
            for ep in enumerate_hoi4d():
                all_eps[ep["seq_id"]] = ep

        episodes = []
        for sid in seq_id_list:
            if sid in all_eps:
                episodes.append(all_eps[sid])
            else:
                print(f"  seq_id not found: {sid}")
    else:
        episodes = []
        if args.dataset in ("ph2d_avp", "both"):
            episodes += enumerate_ph2d_avp()
        if args.dataset in ("egodex", "both", "all_ego"):
            episodes += enumerate_egodex()
        if args.dataset in ("taco", "all_ego"):
            episodes += enumerate_taco_ego()
        if args.dataset in ("hoi4d", "all_ego"):
            episodes += enumerate_hoi4d()

    print(f"\n  Total episodes: {len(episodes)}")

    # ── Motion score pre-screening (if min_motion > 0) ────────────────────────
    if args.min_motion > 0:
        sys.path.insert(0, os.path.join(PROJECT_DIR, "data"))
        from screen_camera_motion import (
            camera_motion_score, sample_frames_ph2d, sample_frames_mp4
        )
        print(f"  Pre-screening motion (threshold={args.min_motion}px)...")
        scored = []
        for ep in tqdm(episodes, desc="Screening"):
            if ep["cam_key"] == "ph2d_avp":
                frames = sample_frames_ph2d(ep["hdf5"])
            else:
                frames = sample_frames_mp4(ep["mp4"])
            score, _ = camera_motion_score(frames)
            ep["motion_score"] = score
            if score >= args.min_motion:
                scored.append(ep)
        print(f"  {len(scored)}/{len(episodes)} passed motion threshold")
        episodes = scored
    else:
        for ep in episodes:
            ep["motion_score"] = None

    # ── Slice for multi-GPU ────────────────────────────────────────────────────
    episodes = episodes[args.start : args.end]

    # ── Resume filter ─────────────────────────────────────────────────────────
    if args.resume:
        n_before = len(episodes)
        episodes = [ep for ep in episodes if not is_done(ep)]
        print(f"  Resume: skipping {n_before - len(episodes)} done, "
              f"{len(episodes)} remaining")

    print(f"  Will process: {len(episodes)} episodes")
    if args.dry_run:
        for ep in episodes[:20]:
            print(f"    {ep['seq_id']}  score={ep.get('motion_score','?')}")
        return

    # ── Load models once ───────────────────────────────────────────────────────
    print("\nLoading models...")
    da_model, da_transform = load_depth_anything()
    ud_model               = load_unidepth()

    # ── Process episodes ───────────────────────────────────────────────────────
    log_path = os.path.join(OUT_BASE, "batch_log.jsonl")
    n_ok = n_fail = n_skip = 0

    for i, ep in enumerate(tqdm(episodes, desc="Episodes")):
        seq_id       = ep["seq_id"]
        cam_key      = ep["cam_key"]
        # calibrated_fx: SLAM K init — see CALIB_FX comment above.
        # Falls back to 60° FoV assumption for unknown cameras.
        # All frames are resized to LONG_DIM before SLAM, so use LONG_DIM as width.
        calibrated_fx = CALIB_FX.get(cam_key, LONG_DIM * CALIB_FX_FALLBACK_RATIO)
        safe_id      = seq_id.replace("/", "_")

        # temp dirs (per-episode, cleaned after)
        frame_dir = os.path.join(WORK_DIR, safe_id, "frames")
        da_dir    = os.path.join(WORK_DIR, safe_id, "da")
        ud_dir    = os.path.join(WORK_DIR, safe_id, "ud")

        t0 = time.time()
        try:
            # 1. Extract frames
            n_frames, W, H = extract_frames_to_dir(ep, frame_dir)
            if n_frames == 0:
                raise RuntimeError("No frames extracted")

            # 2. DepthAnything
            run_da_on_dir(da_model, da_transform, frame_dir, da_dir)

            # 3. UniDepth
            run_unidepth_on_dir(ud_model, frame_dir, ud_dir, calibrated_fx)

            # 4. DROID-SLAM
            depths, motion_prob, K_opt, cam_c2w = run_droid(
                frame_dir, da_dir, ud_dir, seq_id,
                calibrated_fx, opt_intr=args.opt_intr
            )

            # 5. Save
            n_out = min(n_frames, len(depths))
            save_outputs(ep, depths, motion_prob, K_opt, cam_c2w,
                         n_out, (H, W),
                         ep.get("motion_score"), calibrated_fx)

            elapsed = time.time() - t0
            n_ok += 1
            log_entry = {"seq": seq_id, "status": "ok",
                         "n_frames": n_out, "elapsed_s": round(elapsed, 1)}
            tqdm.write(f"  ✅ {seq_id}  {n_out}f  {elapsed:.0f}s")

        except Exception as e:
            n_fail += 1
            log_entry = {"seq": seq_id, "status": "error", "error": str(e)}
            tqdm.write(f"  ❌ {seq_id}  {e}")

        finally:
            # Clean temp work dir to save disk space
            if os.path.exists(os.path.join(WORK_DIR, safe_id)):
                shutil.rmtree(os.path.join(WORK_DIR, safe_id),
                              ignore_errors=True)

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Periodic CUDA cache clear
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  ✅ Done: {n_ok}   ❌ Failed: {n_fail}   ⏭  Skipped: {n_skip}")
    print(f"  Output : {OUT_BASE}")
    print(f"  Log    : {log_path}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
