#!/usr/bin/env python3
"""
MegaSAM full pipeline — focal length estimation via video BA.

Pipeline for each sequence (must be run from mega-sam/ directory):
  Step 1  Extract frames from PH2D HDF5 / EgoDex MP4 → PNG images
  Step 2  DepthAnything → relative disparity per frame (.npy)
  Step 3  UniDepth       → metric depth + FOV per frame (.npz)
  Step 4  DROID-SLAM + opt_intr=True → BA-optimised K matrix
                                         (saved in outputs/{seq}_droid.npz)

Output:
  <OUT_BASE>/megasam_intrinsics.json   — {seq: {fx, fy, cx, cy, rel_err_pct}}
  <OUT_BASE>/{seq}/K_megasam.txt       — optimised K
  Console summary table vs GT

Usage:
  conda activate mega_sam
  cd $PROJ/mega-sam
  python ../data/run_megasam_focal.py [--quick]

  --quick   process only 1 sequence per camera type (fast test)
"""

import os, sys, json, argparse, shutil, subprocess, glob
import numpy as np
import torch
import cv2
from tqdm import tqdm
from natsort import natsorted

# ── paths ──────────────────────────────────────────────────────────────────────
# __file__ = .../V2AP/data/run_megasam_focal.py
PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # V2AP/
MEGASAM_DIR  = os.path.join(PROJECT_DIR, "mega-sam")                         # mega-sam/
sys.path.insert(0, PROJECT_DIR)                                               # import config
sys.path.insert(0, os.path.join(MEGASAM_DIR, "UniDepth"))
sys.path.insert(0, os.path.join(MEGASAM_DIR, "base", "droid_slam"))
sys.path.insert(0, os.path.join(MEGASAM_DIR, "Depth-Anything"))

import config as project_config

DATA_ROOT   = os.path.join(project_config.DATA_HUB, "RawData", "ThirdPersonRawData")
PH2D_ROOT   = os.path.join(DATA_ROOT, "ph2d")
PH2D_META   = os.path.join(DATA_ROOT, "ph2d", "ph2d_metadata.json")
EGODEX_ROOT = os.path.join(DATA_ROOT, "egodex", "test")
OUT_BASE    = os.path.join(project_config.DATA_HUB, "ProcessedData", "megasam_intrinsics")

DA_CKPT   = os.path.join(MEGASAM_DIR, "Depth-Anything", "checkpoints", "depth_anything_vitl14.pth")
MEGA_CKPT = os.path.join(MEGASAM_DIR, "checkpoints", "megasam_final.pth")
UNIDEPTH_REVISION = "1d0d3c52f60b5164629d279bb9a7546458e6dcc4"
UNIDEPTH_HF_ID    = "lpiccinelli/unidepth-v2-vitl14"

LONG_DIM   = 640     # resize long side to this before feeding to UniDepth / DA
MAX_FRAMES = 60      # frames per sequence fed to DROID-SLAM


# ══════════════════════════════════════════════════════════════════════════════
# GT intrinsics
# ══════════════════════════════════════════════════════════════════════════════
GT_CAMERAS = {
    "ph2d_avp": {"fx_raw": 748.9841, "raw_w": 1920},
    "ph2d_zed": {"fx_raw": 730.7215, "raw_w": 1280},
    # egodex: per-episode from HDF5
}


# ══════════════════════════════════════════════════════════════════════════════
# Episode collectors
# ══════════════════════════════════════════════════════════════════════════════

def collect_ph2d_sequences(meta_json, ph2d_root, max_per_cam=1):
    with open(meta_json) as f:
        meta = json.load(f)
    result = {"ph2d_avp": [], "ph2d_zed": []}
    emb_to_key = {"human_avp": "ph2d_avp", "human_zed": "ph2d_zed"}
    for task_name, attrs in meta["per_task_attributes"].items():
        key = emb_to_key.get(attrs.get("embodiment_type", ""))
        if key is None:
            continue
        if len(result[key]) >= max_per_cam:
            continue
        task_dir = os.path.join(ph2d_root, task_name)
        hdf5s = natsorted(glob.glob(os.path.join(task_dir, "*.hdf5")))
        if hdf5s:
            result[key].append((task_name, hdf5s[0]))
    return result


def collect_egodex_sequences(egodex_root, max_seqs=1):
    seqs = []
    for task in natsorted(os.listdir(egodex_root)):
        if len(seqs) >= max_seqs:
            break
        task_dir = os.path.join(egodex_root, task)
        if not os.path.isdir(task_dir):
            continue
        mp4s = natsorted(glob.glob(os.path.join(task_dir, "*.mp4")))
        if mp4s:
            mp4 = mp4s[0]
            stem = os.path.splitext(os.path.basename(mp4))[0]
            hdf5 = os.path.join(task_dir, stem + ".hdf5")
            if os.path.exists(hdf5):
                seqs.append((f"{task}/{stem}", mp4, hdf5))
    return seqs


# ══════════════════════════════════════════════════════════════════════════════
# Frame extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames_ph2d(hdf5_path, frame_dir, max_frames=MAX_FRAMES, long_dim=LONG_DIM):
    """Extract and resize frames from PH2D HDF5 to PNG files."""
    import h5py
    os.makedirs(frame_dir, exist_ok=True)
    with h5py.File(hdf5_path, "r") as f:
        raw = f["observation.image.left"][:]
    n = min(len(raw), max_frames)
    W_out = None
    for i in range(n):
        buf = np.frombuffer(raw[i].tobytes(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w >= h:
            nw, nh = long_dim, int(round(long_dim * h / w))
        else:
            nh, nw = long_dim, int(round(long_dim * w / h))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(frame_dir, f"frame_{i:04d}.png"), img)
        W_out = nw
    print(f"  Extracted {n} frames → {nw}×{nh}  (long_dim={long_dim})")
    return n, nw, nh


def extract_frames_egodex(mp4_path, frame_dir, max_frames=MAX_FRAMES, long_dim=LONG_DIM):
    """Uniformly sample and resize frames from EgoDex MP4 to PNG files."""
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // max_frames)
    idxs  = list(range(0, total, step))[:max_frames]
    nw = nh = None
    for out_i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if w >= h:
            nw, nh = long_dim, int(round(long_dim * h / w))
        else:
            nh, nw = long_dim, int(round(long_dim * w / h))
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(frame_dir, f"frame_{out_i:04d}.png"), frame)
    cap.release()
    n = len(glob.glob(os.path.join(frame_dir, "*.png")))
    print(f"  Extracted {n} frames → {nw}×{nh}  (long_dim={long_dim})")
    return n, nw, nh


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: DepthAnything
# ══════════════════════════════════════════════════════════════════════════════

def run_depthanything(frame_dir, da_out_dir):
    """Run DepthAnything on PNG frames, output .npy disparity maps."""
    os.makedirs(da_out_dir, exist_ok=True)
    n_existing = len(glob.glob(os.path.join(da_out_dir, "*.npy")))
    n_frames   = len(glob.glob(os.path.join(frame_dir, "*.png")))
    if n_existing == n_frames and n_existing > 0:
        print(f"  DA: already done ({n_existing} files), skipping")
        return

    from depth_anything.dpt import DPT_DINOv2
    from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
    from torchvision.transforms import Compose
    import torch.nn.functional as F

    model = DPT_DINOv2(
        encoder="vitl", features=256,
        out_channels=[256, 512, 1024, 1024], localhub=False
    ).cuda()
    model.load_state_dict(torch.load(DA_CKPT, map_location="cpu", weights_only=False), strict=True)
    model.eval()

    transform = Compose([
        Resize(width=768, height=768, resize_target=False,
               keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method="upper_bound",
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        PrepareForNet(),
    ])

    imgs = natsorted(glob.glob(os.path.join(frame_dir, "*.png")))
    for p in tqdm(imgs, desc="  DepthAnything"):
        raw = cv2.imread(p)[..., :3]
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) / 255.0
        h, w = img.shape[:2]
        img_t = transform({"image": img})["image"]
        img_t = torch.from_numpy(img_t).unsqueeze(0).cuda()
        with torch.no_grad():
            disp = model(img_t)
        disp = F.interpolate(disp[None], (h, w), mode="bilinear", align_corners=False)[0,0]
        np.save(os.path.join(da_out_dir, os.path.basename(p)[:-4] + ".npy"),
                disp.cpu().float().numpy())

    del model
    torch.cuda.empty_cache()
    print(f"  DA done: {len(imgs)} frames → {da_out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: UniDepth
# ══════════════════════════════════════════════════════════════════════════════

def run_unidepth(frame_dir, ud_out_dir):
    """Run UniDepthV2 on PNG frames, output .npz with depth+fov."""
    os.makedirs(ud_out_dir, exist_ok=True)
    n_existing = len(glob.glob(os.path.join(ud_out_dir, "*.npz")))
    n_frames   = len(glob.glob(os.path.join(frame_dir, "*.png")))
    if n_existing == n_frames and n_existing > 0:
        print(f"  UniDepth: already done ({n_existing} files), skipping")
        return

    from unidepth.models import UniDepthV2
    model = UniDepthV2.from_pretrained(UNIDEPTH_HF_ID, revision=UNIDEPTH_REVISION)
    model = model.to("cuda").eval()

    imgs = natsorted(glob.glob(os.path.join(frame_dir, "*.png")))
    for p in tqdm(imgs, desc="  UniDepth"):
        rgb = np.array(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
        rgb_t = torch.from_numpy(rgb).permute(2,0,1).cuda()
        with torch.no_grad():
            pred = model.infer(rgb_t)
        depth = pred["depth"][0,0].cpu().float().numpy()
        K = pred["intrinsics"][0].cpu().numpy()
        fov = float(np.degrees(2 * np.arctan(depth.shape[1] / (2 * K[0,0]))))
        np.savez(os.path.join(ud_out_dir, os.path.basename(p)[:-4] + ".npz"),
                 depth=np.float32(depth), fov=fov)

    del model
    torch.cuda.empty_cache()
    print(f"  UniDepth done: {len(imgs)} frames → {ud_out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: DROID-SLAM + focal length BA
# ══════════════════════════════════════════════════════════════════════════════

def run_megasam_tracking(frame_dir, da_dir, ud_dir, seq_name, out_npz):
    """Camera tracking with BA focal length optimisation → saves *_droid.npz."""
    if os.path.exists(out_npz):
        print(f"  Tracking: already done → {out_npz}")
        return True

    # MegaSAM test_demo.py must be run from mega-sam/ dir
    cmd = [
        sys.executable,
        os.path.join(MEGASAM_DIR, "camera_tracking_scripts", "test_demo.py"),
        f"--datapath={frame_dir}",
        f"--weights={MEGA_CKPT}",
        f"--scene_name={seq_name}",
        f"--mono_depth_path={os.path.dirname(da_dir)}",   # parent dir of scene
        f"--metric_depth_path={os.path.dirname(ud_dir)}",
        "--disable_vis",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{MEGASAM_DIR}/UniDepth:"
        f"{MEGASAM_DIR}/base/droid_slam:"
        f"{MEGASAM_DIR}/Depth-Anything:"
        + env.get("PYTHONPATH", "")
    )
    print(f"  Running DROID-SLAM tracking for '{seq_name}'...")
    ret = subprocess.run(cmd, cwd=MEGASAM_DIR, env=env, capture_output=False)
    if ret.returncode != 0:
        print(f"  ⚠️  Tracking failed (exit {ret.returncode})")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Read optimised K from droid.npz
# ══════════════════════════════════════════════════════════════════════════════

def read_optimised_K(seq_name):
    """Read intrinsic from reconstructions/<seq>/intrinsics.npy (BA-optimised)."""
    # MegaSAM saves optimised intrinsics in two places:
    # 1. reconstructions/{seq}/intrinsics.npy  shape (N,4) [fx,fy,cx,cy] per frame
    # 2. outputs/{seq}_droid.npz               field 'intrinsic' (3,3) = first-frame K
    recon_path = os.path.join(MEGASAM_DIR, "reconstructions", seq_name, "intrinsics.npy")
    npz_path   = os.path.join(MEGASAM_DIR, "outputs", f"{seq_name}_droid.npz")

    if os.path.exists(recon_path):
        intr = np.load(recon_path)   # (N, 4)
        fx_arr = intr[:, 0]
        fx_opt = float(np.median(fx_arr))
        print(f"  Optimised fx per-frame: min={fx_arr.min():.1f} "
              f"median={fx_opt:.1f} max={fx_arr.max():.1f}")
        K = np.eye(3)
        K[0,0] = fx_opt;  K[1,1] = intr[0,1]
        K[0,2] = intr[0,2]; K[1,2] = intr[0,3]
        return fx_opt, K
    elif os.path.exists(npz_path):
        data = np.load(npz_path)
        K = data["intrinsic"]
        fx_opt = float(K[0,0])
        print(f"  Optimised K from npz: fx={fx_opt:.1f}")
        return fx_opt, K
    else:
        print(f"  ⚠️  No output found at {recon_path} or {npz_path}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def process_sequence(cam_key, seq_id, frame_extractor_fn, gt_fx_raw, gt_raw_w,
                     decoded_w, results_list):
    """Run full MegaSAM pipeline for one sequence."""
    safe_id  = seq_id.replace("/", "_")
    work_dir = os.path.join(OUT_BASE, "workdir", safe_id)

    frame_dir = os.path.join(work_dir, "frames")
    da_dir    = os.path.join(work_dir, "da_disp", safe_id)
    ud_dir    = os.path.join(work_dir, "ud_depth", safe_id)
    out_npz   = os.path.join(MEGASAM_DIR, "outputs", f"{safe_id}_droid.npz")

    print(f"\n{'─'*60}")
    print(f"  Seq: {seq_id}  |  cam: {cam_key}")

    # Step 1: extract frames
    n, W, H = frame_extractor_fn(frame_dir)
    if n == 0:
        print("  ⚠️  No frames extracted, skipping")
        return

    # Step 2: DepthAnything
    run_depthanything(frame_dir, da_dir)

    # Step 3: UniDepth
    run_unidepth(frame_dir, ud_dir)

    # Step 4: DROID-SLAM tracking
    ok = run_megasam_tracking(frame_dir, da_dir, ud_dir, safe_id, out_npz)
    if not ok:
        return

    # Read optimised fx
    fx_opt, K_opt = read_optimised_K(safe_id)
    if fx_opt is None:
        return

    # Scale GT to decoded resolution
    if gt_fx_raw is not None:
        fx_gt_sc = gt_fx_raw * (W / gt_raw_w)
    else:
        fx_gt_sc = None   # EgoDex per-episode GT already at 1920px

    rel_err = None
    if fx_gt_sc is not None:
        rel_err = (fx_opt - fx_gt_sc) / fx_gt_sc * 100.0
        flag = "✅" if abs(rel_err) < 5 else "⚠️ "
        print(f"\n  GT fx (scaled)  : {fx_gt_sc:.2f} px")
        print(f"  MegaSAM fx opt  : {fx_opt:.2f} px")
        print(f"  Relative error  : {rel_err:+.2f}%  {flag}")

    # Save K
    out_dir = os.path.join(OUT_BASE, safe_id)
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "K_megasam.txt"), K_opt, fmt="%.6f")

    results_list.append({
        "seq": seq_id, "cam": cam_key,
        "fx_gt": fx_gt_sc, "fx_opt": fx_opt,
        "rel_err_pct": rel_err, "hw": (H, W),
    })


def screen_and_select(dataset, top_n):
    """Run camera motion screening; return top-N most-moving episodes."""
    sys.path.insert(0, os.path.join(PROJECT_DIR, "data"))
    from screen_camera_motion import scan_ph2d_avp, scan_egodex, CAM_MOTION_THRESH
    if dataset == "ph2d_avp":
        print("  Screening PH2D AVP for camera motion...")
        results = scan_ph2d_avp()
    else:
        print("  Screening EgoDex for camera motion...")
        results = scan_egodex()
    good = [r for r in results if r["score"] >= CAM_MOTION_THRESH]
    print(f"  {len(good)} GOOD / {len(results)} total  (threshold={CAM_MOTION_THRESH}px)")
    top = good[:top_n]
    for r in top:
        parts = r["path"].split("/")
        print(f"    score={r['score']:.1f}  {parts[-2]}/{parts[-1]}")
    return top


def compute_global_median(per_seq_results, cam_key):
    """Return median/std of BA-estimated fx with outlier filtering.

    Rejects:
      1. fx <= 0  → SLAM diverged (degenerate trajectory)
      2. Values outside Q25-3*IQR ~ Q75+3*IQR  → gross outliers
    """
    all_items = [(r["fx_opt"], r["seq"]) for r in per_seq_results
                 if r["cam"] == cam_key and r["fx_opt"] is not None]
    if not all_items:
        return None

    # Step 1: remove physically impossible values (SLAM diverged)
    valid  = [(fx, s) for fx, s in all_items if fx > 0]
    reject = [(fx, s) for fx, s in all_items if fx <= 0]
    if reject:
        print(f"  ⚠️  {cam_key}: {len(reject)} SLAM-diverged removed (fx≤0):")
        for fx, s in reject:
            print(f"       fx={fx:.1f}  {s}")
    if not valid:
        return None

    fx_arr = np.array([fx for fx, _ in valid])

    # Step 2: IQR-based outlier removal
    q25, q75 = np.percentile(fx_arr, 25), np.percentile(fx_arr, 75)
    iqr  = max(q75 - q25, 1.0)   # avoid division by zero
    lo, hi = q25 - 3 * iqr, q75 + 3 * iqr
    mask = (fx_arr >= lo) & (fx_arr <= hi)
    if not mask.all():
        removed = [(fx, s) for (fx, s), m in zip(valid, mask) if not m]
        print(f"  ⚠️  {cam_key}: {len(removed)} IQR-outliers removed:")
        for fx, s in removed:
            print(f"       fx={fx:.1f}  {s}")
    fx_clean = fx_arr[mask]

    return {
        "median_fx":  float(np.median(fx_clean)),
        "mean_fx":    float(np.mean(fx_clean)),
        "std_fx":     float(np.std(fx_clean)),
        "n_seqs":     int(mask.sum()),
        "n_total":    len(all_items),
        "n_rejected": len(all_items) - int(mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="MegaSAM focal length calibration using top-N highest-motion episodes."
    )
    parser.add_argument("--quick", action="store_true",
                        help="1 seq per dataset (fast sanity check)")
    parser.add_argument("--top", type=int, default=10,
                        help="Top-N highest-motion episodes per dataset (default: 10)")
    parser.add_argument("--skip-screen", action="store_true",
                        help="Skip motion screening, use first N in metadata order")
    args = parser.parse_args()
    top_n = 1 if args.quick else args.top

    os.makedirs(OUT_BASE, exist_ok=True)
    results = []
    import h5py

    # ── PH2D AVP ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*62}\n  PH2D AVP — selecting top {top_n} sequences\n{'═'*62}")

    if args.skip_screen:
        with open(PH2D_META) as f:
            meta = json.load(f)
        avp_tasks = [(t, a) for t, a in meta["per_task_attributes"].items()
                     if a.get("embodiment_type") == "human_avp"]
        ph2d_cands = []
        for task_name, _ in avp_tasks[:top_n]:
            task_dir = os.path.join(PH2D_ROOT, task_name)
            hdf5s = natsorted(glob.glob(os.path.join(task_dir, "*.hdf5")))
            if hdf5s:
                ph2d_cands.append({"path": hdf5s[0], "task": task_name})
    else:
        ph2d_cands = screen_and_select("ph2d_avp", top_n)

    for cand in ph2d_cands:
        hdf5_path = cand["path"]
        task_name = cand.get("task", hdf5_path.split("/")[-2])
        ep_name   = os.path.splitext(os.path.basename(hdf5_path))[0]
        seq_id    = f"ph2d/avp/{task_name}/{ep_name}"

        def _extractor(frame_dir, hp=hdf5_path):
            return extract_frames_ph2d(hp, frame_dir)

        process_sequence("ph2d_avp", seq_id, _extractor,
                         GT_CAMERAS["ph2d_avp"]["fx_raw"],
                         GT_CAMERAS["ph2d_avp"]["raw_w"],
                         decoded_w=LONG_DIM, results_list=results)

    # ── EgoDex ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*62}\n  EgoDex — selecting top {top_n} sequences\n{'═'*62}")

    if args.skip_screen:
        raw_seqs = collect_egodex_sequences(EGODEX_ROOT, max_seqs=top_n)
        egodex_cands = [{"path": mp4, "hdf5": hdf5, "seq_id": seq_id}
                        for seq_id, mp4, hdf5 in raw_seqs]
    else:
        raw_cands = screen_and_select("egodex", top_n)
        egodex_cands = []
        for cand in raw_cands:
            mp4  = cand["path"]
            stem = os.path.splitext(os.path.basename(mp4))[0]
            task = mp4.split("/")[-2]
            hdf5 = os.path.join(os.path.dirname(mp4), stem + ".hdf5")
            if os.path.exists(hdf5):
                egodex_cands.append({"path": mp4, "hdf5": hdf5,
                                     "seq_id": f"{task}/{stem}"})

    for cand in egodex_cands:
        mp4  = cand["path"];  hdf5 = cand["hdf5"];  seq_id = cand["seq_id"]
        with h5py.File(hdf5, "r") as fh:
            K_raw = fh["camera/intrinsic"][:]
        gt_fx_raw_eg = float(K_raw[0, 0])
        gt_raw_w_eg  = int(K_raw[0, 2] * 2)

        def _extractor(frame_dir, mp=mp4):
            return extract_frames_egodex(mp, frame_dir)

        process_sequence("egodex", seq_id, _extractor,
                         gt_fx_raw_eg, gt_raw_w_eg,
                         decoded_w=LONG_DIM, results_list=results)

    # ── Global medians ────────────────────────────────────────────────────────
    ph2d_stats   = compute_global_median(results, "ph2d_avp")
    egodex_stats = compute_global_median(results, "egodex")

    print(f"\n\n{'═'*72}")
    print("  PER-SEQUENCE results")
    print(f"{'═'*72}")
    print(f"  {'Camera':<12} {'Seq':<32} {'GT fx':>7} {'BA fx':>7} {'Error':>8}")
    print(f"  {'-'*70}")
    for r in results:
        if r["fx_gt"] is not None and r["rel_err_pct"] is not None:
            flag = "✅" if abs(r["rel_err_pct"]) < 5 else \
                   ("🟡" if abs(r["rel_err_pct"]) < 15 else "⚠️ ")
            print(f"  {r['cam']:<12} {r['seq'][:32]:<32} {r['fx_gt']:>7.1f} "
                  f"{r['fx_opt']:>7.1f} {r['rel_err_pct']:>+7.2f}%  {flag}")
        else:
            print(f"  {r['cam']:<12} {r['seq'][:32]:<32}    N/A  "
                  f"{r['fx_opt']:>7.1f}")

    print(f"\n{'═'*72}")
    print("  GLOBAL CALIBRATION — median across all sequences  (@640px width)")
    print(f"{'═'*72}")

    calibration = {}

    if ph2d_stats:
        fx_gt = GT_CAMERAS["ph2d_avp"]["fx_raw"] * (LONG_DIM / GT_CAMERAS["ph2d_avp"]["raw_w"])
        err   = (ph2d_stats["median_fx"] - fx_gt) / fx_gt * 100
        flag  = "✅" if abs(err) < 5 else ("🟡" if abs(err) < 15 else "⚠️ ")
        print(f"\n  PH2D AVP:")
        print(f"    GT fx           : {fx_gt:.2f} px")
        n_ok  = ph2d_stats.get('n_seqs', '?')
        n_rej = ph2d_stats.get('n_rejected', 0)
        print(f"    Median BA fx    : {ph2d_stats['median_fx']:.2f} px  {err:+.2f}%  {flag}")
        print(f"    Std across seqs : ±{ph2d_stats['std_fx']:.2f} px  ({n_ok} valid / {ph2d_stats.get('n_total','?')} total, {n_rej} rejected)")
        calibration["ph2d_avp"] = {
            "fx_global": ph2d_stats["median_fx"], "decoded_w": LONG_DIM,
            "n_seqs": ph2d_stats["n_seqs"], "std": ph2d_stats["std_fx"],
            "gt_fx": fx_gt, "gt_err_pct": err,
        }

    if egodex_stats:
        gt_vals  = [r["fx_gt"] for r in results
                    if r["cam"] == "egodex" and r["fx_gt"] is not None]
        fx_gt_eg = float(np.median(gt_vals)) if gt_vals else None
        err      = ((egodex_stats["median_fx"] - fx_gt_eg) / fx_gt_eg * 100
                    if fx_gt_eg else None)
        flag     = "✅" if err and abs(err) < 5 else ("🟡" if err and abs(err) < 15 else "⚠️ ")
        print(f"\n  EgoDex:")
        print(f"    GT fx (median)  : {fx_gt_eg:.2f} px")
        n_ok  = egodex_stats.get('n_seqs', '?')
        n_rej = egodex_stats.get('n_rejected', 0)
        print(f"    Median BA fx    : {egodex_stats['median_fx']:.2f} px  {err:+.2f}%  {flag}")
        print(f"    Std across seqs : ±{egodex_stats['std_fx']:.2f} px  ({n_ok} valid / {egodex_stats.get('n_total','?')} total, {n_rej} rejected)")
        calibration["egodex"] = {
            "fx_global": egodex_stats["median_fx"], "decoded_w": LONG_DIM,
            "n_seqs": egodex_stats["n_seqs"], "std": egodex_stats["std_fx"],
            "gt_fx": fx_gt_eg, "gt_err_pct": err,
        }

    print(f"\n{'═'*72}")
    print("  READY-TO-USE CONSTANTS")
    print(f"{'═'*72}")
    if "ph2d_avp" in calibration:
        print(f"  PH2D_AVP_FX  = {calibration['ph2d_avp']['fx_global']:.1f}  # px @640px wide")
    if "egodex" in calibration:
        print(f"  EGODEX_FX    = {calibration['egodex']['fx_global']:.1f}  # px @640px wide")
    print()

    out_json = os.path.join(OUT_BASE, "megasam_calibration.json")
    with open(out_json, "w") as f:
        json.dump({"calibration": calibration, "per_sequence": results},
                  f, indent=2, default=str)
    print(f"  Saved → {out_json}")


if __name__ == "__main__":
    main()

