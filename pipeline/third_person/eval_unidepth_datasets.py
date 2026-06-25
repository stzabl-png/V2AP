"""
UniDepth evaluation on PH2D + EgoDex — focal length estimation accuracy.

UniDepthV2 simultaneously outputs:
  - metric depth map  (H, W)
  - camera intrinsic K  (3, 3)  <-- self-estimated, no GT needed

Evaluation:
  Pass 1 — run on ALL episodes, MAX_FRAMES frames each
           → collect per-frame fx estimates (from K[0,0])
           → global median fx per camera setup
  Compare median fx vs GT fx → relative error

Usage:
  conda activate mega_sam
  cd $PROJ
  python data/eval_unidepth_datasets.py [--quick]

  --quick   5 episodes per dataset (fast sanity check)
"""

import os, sys, json, argparse, numpy as np, torch, cv2
from glob import glob
from natsort import natsorted
from tqdm import tqdm

# ── project paths ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# UniDepth lives inside mega-sam submodule
UNIDEPTH_DIR = os.path.join(config.PROJECT_DIR, "mega-sam", "UniDepth")
sys.path.insert(0, UNIDEPTH_DIR)

DATA_ROOT = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")
OUT_BASE  = os.path.join(config.DATA_HUB, "ProcessedData", "third_depth_unidepth")

PH2D_ROOT   = os.path.join(DATA_ROOT, "ph2d")
PH2D_META   = os.path.join(DATA_ROOT, "ph2d", "ph2d_metadata.json")
EGODEX_ROOT = os.path.join(DATA_ROOT, "egodex", "test")

UNIDEPTH_REVISION = "1d0d3c52f60b5164629d279bb9a7546458e6dcc4"
UNIDEPTH_HF_ID    = "lpiccinelli/unidepth-v2-vitl14"

MAX_FRAMES = 30   # frames per episode


# ══════════════════════════════════════════════════════════════════════════════
# Episode collectors (same as eval_depth_pro_datasets.py)
# ══════════════════════════════════════════════════════════════════════════════

def collect_ph2d(meta_json, ph2d_root):
    with open(meta_json) as f:
        meta = json.load(f)
    result = {"ph2d_avp": [], "ph2d_zed": []}
    emb_to_key = {"human_avp": "ph2d_avp", "human_zed": "ph2d_zed"}
    for task_name, attrs in meta["per_task_attributes"].items():
        key = emb_to_key.get(attrs.get("embodiment_type", ""))
        if key is None:
            continue
        task_dir = os.path.join(ph2d_root, task_name)
        if not os.path.isdir(task_dir):
            continue
        for h in natsorted(glob(os.path.join(task_dir, "*.hdf5"))):
            result[key].append((f"ph2d/{task_name}/{os.path.basename(h)}", h))
    return result


def collect_egodex(egodex_root):
    episodes = []
    for task in natsorted(os.listdir(egodex_root)):
        task_dir = os.path.join(egodex_root, task)
        if not os.path.isdir(task_dir):
            continue
        for mp4 in natsorted(glob(os.path.join(task_dir, "*.mp4"))):
            stem = os.path.splitext(os.path.basename(mp4))[0]
            hdf5 = os.path.join(task_dir, stem + ".hdf5")
            if os.path.exists(hdf5):
                episodes.append((f"egodex/{task}/{stem}", mp4, hdf5))
    return episodes


# ══════════════════════════════════════════════════════════════════════════════
# Frame loaders
# ══════════════════════════════════════════════════════════════════════════════

def frames_from_ph2d_hdf5(hdf5_path, max_frames):
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        raw = f["observation.image.left"][:]
    imgs = []
    for i in range(min(len(raw), max_frames)):
        buf = np.frombuffer(raw[i].tobytes(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)
    return imgs


def frames_from_egodex_mp4(mp4_path, max_frames):
    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    step = max(1, total // max_frames)
    imgs = []
    for idx in range(0, total, step)[:max_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            imgs.append(frame)
    cap.release()
    return imgs


def gt_fx_from_egodex_hdf5(hdf5_path):
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        K = f["camera/intrinsic"][:]
    return float(K[0, 0]), int(K[0, 2] * 2), int(K[1, 2] * 2)


# ══════════════════════════════════════════════════════════════════════════════
# UniDepth inference
# ══════════════════════════════════════════════════════════════════════════════

def run_unidepth(model, images):
    """Run UniDepthV2 on a list of BGR numpy images.

    Returns:
        depths  list of (H,W) float32 arrays  — metric depth in metres
        fx_list list of float  — estimated fx (K[0,0]) per frame
        hw      (H, W) of output depth
    """
    device = next(model.parameters()).device
    depth_list, fx_list = [], []
    H_ref = W_ref = None

    for bgr in images:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # UniDepth expects uint8 tensor (3, H, W)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).to(device)

        with torch.no_grad():
            pred = model.infer(rgb_t)

        depth = pred["depth"][0, 0].cpu().float().numpy()   # (H, W)
        K     = pred["intrinsics"][0].cpu().numpy()          # (3, 3)
        fx    = float(K[0, 0])

        depth_list.append(depth)
        fx_list.append(fx)
        if H_ref is None:
            H_ref, W_ref = depth.shape

    return depth_list, fx_list, (H_ref, W_ref)


# ══════════════════════════════════════════════════════════════════════════════
# Generic two-pass runner
# ══════════════════════════════════════════════════════════════════════════════

def run_dataset(label, episodes, model,
                loader_fn, gt_fx_raw, gt_raw_w,
                out_key, max_episodes, pass2_save=3):
    """
    global-median two-pass evaluation with UniDepth.

    gt_fx_raw, gt_raw_w: ground-truth at raw resolution.
                         Pass None,None to use per-episode GT (EgoDex).
    """
    ep_eval = episodes if max_episodes is None else episodes[:max_episodes]
    print(f"\n{'═'*62}")
    print(f"  Dataset : {label}")
    print(f"  Episodes for Pass 1: {len(ep_eval)}")
    print(f"{'═'*62}")

    all_fx, all_gt_fx_raw, all_gt_raw_w = [], [], []
    H = W = None

    for ep_tuple in tqdm(ep_eval, desc=f"UniDepth/{label}"):
        ep_id = ep_tuple[0]
        paths = ep_tuple[1:]

        imgs = loader_fn(paths)
        if not imgs:
            continue

        # GT intrinsics (fixed or per-episode)
        if gt_fx_raw is not None:
            _fx_raw, _rw = gt_fx_raw, gt_raw_w
        else:
            _fx_raw, _rw, _ = gt_fx_from_egodex_hdf5(paths[1])

        _, fx_list, (h, w) = run_unidepth(model, imgs)
        all_fx.extend(fx_list)
        all_gt_fx_raw.append(_fx_raw)
        all_gt_raw_w.append(_rw)
        H, W = h, w
        tqdm.write(f"  {ep_id}: {len(imgs)}f  seq_fx={np.median(fx_list):.1f}")

    if not all_fx:
        print("  ⚠️  No frames processed")
        return None

    # ── Global median ─────────────────────────────────────────────────────────
    global_fx = float(np.median(all_fx))
    arr = np.array(all_fx)
    print(f"\n  Pass 1: {len(arr)} total frames")
    print(f"  fx  min={arr.min():.1f}  median={global_fx:.1f}  "
          f"max={arr.max():.1f}  std={arr.std():.1f}")

    # GT scaled to decoded resolution W
    med_gt_fx  = float(np.median(all_gt_fx_raw))
    med_raw_w  = float(np.median(all_gt_raw_w))
    fx_gt_sc   = med_gt_fx * (W / med_raw_w)
    rel_err    = (global_fx - fx_gt_sc) / fx_gt_sc * 100.0

    print(f"\n  Decoded  : {H}×{W}")
    print(f"  GT fx (raw={med_gt_fx:.1f} @ {med_raw_w:.0f}px → scaled): {fx_gt_sc:.2f} px")
    print(f"  UniDepth median fx : {global_fx:.2f} px")
    flag = "✅" if abs(rel_err) < 5 else "⚠️ "
    print(f"  Relative error     : {rel_err:+.2f}%  {flag}")

    # ── Pass 2: save a few episodes' depths ───────────────────────────────────
    save_eps = episodes[:pass2_save]
    print(f"\n  Saving {len(save_eps)} episode depth maps...")
    for ep_tuple in tqdm(save_eps, desc=f"Save/{label}"):
        ep_id = ep_tuple[0]
        paths = ep_tuple[1:]
        imgs  = loader_fn(paths)
        if not imgs:
            continue
        depth_list, _, _ = run_unidepth(model, imgs)
        depths = np.stack(depth_list).astype(np.float32)
        out_dir = os.path.join(OUT_BASE, out_key, ep_id)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "depths.npz"), depths=depths)
        # Save estimated K (using global median fx)
        K_est = np.array([[global_fx, 0, W/2], [0, global_fx, H/2], [0,0,1]], dtype=np.float32)
        np.savetxt(os.path.join(out_dir, "K.txt"), K_est, fmt="%.6f")
        # GT K at decoded resolution
        if gt_fx_raw is not None:
            _fx_gt_ep = gt_fx_raw * (W / gt_raw_w)
        else:
            _fx_raw_ep, _rw_ep, _ = gt_fx_from_egodex_hdf5(paths[1])
            _fx_gt_ep = _fx_raw_ep * (W / _rw_ep)
        K_gt = np.array([[_fx_gt_ep,0,W/2],[0,_fx_gt_ep,H/2],[0,0,1]], dtype=np.float32)
        np.savetxt(os.path.join(out_dir, "K_gt.txt"), K_gt, fmt="%.6f")
        tqdm.write(f"  ✅ {ep_id}  depths={depths.shape}")

    torch.cuda.empty_cache()
    return {"label": label, "fx_gt": fx_gt_sc, "fx_est": global_fx,
            "rel_err_pct": rel_err, "n_frames": len(arr), "hw": (H, W)}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    from unidepth.models import UniDepthV2

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="5 episodes per dataset for quick test")
    args  = parser.parse_args()
    max_ep = 5 if args.quick else None

    # ── collect episodes ──────────────────────────────────────────────────────
    ph2d_eps   = collect_ph2d(PH2D_META, PH2D_ROOT)
    egodex_eps = collect_egodex(EGODEX_ROOT)
    print(f"PH2D  avp : {len(ph2d_eps['ph2d_avp'])} episodes")
    print(f"PH2D  zed : {len(ph2d_eps['ph2d_zed'])} episodes")
    print(f"EgoDex    : {len(egodex_eps)} episodes")

    # ── load UniDepthV2 ───────────────────────────────────────────────────────
    print("\nLoading UniDepthV2...")
    model = UniDepthV2.from_pretrained(UNIDEPTH_HF_ID, revision=UNIDEPTH_REVISION)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    model.resolution_level = 5   # 0=fastest, 10=highest quality
    print(f"✅ UniDepthV2 loaded on {next(model.parameters()).device}\n")

    results = []

    # ── 1. PH2D AVP ──────────────────────────────────────────────────────────
    r = run_dataset(
        label      = "PH2D / Apple Vision Pro",
        episodes   = [(ep_id, hdf5) for ep_id, hdf5 in ph2d_eps["ph2d_avp"]],
        model      = model,
        loader_fn  = lambda paths: frames_from_ph2d_hdf5(paths[0], MAX_FRAMES),
        gt_fx_raw  = 748.9841404939381,
        gt_raw_w   = 1920,
        out_key    = "ph2d",
        max_episodes = max_ep,
    )
    if r: results.append(r)

    # ── 2. PH2D ZED ──────────────────────────────────────────────────────────
    r = run_dataset(
        label      = "PH2D / ZED stereo camera",
        episodes   = [(ep_id, hdf5) for ep_id, hdf5 in ph2d_eps["ph2d_zed"]],
        model      = model,
        loader_fn  = lambda paths: frames_from_ph2d_hdf5(paths[0], MAX_FRAMES),
        gt_fx_raw  = 730.7215576171875,
        gt_raw_w   = 1280,
        out_key    = "ph2d",
        max_episodes = max_ep,
    )
    if r: results.append(r)

    # ── 3. EgoDex ────────────────────────────────────────────────────────────
    r = run_dataset(
        label      = "EgoDex / Apple Vision Pro",
        episodes   = egodex_eps,
        model      = model,
        loader_fn  = lambda paths: frames_from_egodex_mp4(paths[0], MAX_FRAMES),
        gt_fx_raw  = None,   # read per-episode from HDF5
        gt_raw_w   = None,
        out_key    = "egodex",
        max_episodes = max_ep,
    )
    if r: results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'═'*68}")
    print("  SUMMARY — UniDepthV2 focal-length estimation vs GT")
    print(f"{'═'*68}")
    print(f"  {'Dataset':<32} {'GT fx':>7} {'Est fx':>7} {'Error':>7} {'Frames':>7}")
    print(f"  {'-'*66}")
    for r in results:
        flag = "✅" if abs(r["rel_err_pct"]) < 5 else "⚠️ "
        print(f"  {r['label']:<32} {r['fx_gt']:>7.1f} {r['fx_est']:>7.1f} "
              f"{r['rel_err_pct']:>+6.2f}% {r['n_frames']:>7}  {flag}")
    print(f"\n  Reference (Depth Pro baselines):")
    print(f"    DexYCB  (RealSense, ~70° HFOV) :  +0.20%  ✅")
    print(f"    HO3D v3 (Kinect,    ~73° HFOV) :  -1.50%  ✅")
    print(f"  Depth Pro on these cameras:")
    print(f"    PH2D AVP  (~100° HFOV)         : +118.69% ⚠️")
    print(f"    PH2D ZED  ( ~82° HFOV)         :  +64.34% ⚠️")
    print(f"    EgoDex AVP (~100° HFOV)        : +176.23% ⚠️")
    print(f"{'═'*68}\n")


if __name__ == "__main__":
    main()
