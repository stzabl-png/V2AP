"""
Unified Depth Pro evaluation — PH2D + EgoDex.

Evaluates Depth Pro focal-length estimation accuracy on three camera setups:
  1. ph2d_avp  — PH2D / Apple Vision Pro   (GT fx≈748.98 @ 1920×1090)
  2. ph2d_zed  — PH2D / ZED stereo camera   (GT fx≈730.72 @ 1280×720 proxy)
  3. egodex    — EgoDex / Apple Vision Pro  (GT fx≈736.63 @ 1920×1080, per-HDF5)

Two-pass strategy (identical to batch_depth_pro.py validated approach):
  Pass 1 — ALL episodes, MAX_FRAMES frames each → collect all per-frame fx →
            one global-median fx per camera setup
  Pass 2 — inject global-median fx, save metric depths (first 3 episodes)

Usage:
  conda activate depth-pro
  python data/eval_depth_pro_datasets.py [--quick]

  --quick  Use only 5 episodes per dataset for a fast sanity check.

Output per dataset:
  data_hub/ProcessedData/third_depth/{dataset_key}/{episode_id}/
    depths.npz    (N, H, W) float32 metric depth
    K.txt         estimated intrinsic (global-median fx)
    K_gt.txt      GT intrinsic (scaled to decoded resolution)
    frame_ids.txt
"""

import os, sys, json, argparse, numpy as np, torch, cv2
from glob import glob
from natsort import natsorted
from tqdm import tqdm

# ── project paths ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

DEPTH_PRO_DIR = os.path.join(config.PROJECT_DIR, "third_party", "ml-depth-pro", "src")
sys.path.insert(0, DEPTH_PRO_DIR)

DATA_ROOT = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")
OUT_BASE  = os.path.join(config.DATA_HUB, "ProcessedData", "third_depth")

PH2D_ROOT   = os.path.join(DATA_ROOT, "ph2d")
PH2D_META   = os.path.join(DATA_ROOT, "ph2d", "ph2d_metadata.json")
EGODEX_ROOT = os.path.join(DATA_ROOT, "egodex", "test")

MAX_FRAMES = 30   # frames per episode in Pass 1


# ══════════════════════════════════════════════════════════════════════════════
# Episode collectors
# ══════════════════════════════════════════════════════════════════════════════

def collect_ph2d(meta_json, ph2d_root):
    """Return dict {cam_key: [(task_name, hdf5_path), ...]}."""
    with open(meta_json) as f:
        meta = json.load(f)
    result = {"ph2d_avp": [], "ph2d_zed": []}
    emb_to_key = {"human_avp": "ph2d_avp", "human_zed": "ph2d_zed"}
    for task_name, attrs in meta["per_task_attributes"].items():
        emb = attrs.get("embodiment_type", "")
        key = emb_to_key.get(emb)
        if key is None:
            continue
        task_dir = os.path.join(ph2d_root, task_name)
        if not os.path.isdir(task_dir):
            continue
        hdf5s = natsorted(glob(os.path.join(task_dir, "*.hdf5")))
        for h in hdf5s:
            result[key].append((f"ph2d/{task_name}/{os.path.basename(h)}", h))
    return result


def collect_egodex(egodex_root):
    """Return list of (episode_id, mp4_path, hdf5_path) for EgoDex test set."""
    episodes = []
    for task in natsorted(os.listdir(egodex_root)):
        task_dir = os.path.join(egodex_root, task)
        if not os.path.isdir(task_dir):
            continue
        for mp4 in natsorted(glob(os.path.join(task_dir, "*.mp4"))):
            stem = os.path.splitext(os.path.basename(mp4))[0]
            hdf5 = os.path.join(task_dir, stem + ".hdf5")
            if os.path.exists(hdf5):
                ep_id = f"egodex/{task}/{stem}"
                episodes.append((ep_id, mp4, hdf5))
    return episodes


# ══════════════════════════════════════════════════════════════════════════════
# Frame loaders
# ══════════════════════════════════════════════════════════════════════════════

def frames_from_ph2d_hdf5(hdf5_path, max_frames):
    """Decode JPEG-compressed frames from PH2D observation.image.left."""
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
    """Sample frames uniformly from an EgoDex MP4."""
    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    step = max(1, total // max_frames)
    indices = list(range(0, total, step))[:max_frames]
    imgs = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            imgs.append(frame)
    cap.release()
    return imgs


def gt_fx_from_egodex_hdf5(hdf5_path):
    """Read per-episode GT intrinsic matrix from EgoDex HDF5."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        K = f["camera/intrinsic"][:]   # (3, 3) float32, at raw 1920×1080
    return float(K[0, 0]), int(K[0, 2] * 2), int(K[1, 2] * 2)  # fx, raw_w, raw_h


# ══════════════════════════════════════════════════════════════════════════════
# Depth Pro inference
# ══════════════════════════════════════════════════════════════════════════════

def run_depth_pro(model, transform, images, fixed_fx=None):
    """Run Depth Pro on list of BGR numpy images. Returns (depths, fx_list, (H,W))."""
    import depth_pro
    from PIL import Image as PILImage
    device = next(model.parameters()).device
    f_px_inject = (
        torch.tensor(fixed_fx, dtype=torch.float32).to(device)
        if fixed_fx is not None else None
    )
    depth_list, fx_list = [], []
    H_ref = W_ref = None
    for bgr in images:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_t = transform(PILImage.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.infer(img_t, f_px=f_px_inject)
        depth = pred["depth"].squeeze().cpu().float().numpy()
        depth_list.append(depth)
        fx_list.append(float(pred["focallength_px"]))
        if H_ref is None:
            H_ref, W_ref = depth.shape
    return np.stack(depth_list).astype(np.float32), fx_list, (H_ref, W_ref)


def build_K(fx, H, W):
    return np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=np.float32)


def save_results(out_dir, depths, fx_est, fx_gt, H, W, frame_ids):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "depths.npz"), depths=depths)
    np.savetxt(os.path.join(out_dir, "K.txt"),    build_K(fx_est, H, W), fmt="%.6f")
    np.savetxt(os.path.join(out_dir, "K_gt.txt"), build_K(fx_gt,  H, W), fmt="%.6f")
    with open(os.path.join(out_dir, "frame_ids.txt"), "w") as f:
        f.write("\n".join(frame_ids))


# ══════════════════════════════════════════════════════════════════════════════
# Per-dataset runner
# ══════════════════════════════════════════════════════════════════════════════

def run_dataset(label, episodes, model, transform, loader_fn,
                gt_fx_fn, gt_raw_w, gt_raw_h, out_key,
                max_episodes, pass2_save=3):
    """
    Generic two-pass runner.

    Args:
        label       : display name
        episodes    : list of (ep_id, *paths)
        loader_fn   : fn(paths, max_frames) -> list[BGR ndarray]
        gt_fx_fn    : fn(paths) -> (fx_raw, raw_w, raw_h)
                      if None, use gt_raw_w / gt_raw_h directly
        out_key     : subdirectory under OUT_BASE
    """
    ep_eval = episodes if max_episodes is None else episodes[:max_episodes]
    print(f"\n{'═'*62}")
    print(f"  Dataset : {label}")
    print(f"  Episodes: {len(ep_eval)} for Pass 1")
    print(f"{'═'*62}")

    # ── Pass 1 ──────────────────────────────────────────────────────────────
    all_fx, H, W = [], None, None
    all_gt_fx_raw, all_gt_raw_w = [], []

    for ep_tuple in tqdm(ep_eval, desc=f"Pass1/{label}"):
        ep_id   = ep_tuple[0]
        paths   = ep_tuple[1:]
        imgs = loader_fn(paths, MAX_FRAMES)
        if not imgs:
            continue

        # GT intrinsics (may vary per episode for EgoDex, fixed for PH2D)
        if gt_fx_fn is not None:
            fx_raw_ep, rw, rh = gt_fx_fn(paths)
        else:
            fx_raw_ep, rw, rh = gt_raw_w, gt_raw_h, None  # unused rh

        _, fx_list, (h, w) = run_depth_pro(model, transform, imgs)
        all_fx.extend(fx_list)
        all_gt_fx_raw.append(fx_raw_ep)
        all_gt_raw_w.append(rw)
        H, W = h, w
        tqdm.write(f"  {ep_id}: {len(imgs)}f  seq_fx={np.median(fx_list):.1f}")

    if not all_fx:
        print("  ⚠️  No frames processed — skipping")
        return None

    # ── Global median fx (Pass 1 result) ────────────────────────────────────
    global_fx = float(np.median(all_fx))
    arr = np.array(all_fx)
    print(f"\n  Pass 1: {len(arr)} total frames")
    print(f"  fx  min={arr.min():.1f}  median={global_fx:.1f}  "
          f"max={arr.max():.1f}  std={arr.std():.1f}")

    # GT fx scaled to decoded resolution
    # Use median raw_w across episodes (EgoDex is always 1920×1080)
    med_raw_w   = float(np.median(all_gt_raw_w))
    med_gt_fx   = float(np.median(all_gt_fx_raw))
    fx_gt_scaled = med_gt_fx * (W / med_raw_w)
    rel_err = (global_fx - fx_gt_scaled) / fx_gt_scaled * 100.0

    print(f"\n  Decoded resolution   : {H}×{W}")
    print(f"  GT fx (@ raw)        : {med_gt_fx:.2f} px  (raw_w={med_raw_w:.0f})")
    print(f"  GT fx (scaled)       : {fx_gt_scaled:.2f} px")
    print(f"  Global median fx     : {global_fx:.2f} px  ← Pass 2 uses this")
    print(f"  ┌─ Relative error    : {rel_err:+.2f}%  {'✅' if abs(rel_err)<3 else '⚠️'}")

    # ── Pass 2: inject global_fx, save a few episodes ───────────────────────
    save_eps = episodes[:pass2_save]
    print(f"\n  Pass 2: saving {len(save_eps)} episodes with fx={global_fx:.1f}px...")
    for ep_tuple in tqdm(save_eps, desc=f"Pass2/{label}"):
        ep_id = ep_tuple[0]
        paths = ep_tuple[1:]
        imgs  = loader_fn(paths, MAX_FRAMES)
        if not imgs:
            continue
        if gt_fx_fn is not None:
            fx_raw_ep, rw, _ = gt_fx_fn(paths)
            fx_gt_ep = fx_raw_ep * (W / rw)
        else:
            fx_gt_ep = fx_gt_scaled
        depths, _, (h, w) = run_depth_pro(model, transform, imgs, fixed_fx=global_fx)
        out_dir = os.path.join(OUT_BASE, out_key, ep_id)
        frame_ids = [f"frame_{i:04d}" for i in range(len(imgs))]
        save_results(out_dir, depths, global_fx, fx_gt_ep, h, w, frame_ids)
        tqdm.write(f"  ✅ {ep_id}  depths={depths.shape}")

    torch.cuda.empty_cache()
    return {"label": label, "fx_gt": fx_gt_scaled, "fx_est": global_fx,
            "rel_err_pct": rel_err, "n_frames": len(arr), "hw": (H, W)}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import depth_pro

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Only 5 episodes per dataset (fast sanity check)")
    args = parser.parse_args()
    max_ep = 5 if args.quick else None

    # ── collect episodes ─────────────────────────────────────────────────────
    ph2d_eps = collect_ph2d(PH2D_META, PH2D_ROOT)
    egodex_eps = collect_egodex(EGODEX_ROOT)
    print(f"PH2D  avp: {len(ph2d_eps['ph2d_avp'])} episodes")
    print(f"PH2D  zed: {len(ph2d_eps['ph2d_zed'])} episodes")
    print(f"EgoDex   : {len(egodex_eps)} episodes")

    # ── load Depth Pro ────────────────────────────────────────────────────────
    print("\nLoading Depth Pro model...")
    model, transform = depth_pro.create_model_and_transforms(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        precision=torch.float16,
    )
    model.eval()
    print(f"✅ Model on {next(model.parameters()).device}\n")

    results = []

    # ── 1. PH2D AVP ──────────────────────────────────────────────────────────
    # GT: fx=748.98, raw 1920×1090
    r = run_dataset(
        label      = "PH2D / Apple Vision Pro",
        episodes   = [(ep_id, hdf5) for ep_id, hdf5 in ph2d_eps["ph2d_avp"]],
        model=model, transform=transform,
        loader_fn  = lambda paths, n: frames_from_ph2d_hdf5(paths[0], n),
        gt_fx_fn   = None,
        gt_raw_w   = 748.9841404939381,   # (ab)using gt_raw_w as fx_raw
        gt_raw_h   = 1920,                # (ab)using gt_raw_h as raw_w
        out_key    = "ph2d",
        max_episodes = max_ep,
    )
    if r: results.append(r)

    # ── 2. PH2D ZED ──────────────────────────────────────────────────────────
    # GT: fx=730.72, raw 1280×720 (proxy from H1-ZED, same hardware)
    r = run_dataset(
        label      = "PH2D / ZED stereo camera",
        episodes   = [(ep_id, hdf5) for ep_id, hdf5 in ph2d_eps["ph2d_zed"]],
        model=model, transform=transform,
        loader_fn  = lambda paths, n: frames_from_ph2d_hdf5(paths[0], n),
        gt_fx_fn   = None,
        gt_raw_w   = 730.7215576171875,   # fx_raw
        gt_raw_h   = 1280,                # raw_w
        out_key    = "ph2d",
        max_episodes = max_ep,
    )
    if r: results.append(r)

    # ── 3. EgoDex ────────────────────────────────────────────────────────────
    # GT intrinsics read per-episode from HDF5 camera/intrinsic (fx≈736.63 @ 1920×1080)
    r = run_dataset(
        label      = "EgoDex / Apple Vision Pro",
        episodes   = egodex_eps,
        model=model, transform=transform,
        loader_fn  = lambda paths, n: frames_from_egodex_mp4(paths[0], n),
        gt_fx_fn   = lambda paths: gt_fx_from_egodex_hdf5(paths[1]),
        gt_raw_w   = None,
        gt_raw_h   = None,
        out_key    = "egodex",
        max_episodes = max_ep,
    )
    if r: results.append(r)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n\n{'═'*68}")
    print("  SUMMARY — Depth Pro focal-length error by camera setup")
    print(f"{'═'*68}")
    print(f"  {'Dataset':<32} {'GT fx':>7} {'Est fx':>7} {'Error':>7} {'Frames':>7}")
    print(f"  {'-'*66}")
    for r in results:
        flag = "✅" if abs(r["rel_err_pct"]) < 3 else "⚠️ "
        print(f"  {r['label']:<32} {r['fx_gt']:>7.1f} {r['fx_est']:>7.1f} "
              f"{r['rel_err_pct']:>+6.2f}% {r['n_frames']:>7}  {flag}")
    print(f"\n  Reference baselines:")
    print(f"    DexYCB  (RealSense D435, ~70° HFOV) : +0.20%")
    print(f"    HO3D v3 (Kinect,         ~73° HFOV) : -1.50%")
    print(f"{'═'*68}\n")


if __name__ == "__main__":
    main()
