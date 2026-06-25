"""
Center-crop test: Does cropping wide-angle images to ~70° FOV
fix Depth Pro's focal length estimation?

Approach:
  1. Center-crop image to CROP_RATIO of original width
     → effective FOV drops from ~100° to ~70°
  2. Resize back to original width → Depth Pro sees normal-FOV image
  3. Depth Pro estimates fx_dp (at crop resolution)
  4. Back-calculate: fx_orig = fx_dp * CROP_RATIO
  5. Compare fx_orig vs GT

Why this works:
  - HFOV ≈ 70° ← within Depth Pro's training distribution (HO3D=73°, DexYCB=56°)
  - Hands/objects are usually in frame center → kept after crop
  - Downstream (MANO, HaWoR, FP) uses ORIGINAL video - no data lost

Usage:
  conda activate depth-pro
  cd $PROJ
  python data/eval_crop_depth_pro.py [--quick]
"""

import os, sys, json, argparse, numpy as np, torch, cv2
from glob import glob
from natsort import natsorted
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

DEPTH_PRO_DIR = os.path.join(config.PROJECT_DIR, "third_party", "ml-depth-pro", "src")
sys.path.insert(0, DEPTH_PRO_DIR)

DATA_ROOT   = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")
PH2D_ROOT   = os.path.join(DATA_ROOT, "ph2d")
PH2D_META   = os.path.join(DATA_ROOT, "ph2d", "ph2d_metadata.json")
EGODEX_ROOT = os.path.join(DATA_ROOT, "egodex", "test")

MAX_FRAMES = 30

# ── crop ratios to test ────────────────────────────────────────────────────────
# For AVP at ~104° HFOV:
#   crop 55% → 70° (like HO3D)
#   crop 65% → 78°
#   crop 75% → 84°
CROP_RATIOS = [0.55, 0.65, 0.75]

# GT intrinsics
GT_AVP_FX_RAW = 748.9841   # at 1920px raw
GT_AVP_RAW_W  = 1920


def center_crop_and_resize(img, crop_ratio):
    """Center-crop to crop_ratio of width, resize back to original size."""
    H, W = img.shape[:2]
    crop_w = int(W * crop_ratio)
    x0 = (W - crop_w) // 2
    cropped = img[:, x0 : x0 + crop_w]
    resized  = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)
    return resized, crop_w


def load_ph2d_frames(hdf5_path, max_frames):
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


def load_egodex_frames(mp4_path, max_frames, long_dim=640):
    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // max_frames)
    imgs  = []
    for idx in range(0, total, step)[:max_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        scale = long_dim / max(h, w)
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)), cv2.INTER_AREA)
        imgs.append(frame)
    cap.release()
    return imgs


def run_depth_pro_on_images(model, transform, images, fixed_fx=None):
    from PIL import Image as PILImage
    device = next(model.parameters()).device
    f_px_inject = (torch.tensor(fixed_fx, dtype=torch.float32).to(device)
                   if fixed_fx is not None else None)
    fx_list, H_ref, W_ref = [], None, None
    for bgr in images:
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_t = transform(PILImage.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.infer(img_t, f_px=f_px_inject)
        fx_list.append(float(pred["focallength_px"]))
        if H_ref is None:
            H_ref, W_ref = pred["depth"].shape[-2:]
    return fx_list, H_ref, W_ref


def evaluate_crop(label, images, model, transform,
                  fx_gt_raw, raw_w, results):
    """Test multiple crop ratios on one set of images."""
    if not images:
        return
    H, W = images[0].shape[:2]
    fx_gt_decoded = fx_gt_raw * (W / raw_w)   # GT at decoded resolution

    print(f"\n  {label}  (decoded {H}×{W}, GT fx={fx_gt_decoded:.1f})")
    print(f"  {'Crop':>6} {'HFOV':>6} {'fx_dp':>7} {'fx_orig':>8} {'Error':>8}")
    print(f"  {'-'*46}")

    for ratio in CROP_RATIOS:
        # Apply crop + resize to all frames
        cropped_imgs = []
        crop_w_actual = None
        for img in images:
            c, cw = center_crop_and_resize(img, ratio)
            cropped_imgs.append(c)
            crop_w_actual = cw

        # Run Depth Pro (global median across all frames)
        fx_dp_list, _, _ = run_depth_pro_on_images(model, transform, cropped_imgs)
        fx_dp_median = float(np.median(fx_dp_list))

        # Back-calculate original fx
        fx_orig = fx_dp_median * ratio        # fx in decoded-image space

        # HFOV of the cropped+resized image seen by Depth Pro
        hfov = float(np.degrees(2 * np.arctan(W / (2 * fx_dp_median))))

        rel_err = (fx_orig - fx_gt_decoded) / fx_gt_decoded * 100.0
        flag = "✅" if abs(rel_err) < 5 else ("🟡" if abs(rel_err) < 15 else "⚠️ ")

        print(f"  {ratio:.2f}  {hfov:>6.1f}°  {fx_dp_median:>7.1f}  "
              f"{fx_orig:>8.2f}  {rel_err:>+7.2f}%  {flag}")

        results.append({
            "label": label, "crop_ratio": ratio,
            "hfov_dp": hfov, "fx_dp": fx_dp_median,
            "fx_orig_est": fx_orig, "fx_gt": fx_gt_decoded,
            "rel_err_pct": rel_err,
        })

    torch.cuda.empty_cache()


def main():
    import depth_pro

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="5 frames per seq")
    args = parser.parse_args()
    n_frames = 10 if args.quick else MAX_FRAMES

    print("Loading Depth Pro model...")
    model, transform = depth_pro.create_model_and_transforms(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        precision=torch.float16,
    )
    model.eval()
    print(f"✅ Depth Pro on {next(model.parameters()).device}\n")

    results = []

    # ── PH2D AVP ──────────────────────────────────────────────────────────────
    with open(PH2D_META) as f:
        meta = json.load(f)
    avp_tasks = [(t, a) for t, a in meta["per_task_attributes"].items()
                 if a.get("embodiment_type") == "human_avp"]
    # Take first 3 tasks
    for task_name, _ in avp_tasks[:3]:
        task_dir = os.path.join(PH2D_ROOT, task_name)
        hdf5s = natsorted(glob(os.path.join(task_dir, "*.hdf5")))
        if not hdf5s:
            continue
        imgs = load_ph2d_frames(hdf5s[0], n_frames)
        evaluate_crop(f"PH2D-AVP/{task_name[:30]}", imgs,
                      model, transform,
                      GT_AVP_FX_RAW, GT_AVP_RAW_W, results)

    # ── EgoDex ───────────────────────────────────────────────────────────────
    import h5py
    for task in natsorted(os.listdir(EGODEX_ROOT))[:3]:
        task_dir = os.path.join(EGODEX_ROOT, task)
        if not os.path.isdir(task_dir):
            continue
        mp4s = natsorted(glob(os.path.join(task_dir, "*.mp4")))
        if not mp4s:
            continue
        mp4 = mp4s[0]
        stem = os.path.splitext(os.path.basename(mp4))[0]
        hdf5 = os.path.join(task_dir, stem + ".hdf5")
        if not os.path.exists(hdf5):
            continue
        with h5py.File(hdf5) as fh:
            K_raw = fh["camera/intrinsic"][:]
        gt_fx_eg, gt_rw_eg = float(K_raw[0,0]), int(K_raw[0,2]*2)
        imgs = load_egodex_frames(mp4, n_frames)
        evaluate_crop(f"EgoDex/{task[:30]}", imgs,
                      model, transform, gt_fx_eg, gt_rw_eg, results)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'═'*72}")
    print("  SUMMARY — Center-crop + Depth Pro focal length estimation")
    print(f"{'═'*72}")
    print(f"  {'Dataset':<28} {'Crop':>5} {'HFOV':>6} {'fx_gt':>7} {'fx_est':>7} {'Error':>8}")
    print(f"  {'-'*70}")
    for r in results:
        flag = "✅" if abs(r["rel_err_pct"]) < 5 else ("🟡" if abs(r["rel_err_pct"]) < 15 else "⚠️")
        print(f"  {r['label'][:28]:<28} {r['crop_ratio']:.2f}  {r['hfov_dp']:>5.1f}°  "
              f"{r['fx_gt']:>7.1f}  {r['fx_orig_est']:>7.2f}  "
              f"{r['rel_err_pct']:>+7.2f}%  {flag}")
    print(f"\n  Reference (no crop):")
    print(f"    Depth Pro / PH2D AVP  (104° HFOV): +118.69% ⚠️")
    print(f"    Depth Pro / EgoDex AVP(105° HFOV): +176.23% ⚠️")
    print(f"    Depth Pro / HO3D v3    (73° HFOV):   -1.50% ✅")
    print(f"    Depth Pro / DexYCB     (56° HFOV):   +0.20% ✅")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
