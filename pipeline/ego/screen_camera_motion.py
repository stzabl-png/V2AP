"""
Camera motion pre-screening for MegaSAM calibration selection.

Problem:
  MegaSAM focal length BA only works when the camera moves enough
  (parallax needed to constrain fx via geometry). Static-camera sequences
  give unreliable intrinsics even after SLAM.

Solution:
  Before running expensive MegaSAM, score each episode by how much
  the CAMERA (not just the hand) moves. Track corner features in the
  BACKGROUND regions (borders of frame away from hands) across frames.
  Background optical flow = camera motion, not object motion.

  Threshold: median background flow > CAM_MOTION_THRESH px → good candidate

Usage:
  conda activate mega_sam          # or depth-pro, only needs cv2 + h5py
  cd $PROJ
  python data/screen_camera_motion.py [--dataset ph2d_avp|egodex] [--top N]

Output:
  Prints ranked episode list sorted by camera motion score.
  Saves top-N episode paths for MegaSAM calibration run.
"""

import os, sys, json, argparse, numpy as np, cv2
from glob import glob
from natsort import natsorted
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

DATA_ROOT   = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")
PH2D_ROOT   = os.path.join(DATA_ROOT, "ph2d")
PH2D_META   = os.path.join(DATA_ROOT, "ph2d", "ph2d_metadata.json")
EGODEX_ROOT = os.path.join(DATA_ROOT, "egodex", "test")

CAM_MOTION_THRESH = 8.0    # px — median bg flow above this = camera moving
SAMPLE_FRAMES     = 20     # frames to sample per episode for screening
BG_BORDER_FRAC    = 0.25   # outer 25% of frame = "background" region


# ══════════════════════════════════════════════════════════════════════════════
# Camera motion scorer
# ══════════════════════════════════════════════════════════════════════════════

def camera_motion_score(frames):
    """
    Estimate camera motion from a list of BGR frames.

    Strategy:
      1. Only track corner features in the OUTER border of the frame
         (background region, away from where hands/objects typically appear).
      2. Compute Lucas-Kanade optical flow between consecutive sample pairs.
      3. Camera motion = median magnitude of background-region flow.

    Returns:
      score (float): median background pixel displacement in px.
                     High score = camera moving. Low score = static camera.
      n_valid_pairs: how many frame pairs had sufficient trackable features.
    """
    if len(frames) < 2:
        return 0.0, 0

    scores = []
    H, W = frames[0].shape[:2]

    # Background mask = outer border ring
    bg_mask = np.zeros((H, W), dtype=np.uint8)
    bh, bw = int(H * BG_BORDER_FRAC), int(W * BG_BORDER_FRAC)
    bg_mask[:bh, :]  = 255   # top strip
    bg_mask[-bh:, :] = 255   # bottom strip
    bg_mask[:, :bw]  = 255   # left strip
    bg_mask[:, -bw:] = 255   # right strip

    # Sample frame pairs (skip first few to avoid motion blur at start)
    step = max(1, len(frames) // 8)
    pairs = [(frames[i], frames[i+step])
             for i in range(0, len(frames)-step, step)]
    if not pairs:
        pairs = [(frames[0], frames[-1])]

    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                30, 0.01))

    for f1, f2 in pairs:
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        # Detect corners only in background region
        pts = cv2.goodFeaturesToTrack(
            g1, maxCorners=300, qualityLevel=0.01,
            minDistance=8, mask=bg_mask
        )
        if pts is None or len(pts) < 8:
            scores.append(0.0)
            continue

        # Track with Lucas-Kanade
        pts2, status, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts, None, **lk_params)
        if pts2 is None:
            scores.append(0.0)
            continue

        ok = status.ravel() == 1
        if ok.sum() < 5:
            scores.append(0.0)
            continue

        p1 = pts[ok].reshape(-1, 2)
        p2 = pts2[ok].reshape(-1, 2)

        # Use RANSAC homography to separate camera motion from noise
        if len(p1) >= 8:
            H_mat, inliers = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
            if inliers is not None and inliers.sum() > 4:
                p1 = p1[inliers.ravel() == 1]
                p2 = p2[inliers.ravel() == 1]

        flow = np.linalg.norm(p2 - p1, axis=1)
        scores.append(float(np.median(flow)))

    if not scores:
        return 0.0, 0
    return float(np.median(scores)), len([s for s in scores if s > 0])


# ══════════════════════════════════════════════════════════════════════════════
# Frame samplers
# ══════════════════════════════════════════════════════════════════════════════

def sample_frames_ph2d(hdf5_path, n=SAMPLE_FRAMES):
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        raw = f["observation.image.left"][:]
    total = len(raw)
    idxs = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames = []
    for i in idxs:
        buf = np.frombuffer(raw[i].tobytes(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def sample_frames_mp4(mp4_path, n=SAMPLE_FRAMES):
    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Dataset scanners
# ══════════════════════════════════════════════════════════════════════════════

def scan_ph2d_avp(top_n=None):
    with open(PH2D_META) as f:
        meta = json.load(f)
    avp_tasks = [(t, a) for t, a in meta["per_task_attributes"].items()
                 if a.get("embodiment_type") == "human_avp"]

    results = []
    for task_name, _ in tqdm(avp_tasks, desc="Screening PH2D AVP"):
        task_dir = os.path.join(PH2D_ROOT, task_name)
        hdf5s = natsorted(glob(os.path.join(task_dir, "*.hdf5")))
        if not hdf5s:
            continue
        # Score each episode in this task
        for hdf5_path in hdf5s[:3]:  # max 3 episodes per task
            frames = sample_frames_ph2d(hdf5_path)
            score, n_valid = camera_motion_score(frames)
            results.append({
                "dataset": "ph2d_avp",
                "task": task_name,
                "path": hdf5_path,
                "score": score,
                "n_valid_pairs": n_valid,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def scan_egodex(top_n=None):
    results = []
    tasks = natsorted(os.listdir(EGODEX_ROOT))
    for task in tqdm(tasks, desc="Screening EgoDex"):
        task_dir = os.path.join(EGODEX_ROOT, task)
        if not os.path.isdir(task_dir):
            continue
        mp4s = natsorted(glob(os.path.join(task_dir, "*.mp4")))
        for mp4 in mp4s[:2]:  # max 2 episodes per task
            stem = os.path.splitext(os.path.basename(mp4))[0]
            hdf5 = os.path.join(task_dir, stem + ".hdf5")
            if not os.path.exists(hdf5):
                continue
            frames = sample_frames_mp4(mp4)
            score, n_valid = camera_motion_score(frames)
            results.append({
                "dataset": "egodex",
                "task": task,
                "path": mp4,
                "score": score,
                "n_valid_pairs": n_valid,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results, top_n, dataset_label):
    good  = [r for r in results if r["score"] >= CAM_MOTION_THRESH]
    maybe = [r for r in results if 3.0 <= r["score"] < CAM_MOTION_THRESH]
    bad   = [r for r in results if r["score"] < 3.0]

    print(f"\n{'═'*70}")
    print(f"  {dataset_label}  —  Camera Motion Screening")
    print(f"  Threshold: {CAM_MOTION_THRESH} px background flow")
    print(f"{'═'*70}")
    print(f"  {'Score':>6}  {'Status':<8}  {'Task / Path'}")
    print(f"  {'-'*68}")
    for r in results[:top_n]:
        if r["score"] >= CAM_MOTION_THRESH:
            flag = "✅ GOOD "
        elif r["score"] >= 3.0:
            flag = "🟡 MAYBE"
        else:
            flag = "❌ BAD  "
        short_path = r["path"].split("/")[-2] + "/" + r["path"].split("/")[-1][-30:]
        print(f"  {r['score']:>6.2f}  {flag}  {short_path}")

    print(f"\n  Summary:  ✅ {len(good)} good  🟡 {len(maybe)} marginal  ❌ {len(bad)} static")
    print(f"  Top candidates for MegaSAM calibration:")
    for r in good[:5]:
        print(f"    score={r['score']:.1f}  {r['path']}")
    if not good:
        print(f"    (none above threshold — use best available:)")
        for r in results[:3]:
            print(f"    score={r['score']:.1f}  {r['path']}")

    return good[:top_n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ph2d_avp", "egodex", "both"],
                        default="both")
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N episodes by motion score")
    parser.add_argument("--save", type=str, default=None,
                        help="Save good episode paths to this txt file")
    args = parser.parse_args()

    all_good = []

    if args.dataset in ("ph2d_avp", "both"):
        results = scan_ph2d_avp()
        good = print_results(results, args.top, "PH2D AVP")
        all_good.extend(good)

    if args.dataset in ("egodex", "both"):
        results = scan_egodex()
        good = print_results(results, args.top, "EgoDex")
        all_good.extend(good)

    if args.save and all_good:
        with open(args.save, "w") as f:
            for r in all_good:
                f.write(r["path"] + "\n")
        print(f"\n  Saved {len(all_good)} good paths → {args.save}")


if __name__ == "__main__":
    main()
