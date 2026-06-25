#!/usr/bin/env python3
"""
Baseline1 — Human Retarget DP : data collection (DexYCB, route A = dataset GT)
=============================================================================
Same human data as the main method, but learn the *full EE trajectory* instead
of contact points. For each DexYCB grasp sequence (subject / session / camera):

  1. object point cloud  ← SAM3D mesh (ycb_dex_NN) surface-sampled,
                            transformed by the dataset's GT object 6D pose (pose_y)
  2. hand                ← dataset's GT MANO joints (joint_3d, camera-frame, metres)
  3. retarget MANO → Franka 2-finger gripper EE pose (analytic, see below)
  4. everything → object-centric frame (subtract object centroid at frame 0)
  5. state[t] = [x,y,z, qw,qx,qy,qz, gripper]  (8D, robot/object frame)
     action[t] = state[t+1]

DP3 episode (HDF5, one per subject__session__camera):
    point_cloud : (T, 4096, 3)  object pts, object-centric frame
    state       : (T, 8)
    action      : (T, 8)
    finger_angles : (T, 45)     GT MANO articulation axis-angles (metadata, not used by DP3)
    wrist_pose    : (T, 4, 4)   GT MANO root pose, object-centric frame (metadata)

Why GT annotations and not our HaPTIC/FoundationPose outputs:
    HaPTIC's MANO (hand-size-anchored metric) and FoundationPose's object pose
    (DepthPro-fx-anchored metric) are scale-inconsistent in camera frame — that
    is exactly why the main method's contact-align step works in 2D pixel space.
    Baseline1 needs a 3D EE trajectory, so hand and object must share one metric;
    DexYCB ships per-frame GT MANO + GT object pose in one consistent camera frame.

Retarget (analytic, parallel gripper):
    t = thumb tip (joint_3d[4]),  i = index tip (joint_3d[8]),  w = wrist (joint_3d[0])
    c  = (t + i) / 2                                  pinch midpoint  ≈ between fingertips
    ex = norm(i - t)                                  gripper closing/opening axis
    ez = norm( (c - w) − ((c-w)·ex) ex )              approach axis (gripper points this way)
    ey = ez × ex                                      (right-handed by construction)
    R_ee  = [ex | ey | ez]
    p_ee  = c − EE_OFFSET · ez                        EE frame ~EE_OFFSET behind fingertips
    gripper = clip(1 − ‖i − t‖ / 0.08, 0, 1)          Franka 8 cm fingertip span; 0=open 1=closed
    + quaternion sign-continuity along the trajectory.

CAVEAT (v1): the object point cloud is the SAM3D `ycb_dex_NN` mesh (same mesh the
main method / Baseline2 use, max parity) placed by the dataset's `pose_y`, which is
defined w.r.t. the *YCB CAD model* frame. The SAM3D mesh's canonical frame differs
from the YCB CAD frame by a constant per-object rigid offset, so the rendered object
orientation carries that constant offset (the *size* is correct — scale.json is
calibrated to real_diameter_m, and the *location* is GT-correct). This is consistent
within every sequence of an object. A clean v2 could ICP-align each ycb_dex_NN mesh
to the YCB CAD model and compose, removing the offset.

Usage:
    # validation: 2 sessions of subject-07
    python Baseline1/retarget_human_to_ee.py --subject 20200928-subject-07 --limit 2
    # full route-A run (subjects 07-10, the ones with full GT on disk)
    python Baseline1/retarget_human_to_ee.py
    # then:
    python Baseline1/convert_to_zarr.py --input_dir Baseline1/data/episodes \
        --output_zarr Baseline1/data/human_dp_baseline.zarr
"""
import os, sys, argparse, json, glob, pickle
import numpy as np
import yaml
import h5py
import trimesh
from scipy.spatial.transform import Rotation
from natsort import natsorted

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import config

# ── paths ─────────────────────────────────────────────────────────────────────
DEXYCB_RAW = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData", "dexycb")
MESH_BASE  = os.path.join(config.DATA_HUB, "ProcessedData", "obj_meshes", "ycb")  # ycb_dex_NN/mesh.ply

# OakInk paths
OAKINK_ANNO_BASE = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData",
                                "oakink_v1", "image", "anno")
OAKINK_MESH_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "obj_meshes", "oakink")

# DexYCB camera serials. The main Phase-1A pipeline uses these 6 of the 8.
PIPELINE_CAMS = ["841412060263", "840412060917", "932122060857",
                 "836212060125", "932122061900", "932122062010"]
# DexYCB subjects with full GT on disk (labels_*.npz / meta.yml / pose.npz).
# Subjects 01-06 were color-only until 2026-05-14; full tarballs extracted then,
# bringing us to 10/10 coverage. All 10 subjects together cover all 21 YCB classes
# (foam_brick ycb_id=21 still skipped at retarget time — no SAM3D ycb_dex_21 mesh).
GT_SUBJECTS = ["20200709-subject-01", "20200813-subject-02", "20200820-subject-03",
               "20200903-subject-04", "20200908-subject-05", "20200918-subject-06",
               "20200928-subject-07", "20201002-subject-08", "20201015-subject-09",
               "20201022-subject-10"]
# per-object SAM3D↔YCB-CAD canonical alignment (produced by compute_sam3d_align.py)
ALIGN_DIR = os.path.join(PROJ, "Baseline1", "assets", "sam3d_align")

# DexYCB joint_3d layout (verified on data): 0=wrist, then 5 fingers × [knuckle/MCP, PIP, DIP, tip],
# finger order = thumb, index, middle, ring, pinky.  → thumb=1..4, index=5..8, middle=9..12, ring=13..16, pinky=17..20
J_WRIST, J_THUMB_TIP, J_INDEX_TIP = 0, 4, 8
# (MCP, tip) pairs for the 4 non-thumb fingers — used to read finger flexion → gripper open/close
FLEX_PAIRS = [(5, 8), (9, 12), (13, 16), (17, 20)]
D_EXT, D_FLEX = 0.085, 0.025   # m — ‖tip − MCP‖ for a fully extended / fully curled finger

EE_OFFSET    = 0.10   # m — EE frame this far behind the fingertip pinch point (≈ Franka panda_hand)
GRIPPER_SPAN = 0.08   # m — Franka 2-finger fingertip span (matches Baseline2's 1−mean(fingers)/0.04)
GRASP_MARGIN = 0.04   # m — "grasp onset" = first frame the hand is within (d_min + this) of the object
N_POINTS    = 4096
MAX_PINCH_M = 0.20    # thumb-index gap above this ⇒ bad joints ⇒ drop frame
MIN_FRAMES  = 5       # need at least this many valid frames to emit an episode
INVALID     = -0.5    # joint_3d / pose marker; valid coords are well above this in abs metric


# ── object mesh: SAM3D ycb_dex_NN, scaled, surface-sampled (cached per object) ─
_MESH_PTS_CACHE: dict = {}
_ALIGN_WARNED: set = set()

def get_object_points(dataset, obj_id, n_points, align_mode="auto"):
    """Return (n_points, 3) surface samples of the SAM3D mesh for `obj_id`, in metric metres.

    dataset: "dexycb" or "oakink".
    obj_id : str — DexYCB uses "ycb_dex_NN", OakInk uses e.g. "A01001".
    align_mode:
      'none'  — points in the SAM3D mesh's own canonical frame (v1 behaviour: constant
                per-object rotation offset vs the dataset's CAD frame).
      'sam3d' — apply Baseline1/assets/sam3d_align/{...}.json (R_align: SAM3D→CAD canonical;
                for OakInk this is an affine that also rescales to real metres). Errors if missing.
      'auto'  — use 'sam3d' if the json exists for this object, else 'none' (warn once).
    None if mesh missing. Cached per (dataset, obj_id, n_points, align_mode)."""
    key = (dataset, obj_id, n_points, align_mode)
    if key in _MESH_PTS_CACHE:
        return _MESH_PTS_CACHE[key]

    if dataset == "dexycb":
        obj_dir    = os.path.join(MESH_BASE, obj_id)                  # ycb_dex_NN/
        align_path = os.path.join(ALIGN_DIR, f"{obj_id}.json")
    elif dataset == "oakink":
        obj_dir    = os.path.join(OAKINK_MESH_BASE, obj_id)           # A01001/
        align_path = os.path.join(ALIGN_DIR, "oakink", f"{obj_id}.json")
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    mesh_path = os.path.join(obj_dir, "mesh.ply")
    if not os.path.exists(mesh_path):
        _MESH_PTS_CACHE[key] = None
        return None
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    # DexYCB SAM3D meshes have scale.json (calibrated to real diameter). OakInk doesn't —
    # its scale is folded into R_align (the ICP we run for OakInk used scale=True).
    sj_path = os.path.join(obj_dir, "scale.json")
    if os.path.exists(sj_path):
        sf = float(json.load(open(sj_path)).get("scale_factor", 1.0))
        if sf != 1.0:
            mesh.vertices = mesh.vertices * sf
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    pts = pts.astype(np.float64)

    if align_mode in ("sam3d", "auto"):
        if os.path.exists(align_path):
            R = np.asarray(json.load(open(align_path))["R_align_4x4"], dtype=np.float64)
            pts = (R[:3, :3] @ pts.T).T + R[:3, 3]
        elif align_mode == "sam3d":
            raise FileNotFoundError(
                f"--align-mode sam3d but no alignment for {dataset}/{obj_id}; "
                f"run Baseline1/compute_sam3d_align.py --dataset {dataset} first.")
        elif obj_id not in _ALIGN_WARNED:
            print(f"  [align] no SAM3D↔CAD alignment for {dataset}/{obj_id} yet — "
                  f"using SAM3D canonical frame (v1 caveat applies)")
            _ALIGN_WARNED.add(obj_id)
    pts = pts.astype(np.float32)
    _MESH_PTS_CACHE[key] = pts
    return pts


# ── retarget: MANO joints → EE pose (analytic) ────────────────────────────────
def finger_flexion(joint_3d):
    """How curled the 4 non-thumb fingers are, ∈ [0,1] (0 = extended, 1 = fully curled).
    Stored as episode metadata. NOT used as the gripper command: for a *parallel* gripper the
    command should mean "is the hand grasping the object", which is read from hand-object proximity
    (a human can grasp with extended fingers — a wrap grasp around a cylinder — or curled fingers)."""
    vals = []
    for mcp, tip in FLEX_PAIRS:
        d = float(np.linalg.norm(joint_3d[tip] - joint_3d[mcp]))
        vals.append(np.clip((D_EXT - d) / (D_EXT - D_FLEX), 0.0, 1.0))
    return float(np.mean(vals))


def mano_joints_to_ee(joint_3d):
    """joint_3d: (21,3) camera-frame metres → (p_ee(3), quat_wxyz(4), flex(1)) or None.
    `flex` is the finger-curl metadata; the gripper *command* is computed later from proximity."""
    w = joint_3d[J_WRIST]; t = joint_3d[J_THUMB_TIP]; i = joint_3d[J_INDEX_TIP]
    pinch = i - t
    pinch_d = float(np.linalg.norm(pinch))
    if pinch_d < 1e-4 or pinch_d > MAX_PINCH_M:
        return None
    ex = pinch / pinch_d
    fwd = ((t + i) * 0.5) - w
    fwd_n = float(np.linalg.norm(fwd))
    if fwd_n < 1e-3:
        return None
    fwd = fwd / fwd_n
    ez = fwd - np.dot(fwd, ex) * ex
    ez_n = float(np.linalg.norm(ez))
    if ez_n < 1e-3:
        return None
    ez = ez / ez_n
    ey = np.cross(ez, ex)                       # right-handed: ex × ey = ez
    R = np.column_stack([ex, ey, ez])
    c = (t + i) * 0.5
    p_ee = c - EE_OFFSET * ez
    q_xyzw = Rotation.from_matrix(R).as_quat()  # [x,y,z,w]
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    return p_ee.astype(np.float64), q_wxyz, finger_flexion(joint_3d)


def axisangle_to_mat4(rotvec3, trans3):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec3, dtype=np.float64)).as_matrix()
    T[:3, 3]  = np.asarray(trans3, dtype=np.float64)
    return T


# ── one (subject, session, camera) → episode dict or None ─────────────────────
def build_episode_dexycb(sess_dir, cam, meta, n_points, align_mode="auto"):
    grasp_ind = int(meta["ycb_grasp_ind"])
    ycb_ids   = list(meta["ycb_ids"])
    if not (0 <= grasp_ind < len(ycb_ids)):
        return None, "bad ycb_grasp_ind"
    ycb_class_id = int(ycb_ids[grasp_ind])
    obj_id  = f"ycb_dex_{ycb_class_id:02d}"
    obj_pts = get_object_points("dexycb", obj_id, n_points, align_mode)
    if obj_pts is None:
        return None, f"mesh {obj_id} not found"
    side = (meta.get("mano_sides") or ["right"])[0]
    n_frames = int(meta["num_frames"])
    cam_dir = os.path.join(sess_dir, cam)

    rec = []           # list of dicts per valid frame
    n_invalid = 0
    for f in range(n_frames):
        lab_path = os.path.join(cam_dir, f"labels_{f:06d}.npz")
        if not os.path.exists(lab_path):
            n_invalid += 1; continue
        try:
            lab = np.load(lab_path, allow_pickle=True)
            j   = np.asarray(lab["joint_3d"][0], dtype=np.float64)   # (21,3)
            pm  = np.asarray(lab["pose_m"][0],   dtype=np.float64)   # (51,)
            py  = np.asarray(lab["pose_y"][grasp_ind], dtype=np.float64)  # (3,4) = [R|t]
        except Exception:
            n_invalid += 1; continue
        # validity: joint_3d / pose_m use -1 marker; pose_y is all-zero when unlabeled
        if np.any(j < INVALID) or not np.any(np.abs(pm) > 1e-6) or not np.any(np.abs(py) > 1e-6):
            n_invalid += 1; continue
        ee = mano_joints_to_ee(j)
        if ee is None:
            n_invalid += 1; continue
        p_ee, q_wxyz, flex = ee
        # object pc in camera frame (caveat: pts in SAM3D canonical, py in YCB-CAD frame)
        pc_cam = (py[:, :3] @ obj_pts.T).T + py[:, 3]            # (N,3)
        oc_cam = py[:, 3].copy()                                # GT object origin in camera frame
        wrist_pose = axisangle_to_mat4(pm[0:3], pm[48:51])      # MANO root pose, camera frame
        rec.append(dict(f=f, pc=pc_cam.astype(np.float32), p=p_ee, oc=oc_cam, q=q_wxyz,
                        flex=flex, ang=pm[3:48].astype(np.float32), wp=wrist_pose))

    if len(rec) < MIN_FRAMES:
        return None, f"only {len(rec)} valid frames"

    # quaternion sign-continuity along the trajectory
    for k in range(1, len(rec)):
        if np.dot(rec[k]["q"], rec[k - 1]["q"]) < 0.0:
            rec[k]["q"] = -rec[k]["q"]

    # gripper command for a parallel gripper: 0 (open) while approaching, 1 (closed) once the hand
    # has reached the object and stays near it. "reached" = within (d_min + GRASP_MARGIN) of the GT
    # object centre; monotone after onset (matches Baseline2's clean 0→1 step; uses GT translations,
    # so unaffected by the SAM3D-vs-YCB mesh-frame caveat).
    d = np.array([float(np.linalg.norm(r["p"] - r["oc"])) for r in rec])   # true hand-object dist
    d_min = float(d.min())
    onset = int(np.argmax(d <= d_min + GRASP_MARGIN))   # first index satisfying the condition
    gripper = np.zeros(len(rec), dtype=np.float64); gripper[onset:] = 1.0

    # object-centric: subtract the object centroid (of the rendered point cloud) at the first frame
    origin = rec[0]["pc"].mean(axis=0).astype(np.float64)       # (3,)

    pcs, states, angs, wps, flexes = [], [], [], [], []
    for k, r in enumerate(rec):
        pc = r["pc"] - origin.astype(np.float32)
        p  = r["p"] - origin
        states.append(np.concatenate([p, r["q"], [gripper[k]]]).astype(np.float32))   # (8,)
        wp = r["wp"].copy(); wp[:3, 3] -= origin
        pcs.append(pc); angs.append(r["ang"]); wps.append(wp.astype(np.float32)); flexes.append(r["flex"])

    pcs    = np.stack(pcs)                      # (T,N,3)
    states = np.stack(states)                   # (T,8)
    angs   = np.stack(angs)                     # (T,45)
    wps    = np.stack(wps)                      # (T,4,4)
    flexes = np.array(flexes, dtype=np.float32) # (T,)
    if len(states) < 2:
        return None, "too short after processing"

    ep = dict(point_cloud=pcs[:-1], state=states[:-1], action=states[1:],
              finger_angles=angs[:-1], wrist_pose=wps[:-1], finger_flex=flexes[:-1],
              ycb_class_id=ycb_class_id, ycb_name=f"ycb_dex_{ycb_class_id:02d}",
              mano_side=side, n_valid=len(rec), n_invalid=n_invalid,
              grasp_onset=onset, min_hand_obj_dist_m=d_min)
    return ep, (f"{len(rec)} frames · grasp onset @ {onset}/{len(rec)} · "
                f"min hand-obj dist {d_min*1000:.0f}mm")


# ── OakInk: per-frame anno pkl loaders + episode discovery + builder ─────────
def _load_oakink_anno(subdir, filename):
    """Load one OakInk anno pkl. Returns ndarray or None.
    OakInk pkls store ndarray for hand_j / hand_v / obj_transf / cam_intr; torch tensors
    for general_info → converted to ndarray here."""
    p = os.path.join(OAKINK_ANNO_BASE, subdir, filename)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            v = pickle.load(f)
    except Exception:
        return None
    if hasattr(v, "numpy"):       # torch tensor
        v = v.numpy()
    return np.asarray(v, dtype=np.float64)


def discover_oakink_episodes():
    """Glob anno/hand_j/*.pkl → group by (seq_id, ts, sbj_flag, cam_id).
    Returns dict[key → list of (frame_idx, filename)] sorted by frame_idx ascending."""
    files = natsorted(glob.glob(os.path.join(OAKINK_ANNO_BASE, "hand_j", "*.pkl")))
    eps = {}
    for path in files:
        fn = os.path.basename(path)
        parts = fn[:-4].split("__")   # {seq_id}__{ts}__{sbj}__{frame}__{cam}.pkl
        if len(parts) != 5:
            continue
        seq_id, ts, sbj, frame_str, cam = parts
        try:
            frame_idx = int(frame_str)
        except ValueError:
            continue
        key = (seq_id, ts, sbj, cam)
        eps.setdefault(key, []).append((frame_idx, fn))
    for k in eps:
        eps[k].sort()
    return eps


def build_episode_oakink(episode_key, frames, n_points, align_mode="auto"):
    """One OakInk "episode" = a (seq_id, ts, sbj_flag, cam_id) tuple → its sorted frames."""
    seq_id, ts, sbj, cam = episode_key
    obj_id = seq_id.split("_")[0]        # 'A01001_0001_0000' → 'A01001'
    obj_pts = get_object_points("oakink", obj_id, n_points, align_mode)
    if obj_pts is None:
        return None, f"mesh oakink/{obj_id} not found"

    rec = []
    n_invalid = 0
    for frame_idx, fn in frames:
        j  = _load_oakink_anno("hand_j",     fn)
        ot = _load_oakink_anno("obj_transf", fn)
        if j is None or ot is None or j.shape != (21, 3) or ot.shape != (4, 4):
            n_invalid += 1; continue
        # validity: GT placeholder is rare in OakInk; check joints have sane values + ot has translation
        if np.any(j < INVALID) or not np.any(np.abs(ot[:3, 3]) > 1e-6):
            n_invalid += 1; continue
        ee = mano_joints_to_ee(j)
        if ee is None:
            n_invalid += 1; continue
        p_ee, q_wxyz, flex = ee
        pc_cam = (ot[:3, :3] @ obj_pts.T).T + ot[:3, 3]   # mesh × obj_transf in camera frame
        oc_cam = ot[:3, 3].copy()                          # object origin in camera frame
        wrist_pose = np.eye(4, dtype=np.float64)           # metadata-only (OakInk MANO params optional)
        rec.append(dict(f=frame_idx, pc=pc_cam.astype(np.float32), p=p_ee, oc=oc_cam, q=q_wxyz,
                        flex=flex, ang=np.zeros(45, dtype=np.float32), wp=wrist_pose))

    if len(rec) < MIN_FRAMES:
        return None, f"only {len(rec)} valid frames"

    # quaternion sign-continuity
    for k in range(1, len(rec)):
        if np.dot(rec[k]["q"], rec[k - 1]["q"]) < 0.0:
            rec[k]["q"] = -rec[k]["q"]

    # gripper: 0 → 1 step at "hand reaches object" (identical to DexYCB)
    d = np.array([float(np.linalg.norm(r["p"] - r["oc"])) for r in rec])
    d_min = float(d.min())
    onset = int(np.argmax(d <= d_min + GRASP_MARGIN))
    gripper = np.zeros(len(rec), dtype=np.float64); gripper[onset:] = 1.0

    # object-centric: subtract obj PC centroid at frame 0
    origin = rec[0]["pc"].mean(axis=0).astype(np.float64)

    pcs, states, angs, wps, flexes = [], [], [], [], []
    for k, r in enumerate(rec):
        pc = r["pc"] - origin.astype(np.float32)
        p  = r["p"] - origin
        states.append(np.concatenate([p, r["q"], [gripper[k]]]).astype(np.float32))
        wp = r["wp"].copy(); wp[:3, 3] -= origin
        pcs.append(pc); angs.append(r["ang"]); wps.append(wp.astype(np.float32)); flexes.append(r["flex"])

    pcs    = np.stack(pcs)
    states = np.stack(states)
    angs   = np.stack(angs)
    wps    = np.stack(wps)
    flexes = np.array(flexes, dtype=np.float32)
    if len(states) < 2:
        return None, "too short after processing"

    ep = dict(point_cloud=pcs[:-1], state=states[:-1], action=states[1:],
              finger_angles=angs[:-1], wrist_pose=wps[:-1], finger_flex=flexes[:-1],
              obj_id=obj_id, mano_side="right",
              n_valid=len(rec), n_invalid=n_invalid,
              grasp_onset=onset, min_hand_obj_dist_m=d_min)
    return ep, (f"{len(rec)} frames · grasp onset @ {onset}/{len(rec)} · "
                f"min hand-obj dist {d_min*1000:.0f}mm")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Baseline1 retarget — route A (dataset GT)")
    ap.add_argument("--dataset", choices=["dexycb", "oakink"], default="dexycb",
                    help="which dataset to retarget (default: dexycb)")
    ap.add_argument("--subject",  default=None, help="single subject dir name, e.g. 20200928-subject-07")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="explicit list of subject dirs (default: the 4 with full GT on disk)")
    ap.add_argument("--cams",     nargs="*", default=None,
                    help="camera serials (default: the 6 used by the main pipeline)")
    ap.add_argument("--all-cams", action="store_true", help="use all 8 DexYCB cameras")
    ap.add_argument("--limit",    type=int, default=0, help="process only the first N sessions per subject")
    ap.add_argument("--n-points", type=int, default=N_POINTS)
    ap.add_argument("--align-mode", choices=["none", "sam3d", "auto"], default="auto",
                    help="object-mesh canonical-frame fix: none=SAM3D frame (v1, has per-object "
                         "rotation offset); sam3d=apply Baseline1/assets/sam3d_align/*.json (errors if missing); "
                         "auto=use the alignment if present, else fall back to none (default)")
    ap.add_argument("--output-dir", default=os.path.join(PROJ, "Baseline1", "data", "episodes"))
    ap.add_argument("--redo",     action="store_true", help="overwrite existing episode files")
    ap.add_argument("--min-hand-obj-dist", type=float, default=0.5,
                    help="drop episodes where the hand never gets within this many m of the object")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    n_emit = n_skip = n_total = 0

    # ─── OakInk branch ───────────────────────────────────────────────────────
    if args.dataset == "oakink":
        eps = discover_oakink_episodes()
        keys = list(eps.keys())
        if args.limit:
            keys = keys[:args.limit]
        print(f"oakink: discovered {len(eps)} episodes (processing {len(keys)})")
        for key in keys:
            seq_id, ts, sbj, cam = key
            n_total += 1
            out = os.path.join(args.output_dir, f"oakink__{seq_id}__{ts}__{sbj}__{cam}.hdf5")
            if os.path.exists(out) and not args.redo:
                continue
            ep, msg = build_episode_oakink(key, eps[key], args.n_points, args.align_mode)
            tag = f"{seq_id}/{ts}/{sbj}/{cam}"
            if ep is None:
                n_skip += 1; print(f"  [skip] {tag}: {msg}"); continue
            if ep["min_hand_obj_dist_m"] > args.min_hand_obj_dist:
                n_skip += 1; print(f"  [skip] {tag}: hand never near object ({msg})"); continue
            with h5py.File(out, "w") as h:
                h.create_dataset("point_cloud",   data=ep["point_cloud"])
                h.create_dataset("state",         data=ep["state"])
                h.create_dataset("action",        data=ep["action"])
                h.create_dataset("finger_angles", data=ep["finger_angles"])
                h.create_dataset("wrist_pose",    data=ep["wrist_pose"])
                h.create_dataset("finger_flex",   data=ep["finger_flex"])
                h.attrs["dataset"]        = "oakink"
                h.attrs["obj_id"]         = ep["obj_id"]
                h.attrs["mano_side"]      = ep["mano_side"]
                h.attrs["seq_id"]         = seq_id
                h.attrs["timestamp"]      = ts
                h.attrs["subject_flag"]   = sbj
                h.attrs["camera"]         = cam
                h.attrs["ee_offset_m"]    = EE_OFFSET
                h.attrs["gripper_span_m"] = GRIPPER_SPAN
                h.attrs["align_mode"]     = args.align_mode
                h.attrs["grasp_onset"]    = int(ep["grasp_onset"])
                h.attrs["n_steps"]        = int(len(ep["state"]))
            n_emit += 1
            print(f"  [ok]   {tag}: {ep['obj_id']} · {len(ep['state'])} steps · {msg}")
        print(f"\nDone. {n_emit} episodes written, {n_skip} skipped, {n_total} episodes seen.")
        print(f"  → {args.output_dir}")
        return

    # ─── DexYCB branch (default) ─────────────────────────────────────────────
    if args.subject:
        subjects = [args.subject]
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = GT_SUBJECTS
    cams = (None if args.all_cams else (args.cams or PIPELINE_CAMS))

    for subj in subjects:
        subj_dir = os.path.join(DEXYCB_RAW, subj)
        if not os.path.isdir(subj_dir):
            print(f"[skip] subject dir not found: {subj_dir}"); continue
        sessions = natsorted([d for d in os.listdir(subj_dir)
                              if os.path.isdir(os.path.join(subj_dir, d)) and not d.startswith(".")])
        if args.limit:
            sessions = sessions[:args.limit]
        for sess in sessions:
            sess_dir = os.path.join(subj_dir, sess)
            meta_path = os.path.join(sess_dir, "meta.yml")
            if not (os.path.exists(meta_path) and os.path.getsize(meta_path) > 0):
                print(f"[skip] {subj}/{sess}: no meta.yml (color-only download?)"); continue
            try:
                meta = yaml.safe_load(open(meta_path))
            except Exception as e:
                print(f"[skip] {subj}/{sess}: meta.yml parse error: {e}"); continue
            serials = list(meta.get("serials", []))
            use_cams = serials if cams is None else [c for c in serials if c in cams]
            for cam in use_cams:
                if not os.path.isdir(os.path.join(sess_dir, cam)):
                    continue
                n_total += 1
                out = os.path.join(args.output_dir, f"dexycb__{subj}__{sess}__{cam}.hdf5")
                if os.path.exists(out) and not args.redo:
                    continue
                ep, msg = build_episode_dexycb(sess_dir, cam, meta, args.n_points, args.align_mode)
                tag = f"{subj}/{sess}/{cam}"
                if ep is None:
                    n_skip += 1; print(f"  [skip] {tag}: {msg}"); continue
                if ep["min_hand_obj_dist_m"] > args.min_hand_obj_dist:
                    n_skip += 1; print(f"  [skip] {tag}: hand never near object ({msg})"); continue
                with h5py.File(out, "w") as h:
                    h.create_dataset("point_cloud",   data=ep["point_cloud"])
                    h.create_dataset("state",         data=ep["state"])
                    h.create_dataset("action",        data=ep["action"])
                    h.create_dataset("finger_angles", data=ep["finger_angles"])
                    h.create_dataset("wrist_pose",    data=ep["wrist_pose"])
                    h.create_dataset("finger_flex",   data=ep["finger_flex"])
                    h.attrs["dataset"]        = "dexycb"
                    h.attrs["obj_id"]         = ep["ycb_name"]
                    h.attrs["ycb_class_id"]   = ep["ycb_class_id"]
                    h.attrs["mano_side"]      = ep["mano_side"]
                    h.attrs["subject"]        = subj
                    h.attrs["session"]        = sess
                    h.attrs["camera"]         = cam
                    h.attrs["ee_offset_m"]    = EE_OFFSET
                    h.attrs["gripper_span_m"] = GRIPPER_SPAN
                    h.attrs["align_mode"]     = args.align_mode
                    h.attrs["grasp_onset"]    = int(ep["grasp_onset"])
                    h.attrs["n_steps"]        = int(len(ep["state"]))
                n_emit += 1
                print(f"  [ok]   {tag}: {ep['ycb_name']} · {len(ep['state'])} steps · {msg}")
    print(f"\nDone. {n_emit} episodes written, {n_skip} skipped, {n_total} (subj,session,cam) tuples seen.")
    print(f"  → {args.output_dir}")


if __name__ == "__main__":
    main()
