#!/usr/bin/env python3
"""
Video → Training Data Bridge
=============================
Converts MANO cache (from HaWoR/HaPTIC) + object mesh → PointNet++ training HDF5.

For each ARCTIC object with cached MANO predictions:
  1. Load object mesh (via mesh_provider)
  2. Load all MANO cache files for that object
  3. Accumulate per-vertex contact frequency (hand ↔ mesh distance < threshold)
  4. Sample point cloud from mesh surface
  5. Transfer contact labels to point cloud
  6. Save as HDF5 compatible with gen_m5_training_data.py output

Output format (matches data_hub/training_m5/):
    point_cloud:  (N, 3)
    normals:      (N, 3)
    human_prior:  (N,)   — video-derived contact probability
    robot_gt:     (N,)   — zeros (no robot GT for video data)
    force_center: (3,)   — centroid of contact region

Usage:
    # Run from project root
    python -m data.video_to_training_data                  # auto: best per object
    python -m data.video_to_training_data --source haptic  # force third-person only
    python -m data.video_to_training_data --source hawor   # force first-person only
    python -m data.video_to_training_data --source both    # force merge all
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from data.mesh_provider import get_provider

# Import MANO forward pass for GT centroid alignment
try:
    sys.path.insert(0, config.HAWOR_DIR)
    from hawor.utils.process import run_mano, run_mano_left
    HAS_MANO = True
except ImportError:
    HAS_MANO = False
    print("⚠️  hawor MANO not available, centroid alignment disabled")

# ─── Constants ───
N_POINTS = 4096
CONTACT_SIGMA = 0.025  # Gaussian smoothing (25mm)

OBJECTS = ['box', 'capsulemachine', 'espressomachine', 'ketchup',
           'laptop', 'microwave', 'mixer', 'notebook', 'phone',
           'scissors', 'waffleiron']


def load_onset_annotations():
    """Load grasp onset frame annotations (start/end per sequence)."""
    if not os.path.exists(config.ONSET_JSON):
        print(f"⚠️  No onset annotations at {config.ONSET_JSON}")
        return {}
    with open(config.ONSET_JSON) as f:
        return json.load(f)


def get_object_from_sequence(seq_name):
    """Extract object name from sequence name like 'box_grab_01'."""
    for obj in OBJECTS:
        if seq_name.startswith(obj):
            return obj
    return None


def load_hawor_caches(object_name):
    """Load all HaWoR cache files for a given object.

    Returns list of (right_verts_cam, left_verts_cam) arrays in camera space.
    Each is (N_frames, 778, 3).
    """
    cache_dir = config.HAWOR_CACHE
    if not os.path.isdir(cache_dir):
        return []

    all_hand_verts = []
    onset = load_onset_annotations()

    for fn in sorted(os.listdir(cache_dir)):
        if not fn.endswith('.npz'):
            continue
        parts = fn.replace('.npz', '').split('__')
        if len(parts) != 2:
            continue
        subject, seq_name = parts
        obj = get_object_from_sequence(seq_name)
        if obj != object_name:
            continue

        data = dict(np.load(os.path.join(cache_dir, fn), allow_pickle=True))

        # Get frame range from onset annotations
        key = f"{subject}__{seq_name}"
        if key in onset:
            start = onset[key].get('clip_start', onset[key].get('start', 0))
            end = onset[key].get('clip_end', onset[key].get('end', -1))
        else:
            start = 0
            end = -1

        # Convert to camera space using R_w2c, t_w2c
        for hand_key in ['right_verts', 'left_verts']:
            if hand_key not in data:
                continue
            verts = data[hand_key]  # (N, 778, 3) world space
            if verts.ndim != 3:
                continue

            R = data.get('R_w2c', None)
            t = data.get('t_w2c', None)

            if end == -1:
                end = len(verts) - 1
            sl = slice(max(0, start), min(end + 1, len(verts)))
            v_clip = verts[sl]

            if R is not None and t is not None:
                R_clip = R[sl]
                t_clip = t[sl]
                # world → camera: v_cam = R @ v_world + t
                v_cam = np.einsum('fij,fvj->fvi', R_clip, v_clip) + t_clip[:, None, :]
            else:
                v_cam = v_clip

            all_hand_verts.append(v_cam)

    return all_hand_verts


def load_haptic_caches(object_name):
    """Load all HaPTIC cache files for a given object.

    Returns list of hand_verts arrays, each (N_frames, 778, 3) in camera space.
    """
    cache_dir = config.HAPTIC_CACHE
    if not os.path.isdir(cache_dir):
        return []

    all_hand_verts = []
    onset = load_onset_annotations()

    for fn in sorted(os.listdir(cache_dir)):
        if not fn.endswith('.npz'):
            continue
        # Parse: s01__box_grab_01_cam1.npz
        base = fn.replace('.npz', '')
        parts = base.split('__')
        if len(parts) != 2:
            continue
        subject = parts[0]
        rest = parts[1]  # box_grab_01_cam1

        obj = get_object_from_sequence(rest)
        if obj != object_name:
            continue

        data = dict(np.load(os.path.join(cache_dir, fn), allow_pickle=True))

        key = f"{subject}__{rest.rsplit('_cam', 1)[0]}"  # Remove _camN suffix

        if key in onset:
            start = onset[key].get('clip_start', onset[key].get('start', 0))
            end = onset[key].get('clip_end', onset[key].get('end', -1))
        else:
            start = 0
            end = -1

        # HaPTIC may store as verts_dict (dict[int→(778,3)]) or hand_verts/right_verts
        verts = None
        is_clip_relative = False  # True when verts are 0-indexed within clip window
        if 'verts_dict' in data:
            vd = data['verts_dict'].item() if data['verts_dict'].ndim == 0 else data['verts_dict']
            if isinstance(vd, dict):
                max_key = max(vd.keys())
                frames = []
                for k in range(max_key + 1):
                    if k in vd:
                        frames.append(vd[k])
                    # skip missing frames
                if frames:
                    verts = np.stack(frames, axis=0)  # (N, 778, 3)
                    is_clip_relative = True  # keys 0..N-1 are clip-relative
        elif 'hand_verts' in data:
            verts = data['hand_verts']
        elif 'right_verts' in data:
            verts = data['right_verts']

        if verts is None:
            continue
        if verts.ndim != 3:
            continue

        if is_clip_relative:
            # verts already spans the contact window; use all frames
            all_hand_verts.append(verts)
        else:
            if end == -1:
                end = len(verts) - 1
            sl = slice(max(0, start), min(end + 1, len(verts)))
            all_hand_verts.append(verts[sl])

    return all_hand_verts


def load_arctic_object_poses(subject, seq_name):
    """Load per-frame ARCTIC GT object pose.

    Returns (obj_rots, obj_trans) where:
        obj_rots: (N, 3) axis-angle rotation
        obj_trans: (N, 3) translation in meters
    Or (None, None) if not available.
    """
    arctic_root = config.ARCTIC_ROOT
    obj_path = os.path.join(arctic_root, 'raw_seqs', subject, f"{seq_name}.object.npy")
    if not os.path.exists(obj_path):
        return None, None
    obj_data = np.load(obj_path, allow_pickle=True)
    # obj_data: (N, 7+) — [angle, rot_x, rot_y, rot_z, tx, ty, tz, ...]
    obj_rots = obj_data[:, 1:4]    # axis-angle
    obj_trans = obj_data[:, 4:7] / 1000.0  # mm → m
    return obj_rots, obj_trans


def load_arctic_egocam(subject, seq_name):
    """Load ARCTIC egocam camera extrinsics.

    Returns (R_ego, T_ego) each (N, 3, 3) and (N, 3).
    Or (None, None) if not available.
    """
    arctic_root = config.ARCTIC_ROOT
    ego_path = os.path.join(arctic_root, 'raw_seqs', subject, f"{seq_name}.egocam.dist.npy")
    if not os.path.exists(ego_path):
        return None, None
    egocam = np.load(ego_path, allow_pickle=True).flat[0]
    return egocam['R_k_cam_np'], egocam['T_k_cam_np']


def load_arctic_3rd_cam(subject, cam_id=1):
    """Load ARCTIC third-person camera extrinsics from misc.json.

    Returns (R_cam, T_cam) as 3x3 and 3-vector.
    """
    import json
    misc_path = os.path.join(config.ARCTIC_ROOT, 'meta', 'misc.json')
    if not os.path.exists(misc_path):
        misc_path = os.path.join(config.ARCTIC_ROOT, '..', 'meta', 'misc.json')
    if not os.path.exists(misc_path):
        return None, None
    with open(misc_path) as f:
        misc = json.load(f)
    subj_key = subject
    if subj_key not in misc:
        return None, None
    w2c = np.array(misc[subj_key]['world2cam'][cam_id])
    return w2c[:3, :3], w2c[:3, 3]


def load_gt_mano_verts(subject, seq_name):
    """Load GT MANO hand vertices in WORLD space for a sequence.

    Returns (gt_right_world, gt_left_world) each (N, 778, 3), or (None, None).

    Note: run_mano/run_mano_left use relative path '_DATA/data/mano' for the
    MANO model, so we temporarily chdir to config.HAWOR_DIR during the forward pass.
    """
    if not HAS_MANO:
        return None, None
    import torch
    mano_path = os.path.join(config.ARCTIC_ROOT, 'raw_seqs', subject, f"{seq_name}.mano.npy")
    if not os.path.exists(mano_path):
        return None, None

    # Save CWD and switch to hawor dir (run_mano needs relative '_DATA/data/mano')
    original_cwd = os.getcwd()
    os.chdir(config.HAWOR_DIR)
    try:
        mano_gt = np.load(mano_path, allow_pickle=True).item()
        N = mano_gt['right']['rot'].shape[0]
        # Right hand
        rt = torch.tensor(np.array(mano_gt['right']['trans']), dtype=torch.float32).unsqueeze(0)
        rr = torch.tensor(np.array(mano_gt['right']['rot']), dtype=torch.float32).unsqueeze(0)
        rp = torch.tensor(np.array(mano_gt['right']['pose']), dtype=torch.float32).unsqueeze(0)
        rs = torch.tensor(np.tile(np.array(mano_gt['right']['shape']), (N, 1)), dtype=torch.float32).unsqueeze(0)
        r_out = run_mano(rt, rr, rp, betas=rs, use_cuda=False)
        gt_right = r_out['vertices'][0].numpy()
        # Left hand
        lt = torch.tensor(np.array(mano_gt['left']['trans']), dtype=torch.float32).unsqueeze(0)
        lr = torch.tensor(np.array(mano_gt['left']['rot']), dtype=torch.float32).unsqueeze(0)
        lp = torch.tensor(np.array(mano_gt['left']['pose']), dtype=torch.float32).unsqueeze(0)
        ls = torch.tensor(np.tile(np.array(mano_gt['left']['shape']), (N, 1)), dtype=torch.float32).unsqueeze(0)
        l_out = run_mano_left(lt, lr, lp, betas=ls, use_cuda=False)
        gt_left = l_out['vertices'][0].numpy()
        return gt_right, gt_left  # (N, 778, 3) each, world space
    except Exception as e:
        print(f"    ⚠️  GT MANO load failed for {subject}/{seq_name}: {e}")
        return None, None
    finally:
        os.chdir(original_cwd)


def accumulate_contact_labels(mesh_verts_m, mesh_faces, hand_verts_list,
                               threshold, obj_poses=None, cam_extrinsics=None,
                               mesh_in_camera_space=False, frame_offsets=None,
                               gt_mano_data=None):
    """Compute per-mesh-vertex contact frequency from accumulated hand predictions.

    Both the mesh and hand must be in the same coordinate space. If obj_poses
    and cam_extrinsics are provided, the mesh template is transformed to camera
    space per-frame to align with hand verts (which are already in camera space).

    If mesh_in_camera_space=True (SAM3D), the mesh is already aligned with
    hand verts and no pose transforms are needed.

    Args:
        mesh_verts_m: (V, 3) object mesh vertices in meters
        mesh_faces: (F, 3) mesh faces
        hand_verts_list: list of (N_frames, 778, 3) hand vertex arrays
        threshold: distance threshold for contact (meters)
        obj_poses: list of (obj_rots, obj_trans) per corresponding hand array
        cam_extrinsics: list of (R_cam, T_cam) per corresponding hand array
        mesh_in_camera_space: if True, skip pose transforms (SAM3D mesh)
        frame_offsets: list of int, onset frame offset per sequence.
            hand_verts[0] corresponds to GT frame frame_offsets[seq_idx].
        gt_mano_data: list of (gt_right_world, gt_left_world, R_cam, T_cam)
            per sequence. Used for centroid alignment of predicted hands.

    Returns:
        contact_freq: (V,) float, contact probability per vertex [0, 1]
    """
    from scipy.spatial.transform import Rotation

    contact_count = np.zeros(len(mesh_verts_m), dtype=np.float32)
    total_frames = 0
    skipped_seqs = 0

    for seq_idx, hand_verts in enumerate(hand_verts_list):
        # Get per-sequence transforms
        has_pose = (obj_poses is not None and seq_idx < len(obj_poses)
                    and obj_poses[seq_idx] is not None)
        has_cam = (cam_extrinsics is not None and seq_idx < len(cam_extrinsics)
                   and cam_extrinsics[seq_idx] is not None)

        # ARCTIC mesh (canonical space) needs GT pose to align with camera-space hands.
        # If no pose available and mesh is NOT already in camera space → skip.
        if not mesh_in_camera_space and not has_pose:
            skipped_seqs += 1
            continue

        # Onset frame offset: hand_verts[0] corresponds to GT frame offset
        offset = 0
        if frame_offsets is not None and seq_idx < len(frame_offsets):
            offset = frame_offsets[seq_idx]

        # Check for GT MANO data for centroid alignment
        has_gt = (gt_mano_data is not None and seq_idx < len(gt_mano_data)
                  and gt_mano_data[seq_idx] is not None)

        for frame_idx in range(len(hand_verts)):
            hand_v = hand_verts[frame_idx]  # (778, 3) in camera space

            # GT frame index for object pose and camera extrinsics
            fi_gt = offset + frame_idx

            if mesh_in_camera_space:
                # SAM3D: mesh already in camera space, use directly
                mesh_cam = mesh_verts_m
            elif has_pose:
                obj_rots, obj_trans = obj_poses[seq_idx]
                fi = min(fi_gt, len(obj_rots) - 1)
                R_obj = Rotation.from_rotvec(obj_rots[fi]).as_matrix()
                t_obj = obj_trans[fi]
                mesh_world = (R_obj @ mesh_verts_m.T).T + t_obj

                if has_cam:
                    R_cam, T_cam = cam_extrinsics[seq_idx]
                    if R_cam.ndim == 3:  # per-frame (egocam)
                        fi_cam = min(fi_gt, len(R_cam) - 1)
                        R_c = R_cam[fi_cam]
                        T_c = T_cam[fi_cam].flatten()
                    else:  # static (3rd-person)
                        R_c = R_cam
                        T_c = T_cam.flatten()
                    mesh_cam = (R_c @ mesh_world.T).T + T_c
                else:
                    mesh_cam = mesh_world

            # Centroid alignment: shift predicted hand to GT MANO centroid
            if has_gt:
                gt_r_world, gt_l_world, gt_R_cam, gt_T_cam = gt_mano_data[seq_idx]
                fi_m = min(fi_gt, len(gt_r_world) - 1)
                # GT hands → camera space
                if gt_R_cam.ndim == 3:  # per-frame
                    fi_c = min(fi_gt, len(gt_R_cam) - 1)
                    Rc, Tc = gt_R_cam[fi_c], gt_T_cam[fi_c].flatten()
                else:
                    Rc, Tc = gt_R_cam, gt_T_cam.flatten()
                gt_r_cam = (Rc @ gt_r_world[fi_m].T).T + Tc
                gt_l_cam = (Rc @ gt_l_world[fi_m].T).T + Tc
                gt_both = np.concatenate([gt_r_cam, gt_l_cam], axis=0)
                gt_centroid = gt_both.mean(0)
                pred_centroid = hand_v.mean(0)
                hand_v = hand_v + (gt_centroid - pred_centroid)

            # For each mesh vertex, find min distance to any hand vertex
            hand_tree = cKDTree(hand_v)
            dists, _ = hand_tree.query(mesh_cam)

            contact_mask = dists < threshold
            contact_count[contact_mask] += 1
            total_frames += 1

    if skipped_seqs > 0:
        print(f"    ⚠️  Skipped {skipped_seqs} sequences (no GT object pose)")

    if total_frames == 0:
        return contact_count

    # Normalize to [0, 1]
    contact_freq = contact_count / total_frames

    return contact_freq


def transfer_labels_to_pointcloud(mesh, contact_per_vertex, n_points=N_POINTS):
    """Sample point cloud from mesh and transfer vertex-level labels.

    Uses barycentric interpolation for accurate label transfer.
    """
    import trimesh

    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    points = points.astype(np.float32)
    normals = mesh.face_normals[face_indices].astype(np.float32)

    # Transfer labels: average of face vertices' contact values
    labels = np.zeros(n_points, dtype=np.float32)
    for i, fi in enumerate(face_indices):
        v_ids = mesh.faces[fi]
        labels[i] = np.mean(contact_per_vertex[v_ids])

    return points, normals, labels


def resolve_source(object_name, source):
    """Resolve data source for an object. Returns (effective_source, label).

    auto mode priority:
      1. haptic (third-person, higher accuracy)
      2. hawor  (first-person, fallback)
      3. both   (if both exist, merge for best coverage)
    """
    if source != 'auto':
        return source, source

    has_haptic = len(load_haptic_caches(object_name)) > 0
    has_hawor = len(load_hawor_caches(object_name)) > 0

    if has_haptic and has_hawor:
        return 'both', 'auto→both'
    elif has_haptic:
        return 'haptic', 'auto→haptic'
    elif has_hawor:
        return 'hawor', 'auto→hawor'
    else:
        return 'none', 'auto→none'


def process_object(object_name, source, mesh_provider, pred_threshold):
    """Process one ARCTIC object, producing training data.

    Args:
        source: 'hawor', 'haptic', 'both', or 'auto'

    Returns (result_dict, effective_source) or (None, None).
    """
    # Step 1: Get mesh (ARCTIC mesh is in mm → convert to meters)
    mesh = mesh_provider.get_mesh(object_name)
    if mesh is None:
        print(f"  ⚠️  No mesh for '{object_name}', skipping")
        return None, None

    mesh_verts_m = np.array(mesh.vertices) / 1000.0  # mm → m

    # Step 2: Resolve source per-object
    effective_source, label = resolve_source(object_name, source)
    if effective_source == 'none':
        print(f"  ⚠️  No MANO cache for '{object_name}'")
        return None, None

    # Step 3: Load MANO caches + collect sequence metadata + onset offsets
    hand_verts_list = []
    seq_metadata = []  # [(subject, seq_name, source_type), ...]
    frame_offsets = []  # onset start frame for each hand_verts entry

    onset = load_onset_annotations()

    if effective_source in ('hawor', 'both'):
        hawor_data = load_hawor_caches(object_name)
        # Extract metadata from cache filenames — must match load_hawor_caches order
        cache_dir = config.HAWOR_CACHE
        for fn in sorted(os.listdir(cache_dir)):
            if not fn.endswith('.npz'):
                continue
            parts = fn.replace('.npz', '').split('__')
            if len(parts) != 2:
                continue
            subject, seq_name = parts
            obj = get_object_from_sequence(seq_name)
            if obj != object_name:
                continue

            # Get onset start for this sequence
            key = f"{subject}__{seq_name}"
            clip_start = onset[key].get('clip_start', onset[key].get('start', 0)) if key in onset else 0

            # HaWoR produces 2 entries per file (right + left), each with same offset
            data = dict(np.load(os.path.join(cache_dir, fn), allow_pickle=True))
            for hand_key in ['right_verts', 'left_verts']:
                if hand_key in data and data[hand_key].ndim == 3:
                    seq_metadata.append((subject, seq_name, 'hawor'))
                    frame_offsets.append(clip_start)
        hand_verts_list.extend(hawor_data)

    n_hawor = len(hand_verts_list)  # boundary: entries [0:n_hawor] are HaWoR

    if effective_source in ('haptic', 'both'):
        haptic_data = load_haptic_caches(object_name)
        cache_dir = config.HAPTIC_CACHE
        for fn in sorted(os.listdir(cache_dir)):
            if not fn.endswith('.npz'):
                continue
            parts = fn.replace('.npz', '').split('__')
            if len(parts) != 2:
                continue
            rest = parts[1]
            obj = get_object_from_sequence(rest)
            if obj != object_name:
                continue
            subject = parts[0]
            seq_name = rest.rsplit('_cam', 1)[0]

            # Get onset start
            key = f"{subject}__{seq_name}"
            clip_start = onset[key].get('clip_start', onset[key].get('start', 0)) if key in onset else 0

            # Extract cam_id from _camN suffix
            cam_id = 1  # default
            if '_cam' in rest:
                try:
                    cam_id = int(rest.rsplit('_cam', 1)[1])
                except ValueError:
                    cam_id = 1
            seq_metadata.append((subject, seq_name, f'haptic_cam{cam_id}'))
            frame_offsets.append(clip_start)
        hand_verts_list.extend(haptic_data)

    n_haptic = len(hand_verts_list) - n_hawor  # entries [n_hawor:] are HaPTIC

    if not hand_verts_list:
        print(f"  ⚠️  No MANO cache for '{object_name}' ({label})")
        return None, None

    total_frames = sum(len(hv) for hv in hand_verts_list)
    total_seqs = len(hand_verts_list)

    # Step 4: Load ARCTIC GT object poses + camera extrinsics + GT MANO per sequence
    obj_poses = []
    cam_extrinsics = []
    gt_mano_data = []  # for centroid alignment
    gt_mano_cache = {}  # cache per (subject, seq_name) to avoid reloading

    for i, (subject, seq_name, src_type) in enumerate(seq_metadata):
        # Object pose (same for both hawor/haptic)
        obj_rots, obj_trans = load_arctic_object_poses(subject, seq_name)
        obj_poses.append((obj_rots, obj_trans) if obj_rots is not None else None)

        # Camera extrinsics
        R_cam_ext, T_cam_ext = None, None
        if src_type == 'hawor':
            # HaWoR uses egocam (per-frame)
            R_cam_ext, T_cam_ext = load_arctic_egocam(subject, seq_name)
            cam_extrinsics.append((R_cam_ext, T_cam_ext) if R_cam_ext is not None else None)
        elif src_type.startswith('haptic_cam'):
            # HaPTIC uses 3rd-person static camera
            cam_id = int(src_type.split('cam')[1])
            R_cam_ext, T_cam_ext = load_arctic_3rd_cam(subject, cam_id)
            cam_extrinsics.append((R_cam_ext, T_cam_ext) if R_cam_ext is not None else None)
        else:
            cam_extrinsics.append(None)

        # GT MANO for centroid alignment
        cache_key = (subject, seq_name)
        if cache_key not in gt_mano_cache:
            gt_mano_cache[cache_key] = load_gt_mano_verts(subject, seq_name)
        gt_r, gt_l = gt_mano_cache[cache_key]

        if gt_r is not None and R_cam_ext is not None:
            gt_mano_data.append((gt_r, gt_l, R_cam_ext, T_cam_ext))
        else:
            gt_mano_data.append(None)

    n_with_pose = sum(1 for p in obj_poses if p is not None)
    n_with_gt = sum(1 for g in gt_mano_data if g is not None)
    print(f"    Loaded {n_with_pose}/{total_seqs} sequences with ARCTIC GT poses")
    print(f"    Loaded {n_with_gt}/{total_seqs} sequences with GT MANO (centroid alignment)")

    # Step 5: Accumulate contact labels on mesh vertices
    # When merging both sources, accumulate each independently then take
    # per-vertex max — each source uses its OWN frame count as denominator,
    # so the contact signal is not diluted by the other source's frames.
    if effective_source == 'both' and n_hawor > 0 and n_haptic > 0:
        # Split data by source
        hawor_hvl = hand_verts_list[:n_hawor]
        hawor_op = obj_poses[:n_hawor]
        hawor_ce = cam_extrinsics[:n_hawor]
        hawor_fo = frame_offsets[:n_hawor]
        hawor_gm = gt_mano_data[:n_hawor]

        haptic_hvl = hand_verts_list[n_hawor:]
        haptic_op = obj_poses[n_hawor:]
        haptic_ce = cam_extrinsics[n_hawor:]
        haptic_fo = frame_offsets[n_hawor:]
        haptic_gm = gt_mano_data[n_hawor:]

        print(f"    Per-source accumulation: HaWoR={len(hawor_hvl)} seqs, "
              f"HaPTIC={len(haptic_hvl)} seqs")

        contact_freq_hawor = accumulate_contact_labels(
            mesh_verts_m, mesh.faces, hawor_hvl, pred_threshold,
            obj_poses=hawor_op, cam_extrinsics=hawor_ce,
            frame_offsets=hawor_fo, gt_mano_data=hawor_gm
        )
        contact_freq_haptic = accumulate_contact_labels(
            mesh_verts_m, mesh.faces, haptic_hvl, pred_threshold,
            obj_poses=haptic_op, cam_extrinsics=haptic_ce,
            frame_offsets=haptic_fo, gt_mano_data=haptic_gm
        )

        # Merge: per-vertex max — overlapping regions take higher value,
        # missing regions from one source get filled by the other
        contact_freq = np.maximum(contact_freq_hawor, contact_freq_haptic)

        h_max = contact_freq_hawor.max()
        p_max = contact_freq_haptic.max()
        print(f"    HaWoR  max={h_max:.4f}, >0.1: {(contact_freq_hawor>0.1).mean():.1%}")
        print(f"    HaPTIC max={p_max:.4f}, >0.1: {(contact_freq_haptic>0.1).mean():.1%}")
        print(f"    Merged max={contact_freq.max():.4f}, >0.1: {(contact_freq>0.1).mean():.1%}")
    else:
        contact_freq = accumulate_contact_labels(
            mesh_verts_m, mesh.faces, hand_verts_list, pred_threshold,
            obj_poses=obj_poses, cam_extrinsics=cam_extrinsics,
            frame_offsets=frame_offsets, gt_mano_data=gt_mano_data
        )

    n_contact_verts = np.sum(contact_freq > 0)
    if n_contact_verts == 0:
        print(f"  ⚠️  No contact detected for '{object_name}' ({label})")
        return None, None

    # Step 6: Sample point cloud and transfer labels
    # Scale mesh to meters for point cloud sampling
    import trimesh as _trimesh
    mesh_m = _trimesh.Trimesh(vertices=mesh_verts_m, faces=mesh.faces)
    points, normals, human_prior = transfer_labels_to_pointcloud(mesh_m, contact_freq)

    # Step 7: Compute force center (centroid of high-contact region)
    high_contact = contact_freq > 0.1
    if np.any(high_contact):
        force_center = np.mean(mesh_verts_m[high_contact], axis=0).astype(np.float32)
    else:
        any_contact = contact_freq > 0
        force_center = np.mean(mesh_verts_m[any_contact], axis=0).astype(np.float32)

    print(f"  ✅ {object_name} [{label}]: {total_seqs} seqs, {total_frames} frames, "
          f"{n_contact_verts}/{len(mesh.vertices)} contact verts, "
          f"coverage={np.mean(human_prior > 0):.1%}")

    result = {
        'point_cloud': points,
        'normals': normals,
        'human_prior': human_prior,
        'robot_gt': np.zeros(N_POINTS, dtype=np.float32),
        'force_center': force_center,
    }
    return result, effective_source


def main():
    parser = argparse.ArgumentParser(description="Video → Training Data Bridge")
    parser.add_argument("--source", choices=['auto', 'hawor', 'haptic', 'both'],
                        default='auto',
                        help="MANO cache source. 'auto' picks best per-object: "
                             "haptic(3rd-person) > hawor(1st-person), merge if both exist")
    parser.add_argument("--mesh_provider", default='arctic',
                        choices=['arctic', 'sam3d'],
                        help="Mesh source (default: arctic)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Contact distance threshold in mm (default: 30mm)")
    parser.add_argument("--objects", nargs='+', default=None,
                        help="Specific objects to process (default: all)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: data_hub/training_m5)")
    args = parser.parse_args()

    # --threshold is in mm; convert to meters for internal use
    if args.threshold is not None:
        pred_th = args.threshold / 1000.0  # mm → m
    else:
        pred_th = config.PRED_CONTACT_TH  # already in meters (0.030 = 30mm)
    out_dir = args.out_dir or os.path.join(config.DATA_HUB, 'training_m5')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Video → Training Data Bridge")
    print(f"  Source:     {args.source}")
    print(f"  Mesh:       {args.mesh_provider}")
    print(f"  Threshold:  {pred_th * 1000:.0f}mm")
    print(f"  Output:     {out_dir}")
    print("=" * 60)

    provider = get_provider(args.mesh_provider)
    objects = args.objects or OBJECTS

    success = 0
    source_stats = defaultdict(int)  # track which source was used per object

    # Dual output directories
    hp_dir = config.HUMAN_PRIOR_DIR  # data_hub/human_prior
    os.makedirs(hp_dir, exist_ok=True)

    for obj_name in objects:
        result, eff_source = process_object(obj_name, args.source, provider, pred_th)
        if result is None:
            continue

        import h5py

        # Output 1: training_m5/ — full training data (for build_dataset.py → train.py)
        out_path = os.path.join(out_dir, f"video_{obj_name}.hdf5")
        with h5py.File(out_path, 'w') as f:
            for key, val in result.items():
                f.create_dataset(key, data=val, compression='gzip')
            f.attrs['source'] = eff_source
            f.attrs['requested_source'] = args.source
            f.attrs['mesh_provider'] = args.mesh_provider
            f.attrs['contact_threshold'] = pred_th

        # Output 2: human_prior/ — for inference 7th channel input
        #   predictor.py reads: point_cloud (N,3), normals (N,3), human_prior (N,)
        hp_path = os.path.join(hp_dir, f"{obj_name}.hdf5")
        with h5py.File(hp_path, 'w') as f:
            f.create_dataset('point_cloud', data=result['point_cloud'], compression='gzip')
            f.create_dataset('normals', data=result['normals'], compression='gzip')
            f.create_dataset('human_prior', data=result['human_prior'], compression='gzip')
            f.create_dataset('force_center', data=result['force_center'], compression='gzip')
            f.attrs['source'] = f"video_{eff_source}"
        # Output 3: grasp_collection/ — mesh for sim grasp testing
        exported = provider.export_to_grasp_collection(obj_name)
        if exported:
            print(f"    → mesh exported to grasp_collection/")

        success += 1
        source_stats[eff_source] += 1

    print(f"\n{'=' * 60}")
    print(f"✅ Generated {success}/{len(objects)} training samples")
    if source_stats:
        print(f"   Source breakdown:")
        for src, cnt in sorted(source_stats.items()):
            print(f"     {src}: {cnt} objects")
    print(f"   Training data: {out_dir}")
    print(f"   Human prior:   {hp_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
