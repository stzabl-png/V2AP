"""
MegaSAM Intrinsics Loader
==========================
Utility to load camera intrinsics estimated by MegaSAM (DROID-SLAM + UniDepth)
for a given scene.  The intrinsics are jointly optimised over the entire video,
so they are significantly more accurate (~1-3%) than the per-frame RANSAC
estimate used by HaWoR or the diagonal heuristic used by HaPTIC.

Supported output layouts (checked in priority order):

  [NEW] batch_megasam.py v2 output (data_hub/ProcessedData/egocentric_depth/):
    data_hub/ProcessedData/egocentric_depth/{scene_name}/
        K.npy            — (3, 3) intrinsic matrix
        cam_c2w.npy      — (N, 4, 4) camera-to-world poses
        depth.npz        — depths (N, H, W) float32 [metres]
        meta.json        — {seq_id, n_frames, hw, calibrated_fx}

  [OLD] mega-sam/outputs/{scene_name}_droid.npz:
        intrinsic  : (3, 3) float32
        cam_c2w    : (N, 4, 4)
        depths     : (N, H, W)

  [OLD] mega-sam/reconstructions/{scene_name}/intrinsics.npy:
        (N, 4) = [fx, fy, cx, cy] per frame

Usage::
    from data.megasam_utils import load_megasam_K, load_megasam_K_fx

    K = load_megasam_K('egodex/slot_batteries/1')  # new format
    K = load_megasam_K('box_grab_01')              # old format
    fx = load_megasam_K_fx('egodex/fry_bread/0')   # float | None
"""

import os
import numpy as np

# ── Path to MegaSAM outputs (relative to this file's project root) ────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
MEGASAM_DIR = os.path.join(_PROJECT_DIR, 'mega-sam')
MEGASAM_OUTPUT_DIR = os.path.join(MEGASAM_DIR, 'outputs')
MEGASAM_RECON_DIR = os.path.join(MEGASAM_DIR, 'reconstructions')

# New-format base (batch_megasam.py v2)
_EGO_DEPTH_BASE = os.path.join(_PROJECT_DIR, 'data_hub', 'ProcessedData', 'egocentric_depth')


def load_megasam_K(scene_name: str) -> "np.ndarray | None":
    """Return the 3×3 intrinsic matrix K estimated by MegaSAM for *scene_name*.

    Lookup priority (first hit wins):
      1. NEW  data_hub/ProcessedData/egocentric_depth/{scene_name}/K.npy
      2. OLD  mega-sam/outputs/{scene_name}_droid.npz  →  'intrinsic' key
      3. OLD  mega-sam/reconstructions/{scene_name}/intrinsics.npy  →  median

    Returns ``None`` if no MegaSAM output is found for this scene.
    """
    # ── Priority 1: new batch_megasam v2 format ────────────────────────────────
    new_k_path = os.path.join(_EGO_DEPTH_BASE, scene_name, 'K.npy')
    if os.path.exists(new_k_path):
        try:
            K = np.load(new_k_path).astype(np.float64)  # (3, 3)
            assert K.shape == (3, 3), f"Unexpected K shape: {K.shape}"
            print(f"[megasam_utils] K loaded (new format): {new_k_path}")
            return K
        except Exception as e:
            print(f"[megasam_utils] Warning: could not load {new_k_path}: {e}")

    # ── Priority 2: old _droid.npz → 'intrinsic' key (3x3) ───────────────────
    droid_path = os.path.join(MEGASAM_OUTPUT_DIR, f'{scene_name}_droid.npz')
    if os.path.exists(droid_path):
        try:
            d = np.load(droid_path, allow_pickle=True)
            K = d['intrinsic'].astype(np.float64)  # (3, 3)
            assert K.shape == (3, 3), f"Unexpected intrinsic shape: {K.shape}"
            return K
        except Exception as e:
            print(f"[megasam_utils] Warning: could not load {droid_path}: {e}")

    # ── Priority 3: reconstructions/{scene}/intrinsics.npy → (N,4) median ────
    recon_path = os.path.join(MEGASAM_RECON_DIR, scene_name, 'intrinsics.npy')
    if os.path.exists(recon_path):
        try:
            intr = np.load(recon_path, allow_pickle=True)  # (N, 4)
            fx, fy, cx, cy = np.median(intr, axis=0)
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float64)
            return K
        except Exception as e:
            print(f"[megasam_utils] Warning: could not load {recon_path}: {e}")

    return None


def load_megasam_K_fx(scene_name: str) -> "float | None":
    """Return only the horizontal focal length *fx* from MegaSAM.

    Convenience wrapper for HaWoR which accepts a scalar ``img_focal``
    (pixels).  Returns ``None`` when MegaSAM has not been run for this scene,
    so callers can fall back to self-estimation without extra bookkeeping.
    """
    K = load_megasam_K(scene_name)
    if K is None:
        return None
    return float(K[0, 0])


def load_megasam_poses(scene_name: str) -> "np.ndarray | None":
    """Return per-frame camera-to-world poses (N, 4, 4) from MegaSAM.

    Checks new format (cam_c2w.npy) first, then old _droid.npz.
    Returns ``None`` if not available.
    """
    # New format: cam_c2w.npy
    new_poses_path = os.path.join(_EGO_DEPTH_BASE, scene_name, 'cam_c2w.npy')
    if os.path.exists(new_poses_path):
        try:
            return np.load(new_poses_path).astype(np.float64)  # (N, 4, 4)
        except Exception as e:
            print(f"[megasam_utils] Warning: could not load {new_poses_path}: {e}")

    # Old format: _droid.npz → 'cam_c2w'
    droid_path = os.path.join(MEGASAM_OUTPUT_DIR, f'{scene_name}_droid.npz')
    if os.path.exists(droid_path):
        try:
            d = np.load(droid_path, allow_pickle=True)
            return d['cam_c2w'].astype(np.float64)  # (N, 4, 4)
        except Exception as e:
            print(f"[megasam_utils] Warning: could not load poses from {droid_path}: {e}")
    return None


def load_megasam_depths(scene_name: str) -> "np.ndarray | None":
    """Return per-frame metric depth maps (N, H, W) from MegaSAM.

    Checks new format (depth.npz) first, then old _droid.npz.
    Returns ``None`` if not available.
    """
    # New format: depth.npz → 'depths' key
    new_depth_path = os.path.join(_EGO_DEPTH_BASE, scene_name, 'depth.npz')
    if os.path.exists(new_depth_path):
        try:
            d = np.load(new_depth_path)
            return d['depths'].astype(np.float32)  # (N, H, W)
        except Exception as e:
            print(f"[megasam_utils] Warning: could not load {new_depth_path}: {e}")

    # Old format: _droid.npz → 'depths'
    droid_path = os.path.join(MEGASAM_OUTPUT_DIR, f'{scene_name}_droid.npz')
    if os.path.exists(droid_path):
        try:
            d = np.load(droid_path, allow_pickle=True)
            return d['depths'].astype(np.float32)  # (N, H, W)
        except Exception as e:
            print(f"[megasam_utils] Warning: could not load depths from {droid_path}: {e}")
    return None


def K_as_haptic_intrinsics(K: np.ndarray) -> list:
    """Flatten 3×3 K to a 9-element float list compatible with HaPTIC's
    ``parse_det_seq(intrinsics=...)`` argument.

    HaPTIC does: ``np.array(intrinsics, dtype=np.float64).reshape(3, 3)``
    """
    return K.flatten().tolist()
