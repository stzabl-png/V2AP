#!/usr/bin/env python3
"""
ego_pipeline_smoke_test.py
===========================
Validates every step of the V2AP egocentric pipeline end-to-end on a single
sequence (egodex clean_surface/14).

Steps tested:
  0. Environment & weight checks (no GPU needed)
  1. MegaSAM   — depth + intrinsics   (mega_sam env)
  2. HaWoR     — MANO hand pose        (hawor env)
  3. Annotation — object mask          (interactive; smoke test checks output exists or skips)
  4. SAM3D     — object mesh           (sam3d-objects env)
  5. Scale     — estimate_obj_scale_ego (hawor env)
  6. FP        — batch_obj_pose_ego    (bundlesdf env)
  7. Contact   — batch_align_ego_mano_fp (bundlesdf env)

Usage:
  # Run all non-interactive checks (env + weights + output validation)
  python tools/eval/ego_pipeline_smoke_test.py

  # Actually execute each pipeline step (needs all envs + GPU, takes ~30min)
  python tools/eval/ego_pipeline_smoke_test.py --run

  # Run specific step only
  python tools/eval/ego_pipeline_smoke_test.py --run --step 1
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent.parent   # V2AP/
sys.path.insert(0, str(PROJECT))
import config as cfg

# ── Test sequence ─────────────────────────────────────────────────────────────
TEST_TASK   = "clean_surface"
TEST_EP     = "14"
TEST_SEQ_ID = f"{TEST_TASK}__{TEST_EP}"   # egodex convention: task__ep
TEST_VIDEO  = PROJECT / "data/egocentric/egodex/test" / TEST_TASK / f"{TEST_EP}.mp4"
TEST_OBJ    = TEST_TASK   # SAM3D / scale key for egocentric

# Expected output locations
EGO_DEPTH_DIR = Path(cfg.DATA_HUB) / "ProcessedData/egocentric_depth" / TEST_SEQ_ID
EGO_MANO_DIR  = Path(cfg.DATA_HUB) / "ProcessedData/mano/Egocentric/Egodex/egodex" / TEST_TASK
EGO_MASK_DIR  = Path(cfg.DATA_HUB) / "ProcessedData/obj_recon_input/egocentric" / TEST_TASK
EGO_MESH_DIR  = Path(cfg.DATA_HUB) / "ProcessedData/obj_meshes/egocentric" / TEST_TASK
EGO_POSE_DIR  = Path(cfg.EGO_POSE_DIR) / "egodex" / TEST_SEQ_ID
EGO_PRIOR_DIR = Path(cfg.HUMAN_PRIOR_DIR)

# ── Results tracking ──────────────────────────────────────────────────────────
results: list[tuple[str, bool, str]] = []   # (label, passed, detail)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m~\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = PASS if condition else FAIL
    print(f"  {icon}  {label}  {detail}")
    results.append((label, condition, detail))
    return condition


def skip(label: str, reason: str = "") -> None:
    print(f"  {SKIP}  {label}  (skipped: {reason})")
    results.append((label, None, reason))


def run_step(label: str, cmd: list, env_name: str | None = None,
             cwd: Path | None = None, timeout: int = 1800) -> bool:
    """Run a pipeline command in the given conda env."""
    print(f"\n  Running: {' '.join(cmd)}")
    full_cmd = cmd
    if env_name:
        full_cmd = ["conda", "run", "-n", env_name, "--no-capture-output"] + cmd
    env = {**os.environ, "XFORMERS_DISABLED": "1"}
    t0 = time.time()
    try:
        proc = subprocess.run(full_cmd, cwd=str(cwd or PROJECT),
                              timeout=timeout, env=env)
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        ok = False
    except Exception as e:
        print(f"  ERROR: {e}")
        ok = False
    elapsed = time.time() - t0
    return check(label, ok, f"({elapsed:.0f}s)")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 0: Environment & weights
# ══════════════════════════════════════════════════════════════════════════════

def phase0_env():
    print("\n[Phase 0] Environment & Weights")

    check("Test video exists", TEST_VIDEO.exists(), str(TEST_VIDEO))

    # Conda envs
    for env in ["mega_sam", "hawor", "bundlesdf"]:
        env_path = Path(os.environ.get("CONDA_PREFIX", "")).parent.parent / "envs" / env
        # Fallback: check via conda run
        r = subprocess.run(["conda", "run", "-n", env, "python", "-c", "import sys; print(sys.version)"],
                           capture_output=True, timeout=15)
        check(f"conda env: {env}", r.returncode == 0)

    # Model weights
    megasam_ckpt = Path(cfg.MEGASAM_DIR) / "checkpoints/megasam_final.pth"
    check("MegaSAM checkpoint", megasam_ckpt.exists(), str(megasam_ckpt))

    hawor_ckpt = Path(cfg.HAWOR_DIR) / "weights/hawor/checkpoints/hawor.ckpt"
    check("HaWoR checkpoint", hawor_ckpt.exists(), str(hawor_ckpt))

    fp_weights = Path(cfg.FP_ROOT) / "weights/2023-10-28-18-33-37/model_best.pth"
    check("FoundationPose weights", fp_weights.exists(), str(fp_weights))

    depthpro_ckpt = Path(cfg.DEPTHPRO_DIR) / "checkpoints/depth_pro.pt"
    check("DepthPro checkpoint", depthpro_ckpt.exists(), str(depthpro_ckpt))

    mano_pkl = Path(cfg.HAPTIC_MANO_DIR) / "MANO_RIGHT.pkl"
    check("MANO_RIGHT.pkl", mano_pkl.exists(), str(mano_pkl))

    # Registry
    registry = PROJECT / "tools/egodex_sequence_registry.json"
    check("egodex_sequence_registry.json", registry.exists())
    if registry.exists():
        reg = json.loads(registry.read_text())
        check(f"Test task '{TEST_TASK}' in registry",
              any(v.get("task") == TEST_TASK for v in reg.values()
                  if isinstance(v, dict)) or TEST_TASK in str(reg))


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: MegaSAM
# ══════════════════════════════════════════════════════════════════════════════

def phase1_megasam(run: bool):
    print("\n[Phase 1] MegaSAM — depth + intrinsics")

    depth_npz = EGO_DEPTH_DIR / "depth.npz"
    meta_json = EGO_DEPTH_DIR / "meta.json"

    if depth_npz.exists() and meta_json.exists():
        # Validate existing output
        import numpy as np
        d = np.load(str(depth_npz))["depths"]
        check("depth.npz shape valid", d.ndim == 3 and d.shape[0] > 0,
              f"shape={d.shape}")
        meta = json.loads(meta_json.read_text())
        scale = meta.get("depth_scale", None)
        check("depth_scale=1.2 in meta.json", scale == 1.20, f"depth_scale={scale}")
        K = np.load(str(EGO_DEPTH_DIR / "K.npy"))
        check("K.npy is 3×3", K.shape == (3, 3), f"shape={K.shape}")
        cam = np.load(str(EGO_DEPTH_DIR / "cam_c2w.npy"))
        check("cam_c2w.npy shape valid", cam.ndim == 3 and cam.shape[-2:] == (4, 4),
              f"shape={cam.shape}")
        return

    if not run:
        skip("MegaSAM execution", "pass --run to execute")
        return

    run_step("MegaSAM step 1",
             ["python", "pipeline/ego/batch_megasam.py",
              "--dataset", "egodex", "--seq-ids", TEST_SEQ_ID],
             env_name="mega_sam", timeout=1800)

    check("depth.npz produced", depth_npz.exists())
    check("depth_scale=1.2 in meta.json",
          json.loads(meta_json.read_text()).get("depth_scale") == 1.20
          if meta_json.exists() else False)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: HaWoR
# ══════════════════════════════════════════════════════════════════════════════

def phase2_hawor(run: bool):
    print("\n[Phase 2] HaWoR — MANO hand pose")

    mano_npz = EGO_MANO_DIR / f"{TEST_EP}.npz"

    if mano_npz.exists():
        import numpy as np
        d = np.load(str(mano_npz))
        check("HaWoR NPZ has right_verts", "right_verts" in d or "pred_trans" in d,
              f"keys={list(d.keys())[:5]}")
        return

    if not run:
        skip("HaWoR execution", "pass --run to execute")
        return

    if not (EGO_DEPTH_DIR / "K.npy").exists():
        skip("HaWoR execution", "MegaSAM K.npy missing — run phase 1 first")
        return

    run_step("HaWoR step 2",
             ["python", "pipeline/ego/batch_hawor.py",
              "--dataset", "egodex", "--seq", f"{TEST_TASK}/{TEST_EP}"],
             env_name="hawor", timeout=1800)

    check("HaWoR NPZ produced", mano_npz.exists())


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Annotation (mask)
# ══════════════════════════════════════════════════════════════════════════════

def phase3_annotation():
    print("\n[Phase 3] Object mask annotation")

    mask_png  = EGO_MASK_DIR / "0.png"
    image_png = EGO_MASK_DIR / "image.png"

    if mask_png.exists() and image_png.exists():
        check("mask 0.png exists", True, str(mask_png))
        check("reference image.png exists", True, str(image_png))
    else:
        skip("Mask annotation",
             "interactive — run: python tools/anno/annotate_ego_masks.py --dataset egodex")
        print(f"    Expected at: {EGO_MASK_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: SAM3D
# ══════════════════════════════════════════════════════════════════════════════

def phase4_sam3d(run: bool):
    print("\n[Phase 4] SAM3D — object mesh")

    mesh_ply = EGO_MESH_DIR / "mesh.ply"

    if mesh_ply.exists():
        size = mesh_ply.stat().st_size
        check("mesh.ply exists and non-empty", size > 1000, f"{size/1024:.0f} KB")
        meta = EGO_MESH_DIR / "meta.json"
        check("SAM3D meta.json exists", meta.exists())
        return

    if not run:
        skip("SAM3D execution", "pass --run to execute")
        return

    if not (EGO_MASK_DIR / "0.png").exists():
        skip("SAM3D execution", "mask 0.png missing — run phase 3 first")
        return

    run_step("SAM3D step 4",
             ["python", "tools/batch/batch_sam3d_recon.py",
              "--datasets", "egodex", "--obj", TEST_TASK, "--limit", "1"],
             env_name="sam3d-objects", timeout=1800)

    check("mesh.ply produced", mesh_ply.exists())


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: Scale estimation
# ══════════════════════════════════════════════════════════════════════════════

def phase5_scale(run: bool):
    print("\n[Phase 5] Object scale estimation")

    scale_json = EGO_MESH_DIR / "scale.json"

    if scale_json.exists():
        s = json.loads(scale_json.read_text())
        sf = s.get("scale_factor", None)
        check("scale.json has scale_factor", sf is not None, f"scale_factor={sf}")
        check("scale_factor > 0", sf is not None and sf > 0)
        return

    if not run:
        skip("Scale estimation", "pass --run to execute")
        return

    if not (EGO_MESH_DIR / "mesh.ply").exists():
        skip("Scale estimation", "mesh.ply missing — run phase 4 first")
        return

    run_step("Scale step 5",
             ["python", "pipeline/object/estimate_obj_scale_ego.py",
              "--obj", TEST_TASK],
             env_name="hawor", timeout=600)

    check("scale.json produced", scale_json.exists())


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6: FoundationPose
# ══════════════════════════════════════════════════════════════════════════════

def phase6_fp(run: bool):
    print("\n[Phase 6] FoundationPose — object pose")

    pose_dir = EGO_POSE_DIR / "ob_in_cam"

    if pose_dir.exists():
        txts = list(pose_dir.glob("*.txt"))
        check("ob_in_cam/*.txt exists", len(txts) > 0, f"{len(txts)} frames")
        if txts:
            import numpy as np
            T = np.loadtxt(str(txts[0]))
            check("pose matrix is 4×4", T.shape == (4, 4))
        return

    if not run:
        skip("FoundationPose execution", "pass --run to execute")
        return

    for req, name in [(EGO_DEPTH_DIR / "K.npy", "MegaSAM K.npy"),
                      (EGO_MESH_DIR / "scale.json", "scale.json"),
                      (EGO_MASK_DIR / "0.png", "mask 0.png")]:
        if not req.exists():
            skip("FoundationPose execution", f"{name} missing")
            return

    run_step("FP step 6",
             ["python", "tools/batch/batch_obj_pose_ego.py",
              "--dataset", "egodex", "--seq", TEST_SEQ_ID, "--limit", "1"],
             env_name="bundlesdf", timeout=3600)

    check("ob_in_cam/ produced", pose_dir.exists())


# ══════════════════════════════════════════════════════════════════════════════
# Phase 7: Contact + HumanPrior
# ══════════════════════════════════════════════════════════════════════════════

def phase7_contact(run: bool):
    print("\n[Phase 7] Contact extraction + HumanPrior")

    # ego_mano symlink / copy check
    ego_mano_path = Path(cfg.DATA_HUB) / "ProcessedData/ego_mano/egodex" / TEST_TASK / f"{TEST_EP}.npz"
    hawor_path    = EGO_MANO_DIR / f"{TEST_EP}.npz"

    if not ego_mano_path.exists() and hawor_path.exists():
        # Create symlink so alignment script can find it
        ego_mano_path.parent.mkdir(parents=True, exist_ok=True)
        ego_mano_path.symlink_to(hawor_path)
        print(f"  Created symlink: ego_mano/egodex/{TEST_TASK}/{TEST_EP}.npz → HaWoR output")

    check("ego_mano NPZ accessible", ego_mano_path.exists() or hawor_path.exists())

    prior_h5 = EGO_PRIOR_DIR / f"{TEST_TASK}.hdf5"
    if prior_h5.exists():
        import numpy as np, h5py
        with h5py.File(str(prior_h5), "r") as f:
            keys = list(f.keys())
        check("HumanPrior HDF5 has point_cloud", "point_cloud" in keys, f"keys={keys}")
        check("HumanPrior HDF5 has human_prior", "human_prior" in keys)
        return

    if not run:
        skip("Contact + prior execution", "pass --run to execute")
        return

    for req, name in [(ego_mano_path, "ego_mano NPZ"),
                      (EGO_POSE_DIR / "ob_in_cam", "FP poses")]:
        if not req.exists():
            skip("Contact step", f"{name} missing — run prior steps first")
            return

    run_step("Contact step 7",
             ["python", "pipeline/ego/batch_align_ego_mano_fp.py",
              "--dataset", "egodex", "--obj", TEST_TASK],
             env_name="bundlesdf", timeout=600)

    check("HumanPrior HDF5 produced", prior_h5.exists())


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "="*60)
    print("  EGO PIPELINE SMOKE TEST SUMMARY")
    print("="*60)
    passed = sum(1 for _, ok, _ in results if ok is True)
    failed = sum(1 for _, ok, _ in results if ok is False)
    skipped = sum(1 for _, ok, _ in results if ok is None)
    for label, ok, detail in results:
        if ok is True:
            icon = PASS
        elif ok is False:
            icon = FAIL
        else:
            icon = SKIP
        print(f"  {icon}  {label}")
    print(f"\n  {passed} passed  {failed} failed  {skipped} skipped")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="V2AP Ego Pipeline smoke test")
    parser.add_argument("--run",  action="store_true",
                        help="Actually execute each pipeline step (needs GPU + all envs)")
    parser.add_argument("--step", type=int, default=0,
                        help="Run only this step (0=all)")
    args = parser.parse_args()

    print("="*60)
    print("  V2AP Ego Pipeline Smoke Test")
    print(f"  Test sequence: {TEST_TASK}/{TEST_EP}")
    print(f"  Project: {PROJECT}")
    print(f"  Mode: {'EXECUTE' if args.run else 'CHECK ONLY (pass --run to execute)'}")
    print("="*60)

    steps = {
        0: lambda: phase0_env(),
        1: lambda: phase1_megasam(args.run),
        2: lambda: phase2_hawor(args.run),
        3: lambda: phase3_annotation(),
        4: lambda: phase4_sam3d(args.run),
        5: lambda: phase5_scale(args.run),
        6: lambda: phase6_fp(args.run),
        7: lambda: phase7_contact(args.run),
    }

    if args.step:
        steps.get(args.step, lambda: print(f"Unknown step {args.step}"))()
    else:
        for fn in steps.values():
            fn()

    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
