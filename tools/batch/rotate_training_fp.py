#!/usr/bin/env python3
"""
rotate_training_fp.py — training_fp HP 与 SAM3D 相同 Rx(+90°) → train_fp_rotated/

读取: data_hub/ProcessedData/training_fp/{subdir}/{obj}.hdf5
写入: data_hub/ProcessedData/train_fp_rotated/{subdir}/{obj}.hdf5

旋转（与 tools/rotate_sam3d_mesh.py 一致）:
  x' = x,  y' = -z,  z' = y   (v' = R @ v)

旋转字段: point_cloud, normals（若有）, force_center（若有）
不变: human_prior, robot_gt 及其它标量数据集

用法:
  python3 tools/rotate_training_fp.py --obj A01026
  python3 tools/rotate_training_fp.py --all
  python3 tools/rotate_training_fp.py --all --skip-existing
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_FP_IN = os.path.join(PROJ, "data_hub", "ProcessedData", "training_fp")
TRAINING_FP_OUT = os.path.join(PROJ, "data_hub", "ProcessedData", "train_fp_rotated")

EULER_XYZ_DEG = [90.0, 0.0, 0.0]
R_FIX = Rotation.from_euler("x", EULER_XYZ_DEG[0], degrees=True).as_matrix().astype(np.float64)

VECTOR_KEYS = ("point_cloud", "normals", "force_center")


def discover_jobs(root: str = TRAINING_FP_IN) -> list[tuple[str, str, str]]:
    """Return list of (subdir, obj_id, src_path)."""
    jobs: list[tuple[str, str, str]] = []
    if not os.path.isdir(root):
        return jobs
    for sub in sorted(os.listdir(root)):
        sub_dir = os.path.join(root, sub)
        if not os.path.isdir(sub_dir):
            continue
        for name in sorted(os.listdir(sub_dir)):
            if not name.endswith(".hdf5"):
                continue
            obj_id = name[:-5]
            src = os.path.join(sub_dir, name)
            jobs.append((sub, obj_id, src))
    return jobs


def _apply_R(points: np.ndarray, R: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim == 1 and p.shape == (3,):
        return (R @ p).astype(np.float32)
    if p.ndim == 2 and p.shape[1] == 3:
        return (R @ p.T).T.astype(np.float32)
    raise ValueError(f"expected (3,) or (N,3), got {p.shape}")


def _extents_cm(pc: np.ndarray) -> list[float]:
    ext = pc.max(axis=0) - pc.min(axis=0)
    return [round(float(ext[i] * 100), 4) for i in range(3)]


def process_one(
    subdir: str,
    obj_id: str,
    src_path: str,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    out_path = os.path.join(TRAINING_FP_OUT, subdir, f"{obj_id}.hdf5")
    if os.path.isfile(out_path) and not force:
        return {"status": "skipped", "subdir": subdir, "obj_id": obj_id}

    with h5py.File(src_path, "r") as fin:
        pc_before = fin["point_cloud"][()].astype(np.float32)
        data: dict[str, np.ndarray] = {}
        for key in fin.keys():
            data[key] = fin[key][()]

        attrs = {k: fin.attrs[k] for k in fin.attrs.keys()}

    rotated: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        if key in VECTOR_KEYS:
            rotated[key] = _apply_R(arr, R_FIX)
        else:
            rotated[key] = np.asarray(arr)

    pc_after = rotated["point_cloud"]
    record = {
        "subdir": subdir,
        "obj_id": obj_id,
        "source": os.path.relpath(src_path, PROJ),
        "output": os.path.relpath(out_path, PROJ),
        "extents_cm_before": _extents_cm(pc_before),
        "extents_cm_after": _extents_cm(pc_after),
        "n_points": int(len(pc_after)),
    }

    if dry_run:
        record["status"] = "dry_run"
        return record

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as fout:
        for key, arr in rotated.items():
            fout.create_dataset(key, data=arr, compression="gzip")
        for k, v in attrs.items():
            fout.attrs[k] = v
        fout.attrs["rotation_euler_xyz_deg"] = EULER_XYZ_DEG
        fout.attrs["rotation_matrix"] = R_FIX.tolist()
        fout.attrs["rotation_convention"] = "v' = R @ v  (+90° about +X, same as rotated_mesh)"
        fout.attrs["rotated_from"] = os.path.relpath(src_path, PROJ)
        fout.attrs["rotated_utc"] = datetime.now(timezone.utc).isoformat()

    record["status"] = "ok"
    return record


def write_manifest(entries: list[dict], path: str) -> None:
    manifest = {
        "rotation_euler_xyz_deg": EULER_XYZ_DEG,
        "rotation_matrix": R_FIX.tolist(),
        "input_root": os.path.relpath(TRAINING_FP_IN, PROJ),
        "output_root": os.path.relpath(TRAINING_FP_OUT, PROJ),
        "n_files": len(entries),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": entries,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate training_fp HDF5 (+90° X)")
    parser.add_argument("--all", action="store_true", help="all HDF5 under training_fp/")
    parser.add_argument("--obj", default=None, help="substring filter on object id")
    parser.add_argument("--subdir", default=None, help="only e.g. oakink or dexycb")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.all and not args.obj:
        parser.error("specify --all and/or --obj")

    jobs = discover_jobs()
    if args.subdir:
        jobs = [j for j in jobs if j[0] == args.subdir]
    if args.obj:
        jobs = [j for j in jobs if args.obj in j[1]]

    if not jobs:
        print("No HDF5 found.")
        return

    force = args.force
    print(f"Rotation: Rx(+90°)  euler={EULER_XYZ_DEG}")
    print(f"Input : {TRAINING_FP_IN}")
    print(f"Output: {TRAINING_FP_OUT}")
    print(f"Jobs  : {len(jobs)}  dry_run={args.dry_run}  force={force}\n")

    entries: list[dict] = []
    ok = skip = fail = 0
    for i, (sub, oid, src) in enumerate(jobs, 1):
        out_path = os.path.join(TRAINING_FP_OUT, sub, f"{oid}.hdf5")
        if args.skip_existing and os.path.isfile(out_path) and not force:
            skip += 1
            continue
        print(f"[{i}/{len(jobs)}] {sub}/{oid}")
        try:
            rec = process_one(sub, oid, src, force=force, dry_run=args.dry_run)
            entries.append(rec)
            st = rec.get("status")
            if st == "ok":
                ok += 1
                print(f"  -> {rec['output']}")
                print(f"     ext cm {rec['extents_cm_before']} -> {rec['extents_cm_after']}")
            elif st == "dry_run":
                ok += 1
                print(f"  dry-run {rec['extents_cm_before']} -> {rec['extents_cm_after']}")
            elif st == "skipped":
                skip += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            fail += 1

    if not args.dry_run and entries:
        manifest_path = os.path.join(TRAINING_FP_OUT, "manifest.json")
        write_manifest(entries, manifest_path)
        print(f"\nManifest: {manifest_path}")

    print(f"\nDone: ok={ok}  skipped={skip}  failed={fail}")


if __name__ == "__main__":
    main()
