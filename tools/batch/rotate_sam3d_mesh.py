#!/usr/bin/env python3
"""
rotate_sam3d_mesh.py — SAM3D mesh 绕 X 轴 +90° → rotated_mesh/

将 data_hub/meshes/SAM3DMesh/meshes/{dataset}/{obj}/mesh.ply
写入 data_hub/meshes/SAM3DMesh/rotated_mesh/{dataset}/{obj}/mesh.ply

旋转（右手系，绕 +X 转 90°）:
  x' = x
  y' = -z
  z' = y

用法:
  python3 tools/rotate_sam3d_mesh.py --obj ycb_dex_19 --dataset ycb
  python3 tools/rotate_sam3d_mesh.py --all --datasets oakink,ycb
  python3 tools/rotate_sam3d_mesh.py --all --skip-existing
  python3 tools/rotate_sam3d_mesh.py --all --force
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM3D_ROOT = os.path.join(PROJ, "data_hub", "meshes", "SAM3DMesh")
MESHES_IN = os.path.join(SAM3D_ROOT, "meshes")
ROTATED_OUT = os.path.join(SAM3D_ROOT, "rotated_mesh")

# 固定：全库统一 +90° 绕 X（与 vis 诊断一致）
EULER_XYZ_DEG = [90.0, 0.0, 0.0]
R_FIX = Rotation.from_euler("x", EULER_XYZ_DEG[0], degrees=True).as_matrix().astype(np.float64)

DEFAULT_DATASETS = ("oakink", "ycb", "egodex")


@dataclass
class MeshJob:
    dataset: str
    obj_id: str
    src_path: str


def discover_jobs(
    datasets: list[str],
    obj_filter: str | None = None,
) -> list[MeshJob]:
    jobs: list[MeshJob] = []
    for ds in datasets:
        ds_dir = os.path.join(MESHES_IN, ds)
        if not os.path.isdir(ds_dir):
            print(f"  [skip] missing {ds_dir}")
            continue
        for obj_id in sorted(os.listdir(ds_dir)):
            if obj_filter and obj_filter not in obj_id:
                continue
            obj_dir = os.path.join(ds_dir, obj_id)
            if not os.path.isdir(obj_dir):
                continue
            src = os.path.join(obj_dir, "mesh.ply")
            if os.path.isfile(src):
                jobs.append(MeshJob(ds, obj_id, src))
    return jobs


def _extents_cm(verts: np.ndarray) -> list[float]:
    ext = verts.max(axis=0) - verts.min(axis=0)
    return [round(float(ext[i] * 100), 4) for i in range(3)]


def _long_axis(verts: np.ndarray, n: int = 8000) -> list[float]:
    v = np.asarray(verts, dtype=np.float64)
    if len(v) > n:
        idx = np.random.default_rng(0).choice(len(v), n, replace=False)
        v = v[idx]
    c = v - v.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return [round(float(x), 4) for x in vt[0]]


def rotate_mesh(mesh: trimesh.Trimesh, R: np.ndarray) -> trimesh.Trimesh:
    """返回新 mesh：顶点/法线左乘 R。"""
    m = mesh.copy()
    v = np.asarray(m.vertices, dtype=np.float64)
    m.vertices = (R @ v.T).T.astype(np.float64)

    if hasattr(m, "vertex_normals") and m.vertex_normals is not None and len(m.vertex_normals):
        n = np.asarray(m.vertex_normals, dtype=np.float64)
        m.vertex_normals = (R @ n.T).T.astype(np.float64)
    return m


def out_paths(job: MeshJob) -> tuple[str, str, str]:
    out_dir = os.path.join(ROTATED_OUT, job.dataset, job.obj_id)
    ply = os.path.join(out_dir, "mesh.ply")
    meta = os.path.join(out_dir, "rotation_meta.json")
    return out_dir, ply, meta


def process_one(
    job: MeshJob,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    out_dir, out_ply, meta_path = out_paths(job)
    if os.path.isfile(out_ply) and not force:
        return {"status": "skipped", "obj": job.obj_id, "dataset": job.dataset}

    mesh_in = trimesh.load(job.src_path, force="mesh", process=False)
    v_before = np.asarray(mesh_in.vertices, dtype=np.float64)
    mesh_out = rotate_mesh(mesh_in, R_FIX)
    v_after = np.asarray(mesh_out.vertices, dtype=np.float64)

    record = {
        "dataset": job.dataset,
        "obj_id": job.obj_id,
        "source_mesh": os.path.relpath(job.src_path, PROJ),
        "output_mesh": os.path.relpath(out_ply, PROJ),
        "rotation": {
            "euler_xyz_deg": EULER_XYZ_DEG,
            "matrix": R_FIX.tolist(),
            "convention": "v' = R @ v  (column vectors, right-handed, +90° about +X)",
        },
        "extents_cm_before": _extents_cm(v_before),
        "extents_cm_after": _extents_cm(v_after),
        "long_axis_before": _long_axis(v_before),
        "long_axis_after": _long_axis(v_after),
        "n_verts": int(len(mesh_out.vertices)),
        "n_faces": int(len(mesh_out.faces)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    if dry_run:
        record["status"] = "dry_run"
        return record

    os.makedirs(out_dir, exist_ok=True)
    mesh_out.export(out_ply)
    with open(meta_path, "w") as f:
        json.dump(record, f, indent=2)
    record["status"] = "ok"
    return record


def write_manifest(entries: list[dict], path: str) -> None:
    manifest = {
        "rotation_euler_xyz_deg": EULER_XYZ_DEG,
        "rotation_matrix": R_FIX.tolist(),
        "input_root": os.path.relpath(MESHES_IN, PROJ),
        "output_root": os.path.relpath(ROTATED_OUT, PROJ),
        "n_objects": len(entries),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objects": entries,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rotate SAM3D meshes +90° about X into SAM3DMesh/rotated_mesh/",
    )
    parser.add_argument("--all", action="store_true", help="process all objects under datasets")
    parser.add_argument("--obj", default=None, help="substring filter on object id")
    parser.add_argument(
        "--dataset",
        default=None,
        help="single dataset (with --obj); for --all use --datasets",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="comma-separated: oakink,ycb,egodex",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing rotated mesh.ply")
    parser.add_argument("--skip-existing", action="store_true", help="skip if output exists")
    parser.add_argument("--dry-run", action="store_true", help="stats only, no write")
    args = parser.parse_args()

    if args.all or args.obj:
        datasets = (
            [args.dataset]
            if args.dataset
            else [d.strip() for d in args.datasets.split(",") if d.strip()]
        )
    else:
        parser.error("specify --all and/or --obj")

    jobs = discover_jobs(datasets, args.obj)
    if not jobs:
        print("No mesh.ply found.")
        return

    if args.skip_existing and not args.force:
        force = False
    else:
        force = args.force

    print(f"Rotation: Rx(+90°)  euler={EULER_XYZ_DEG}")
    print(f"Input : {MESHES_IN}")
    print(f"Output: {ROTATED_OUT}")
    print(f"Jobs  : {len(jobs)}  dry_run={args.dry_run}  force={force}\n")

    entries: list[dict] = []
    ok = skip = fail = 0
    for i, job in enumerate(jobs, 1):
        if args.skip_existing:
            out_ply = out_paths(job)[1]
            if os.path.isfile(out_ply) and not force:
                skip += 1
                continue
        print(f"[{i}/{len(jobs)}] {job.dataset}/{job.obj_id}")
        try:
            rec = process_one(job, force=force, dry_run=args.dry_run)
            entries.append(rec)
            st = rec.get("status", "?")
            if st == "ok":
                ok += 1
                print(f"  -> {rec['output_mesh']}")
                print(f"     ext cm {rec['extents_cm_before']} -> {rec['extents_cm_after']}")
            elif st == "dry_run":
                ok += 1
                print(f"  dry-run ext {rec['extents_cm_before']} -> {rec['extents_cm_after']}")
            elif st == "skipped":
                skip += 1
            else:
                ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            fail += 1

    if not args.dry_run and entries:
        manifest_path = os.path.join(ROTATED_OUT, "manifest.json")
        write_manifest(entries, manifest_path)
        print(f"\nManifest: {manifest_path}")

    print(f"\nDone: ok={ok}  skipped={skip}  failed={fail}")


if __name__ == "__main__":
    main()
