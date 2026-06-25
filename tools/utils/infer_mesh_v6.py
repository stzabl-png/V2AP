#!/usr/bin/env python3
"""
Run PointNet++ v6 affordance on arbitrary triangle meshes (GLB/OBJ/PLY).

Default: all *.glb under data_hub/real_machine/sam3d_glb → npz/ + png/ + montage.

Pre-alignment: +90° about +X (Y→Z) for real-machine GLB (see ``apply_pre_rotation_x``).

Scale handling:
  Per-mesh ``{stem}.scale.json`` next to the GLB (``target_height_m`` on Z after
  pre-rotation). If missing, fall back to ``rescale_mesh_for_v6`` auto band.

Usage::

    python tools/infer_mesh_v6.py

    python tools/infer_mesh_v6.py \\
        --mesh-dir data_hub/real_machine/sam3d_glb \\
        --checkpoint output/affordance_no_rot_executed/min20/checkpoints_v6/best_v6_model.pth \\
        --save-dir output/real_machine/affordance_v6_inf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_MESH_DIR = PROJ / "data_hub" / "real_machine" / "sam3d_glb"
DEFAULT_CHECKPOINT = (
    PROJ / "output" / "affordance_no_rot_executed" / "min20" / "checkpoints_v6" / "best_v6_model.pth"
)
DEFAULT_SAVE_DIR = PROJ / "output" / "real_machine" / "affordance_v6_inf"

# From affordance_all_soft.h5 (82 train objects): max bbox edge length per sample.
TRAIN_MAX_EXTENT_MEDIAN = 0.149
TRAIN_MAX_EXTENT_P10 = 0.100
TRAIN_MAX_EXTENT_P90 = 0.227
# Real-machine GLB are often larger than YCB/OakInk; default target above training p90.
DEFAULT_TARGET_MAX_EXTENT = 0.30
DEFAULT_AUTO_EXTENT_LO = 0.08
DEFAULT_AUTO_EXTENT_HI = 0.38
# When shrinking, do not scale below this factor (avoids over-shrinking long objects).
DEFAULT_MIN_SCALE_FACTOR = 0.45

sys.path.insert(0, str(PROJ))

from model.inference_v6 import (  # noqa: E402
    default_threshold,
    predict_heatmap_batch,
    save_inference_montage,
    save_vis_png,
)


@dataclass
class ScaleReport:
    obj_id: str
    extent_before: np.ndarray
    extent_after: np.ndarray
    max_extent_before: float
    max_extent_after: float
    scale_applied: float
    centered: bool
    mode: str
    skipped_scale: bool
    height_before: float = 0.0
    height_after: float = 0.0
    target_height_m: float | None = None
    height_axis: str = "z"
    scale_json_path: str | None = None


def load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh geometry in scene: {path}")
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(loaded)} from {path}")
    if len(loaded.faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")
    return loaded


def mesh_extent(vertices: np.ndarray) -> np.ndarray:
    return (vertices.max(axis=0) - vertices.min(axis=0)).astype(np.float64)


def axis_index(axis: str) -> int:
    ax = axis.lower().strip()
    if ax not in ("x", "y", "z"):
        raise ValueError(f"height_axis must be x/y/z, got {axis!r}")
    return {"x": 0, "y": 1, "z": 2}[ax]


def mesh_height(vertices: np.ndarray, axis: str = "z") -> float:
    idx = axis_index(axis)
    return float(vertices[:, idx].max() - vertices[:, idx].min())


def scale_json_path_for_mesh(mesh_path: Path) -> Path:
    """``IMG_4475.glb`` → ``IMG_4475.scale.json`` in the same directory."""
    return mesh_path.parent / f"{mesh_path.stem}.scale.json"


def load_scale_json(mesh_path: Path) -> dict | None:
    path = scale_json_path_for_mesh(mesh_path)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    if "target_height_m" not in spec:
        raise ValueError(f"{path} must contain target_height_m")
    return spec


def apply_target_height_scale(
    mesh: trimesh.Trimesh,
    spec: dict,
    *,
    center_mesh: bool = True,
) -> tuple[float, float, float]:
    """
    Uniformly scale mesh so bbox height along ``height_axis`` equals ``target_height_m``.

    Returns (scale_applied, height_before, height_after).
    """
    axis = str(spec.get("height_axis", "z"))
    target_h = float(spec["target_height_m"])
    if center_mesh:
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        mesh.vertices -= mesh.centroid

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    h_before = mesh_height(verts, axis)
    s = target_h / max(h_before, 1e-8)
    mesh.vertices = (verts * s).astype(np.float64)
    h_after = mesh_height(np.asarray(mesh.vertices), axis)
    return float(s), h_before, h_after


def apply_pre_rotation_x(
    mesh: trimesh.Trimesh,
    degrees: float = 90.0,
) -> np.ndarray:
    """
    Rotate mesh about +X (right-hand rule): +Y → +Z, +Z → -Y when degrees=+90.

    SAM3D real-machine GLB exports need this to align with affordance training frame.
    """
    from scipy.spatial.transform import Rotation

    R = Rotation.from_euler("x", float(degrees), degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    mesh.apply_transform(T)
    return R.astype(np.float32)


def rescale_mesh_for_v6(
    mesh: trimesh.Trimesh,
    *,
    target_max_extent: float = DEFAULT_TARGET_MAX_EXTENT,
    scale_mode: str = "auto",
    extent_lo: float = DEFAULT_AUTO_EXTENT_LO,
    extent_hi: float = DEFAULT_AUTO_EXTENT_HI,
    min_scale_factor: float = DEFAULT_MIN_SCALE_FACTOR,
    center_mesh: bool = True,
) -> tuple[trimesh.Trimesh, ScaleReport]:
    """
    Put mesh in a metric scale band similar to affordance training data.

    Steps:
      1. Optional centroid centering (training point clouds are near origin).
      2. Uniform scale toward ``target_max_extent`` (max bbox edge, meters).
         When shrinking, ``s`` is clamped to ``>= min_scale_factor`` so large GLB
         exports are not crushed to training-median size (~0.15 m).
         ``auto``: scale only if max extent is outside [extent_lo, extent_hi].
         ``always``: always apply s (still respects min_scale_factor when shrinking).
         ``never``: s = 1.
    Normals are unchanged under uniform scale; positions are scaled.
    """
    mesh = mesh.copy()
    obj_id = "mesh"
    ext_before = mesh_extent(np.asarray(mesh.vertices))
    max_before = float(ext_before.max())

    if center_mesh:
        mesh.vertices -= mesh.centroid

    skipped = False
    if scale_mode == "never":
        s = 1.0
        skipped = True
    elif scale_mode in ("always", "auto"):
        if scale_mode == "auto" and extent_lo <= max_before <= extent_hi:
            s = 1.0
            skipped = True
        else:
            s = target_max_extent / max(max_before, 1e-8)
            if s < 1.0:
                s = max(float(s), float(min_scale_factor))
            elif s > 1.0:
                s = min(float(s), 1.0 / float(min_scale_factor))
    else:
        raise ValueError(f"unknown scale_mode: {scale_mode}")

    if abs(s - 1.0) > 1e-9:
        mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) * s).astype(np.float64)

    ext_after = mesh_extent(np.asarray(mesh.vertices))
    report = ScaleReport(
        obj_id=obj_id,
        extent_before=ext_before,
        extent_after=ext_after,
        max_extent_before=max_before,
        max_extent_after=float(ext_after.max()),
        scale_applied=float(s),
        centered=center_mesh,
        mode=scale_mode,
        skipped_scale=skipped,
    )
    return mesh, report


def rescale_mesh_with_optional_json(
    mesh: trimesh.Trimesh,
    mesh_path: Path,
    *,
    scale_mode: str,
    target_max_extent: float,
    extent_lo: float,
    extent_hi: float,
    min_scale_factor: float,
    center_mesh: bool,
    prefer_scale_json: bool = True,
) -> tuple[trimesh.Trimesh, ScaleReport]:
    """Use ``{stem}.scale.json`` when present; else fallback auto band rescale."""
    spec = load_scale_json(mesh_path) if prefer_scale_json else None
    ext_before = mesh_extent(np.asarray(mesh.vertices))

    if spec is not None:
        s, h0, h1 = apply_target_height_scale(mesh, spec, center_mesh=center_mesh)
        ext_after = mesh_extent(np.asarray(mesh.vertices))
        report = ScaleReport(
            obj_id=mesh_path.stem,
            extent_before=ext_before,
            extent_after=ext_after,
            max_extent_before=float(ext_before.max()),
            max_extent_after=float(ext_after.max()),
            scale_applied=s,
            centered=center_mesh,
            mode="scale_json",
            skipped_scale=False,
            height_before=h0,
            height_after=h1,
            target_height_m=float(spec["target_height_m"]),
            height_axis=str(spec.get("height_axis", "z")),
            scale_json_path=str(scale_json_path_for_mesh(mesh_path)),
        )
        return mesh, report

    mesh, report = rescale_mesh_for_v6(
        mesh,
        target_max_extent=target_max_extent,
        scale_mode=scale_mode,
        extent_lo=extent_lo,
        extent_hi=extent_hi,
        min_scale_factor=min_scale_factor,
        center_mesh=center_mesh,
    )
    report.height_axis = "z"
    report.height_before = mesh_height(np.asarray(mesh.vertices), "z")
    report.height_after = report.height_before
    return mesh, report


def sample_mesh_points(
    mesh: trimesh.Trimesh,
    num_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Surface sample like prepare_affordance_executed (face normals)."""
    rng = np.random.default_rng(seed)
    points, face_idx = trimesh.sample.sample_surface(mesh, num_points, seed=rng)
    normals = mesh.face_normals[face_idx].astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return points.astype(np.float32), normals


def discover_meshes(mesh_dir: Path, patterns: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(sorted(mesh_dir.glob(pat)))
    return sorted(set(paths))


def obj_id_from_path(path: Path) -> str:
    return path.stem


def main() -> None:
    p = argparse.ArgumentParser(description="Affordance v6 inference on triangle meshes")
    p.add_argument(
        "--mesh-dir",
        type=Path,
        default=DEFAULT_MESH_DIR,
        help=f"Directory of meshes (default: {DEFAULT_MESH_DIR})",
    )
    p.add_argument(
        "--mesh",
        type=Path,
        action="append",
        default=None,
        help="Single mesh file (repeatable); overrides mesh-dir scan if set",
    )
    p.add_argument("--glob", default="*.glb", help="Glob under --mesh-dir (default: *.glb)")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="For default_threshold lookup (default: parent of checkpoint)")
    p.add_argument("--num-points", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8, help="Meshes per GPU forward")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-vis", action="store_true")
    p.add_argument("--no-grid", action="store_true")
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument(
        "--scale-mode",
        choices=("auto", "always", "never"),
        default="auto",
        help="Fallback when no {stem}.scale.json (default: auto)",
    )
    p.add_argument(
        "--ignore-scale-json",
        action="store_true",
        help="Ignore {stem}.scale.json next to each GLB",
    )
    p.add_argument(
        "--target-max-extent",
        type=float,
        default=DEFAULT_TARGET_MAX_EXTENT,
        help="Target max bbox edge (m) after rescale (default: 0.30, above train p90)",
    )
    p.add_argument(
        "--min-scale-factor",
        type=float,
        default=DEFAULT_MIN_SCALE_FACTOR,
        help="When shrinking, s will not be below this (default: 0.45)",
    )
    p.add_argument(
        "--auto-extent-lo",
        type=float,
        default=DEFAULT_AUTO_EXTENT_LO,
        help="auto mode: skip scale if max extent in [lo, hi]",
    )
    p.add_argument(
        "--auto-extent-hi",
        type=float,
        default=DEFAULT_AUTO_EXTENT_HI,
        help="auto mode: upper band in meters (default: 0.38)",
    )
    p.add_argument("--no-center", action="store_true", help="Do not center mesh at centroid")
    p.add_argument(
        "--no-pre-rotate-x",
        action="store_true",
        help="Skip +90° rotation about +X (Y→Z) applied to real-machine GLB",
    )
    p.add_argument(
        "--pre-rotate-x-deg",
        type=float,
        default=90.0,
        help="Pre-rotation about +X in degrees (default: +90, Y toward Z)",
    )
    args = p.parse_args()

    import torch
    from model.inference_v6 import load_model

    if args.mesh:
        mesh_paths = [Path(m) for m in args.mesh]
    else:
        mesh_dir = args.mesh_dir.expanduser().resolve()
        if not mesh_dir.is_dir():
            raise SystemExit(f"--mesh-dir not found: {mesh_dir}")
        mesh_paths = discover_meshes(mesh_dir, (args.glob,))

    if not mesh_paths:
        raise SystemExit("No mesh files found")

    ckpt = str(args.checkpoint.expanduser().resolve())
    if not os.path.isfile(ckpt):
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    save_dir = args.save_dir.expanduser().resolve()
    npz_dir = save_dir / "npz"
    png_dir = save_dir / "png"
    npz_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_vis:
        png_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        dataset_dir = args.checkpoint.parent.parent
    thresh = args.threshold if args.threshold is not None else default_threshold(
        ckpt, str(dataset_dir),
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt_meta = load_model(ckpt, device)

    print("=" * 60)
    print("infer_mesh_v6")
    print(f"  Meshes:     {len(mesh_paths)}")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Save:       {save_dir}")
    print(f"  Device:     {device}")
    print(f"  Points:     {args.num_points}")
    print(f"  Threshold:  {thresh:.3f}")
    print(f"  Scale:      mode={args.scale_mode}  target_max_extent={args.target_max_extent:.4f} m")
    print(f"              auto band [{args.auto_extent_lo:.3f}, {args.auto_extent_hi:.3f}] m")
    print(f"              min_scale_factor={args.min_scale_factor:.2f}  (train median {TRAIN_MAX_EXTENT_MEDIAN:.3f} m)")
    print(f"  Center:     {not args.no_center}")
    print(
        f"  Pre-rot X:  {args.pre_rotate_x_deg:.1f}°"
        + (" (off)" if args.no_pre_rotate_x else "  (Y→Z)"),
    )
    print("=" * 60)

    scale_rows: list[dict] = []
    prepared: list[dict] = []

    for i, mpath in enumerate(mesh_paths):
        oid = obj_id_from_path(mpath)
        print(f"\n[{i+1}/{len(mesh_paths)}] {oid}  ←  {mpath.name}")
        mesh = load_triangle_mesh(mpath)
        pre_rot_R = None
        if not args.no_pre_rotate_x:
            pre_rot_R = apply_pre_rotation_x(mesh, args.pre_rotate_x_deg)
            print(f"  pre-rotate +X {args.pre_rotate_x_deg:.1f}°  extent {np.round(mesh_extent(mesh.vertices), 4)}")
        mesh, srep = rescale_mesh_with_optional_json(
            mesh,
            mpath,
            scale_mode=args.scale_mode,
            target_max_extent=args.target_max_extent,
            extent_lo=args.auto_extent_lo,
            extent_hi=args.auto_extent_hi,
            min_scale_factor=args.min_scale_factor,
            center_mesh=not args.no_center,
            prefer_scale_json=not args.ignore_scale_json,
        )
        srep.obj_id = oid
        if srep.scale_json_path:
            print(
                f"  scale.json: {srep.scale_json_path}  "
                f"target_h({srep.height_axis})={srep.target_height_m:.4f} m",
            )
            print(
                f"  height: {srep.height_before:.4f} → {srep.height_after:.4f} m  "
                f"uniform scale={srep.scale_applied:.4f}",
            )
        print(
            f"  extent before: {np.round(srep.extent_before, 4)}  max={srep.max_extent_before:.4f} m",
        )
        print(
            f"  extent after:  {np.round(srep.extent_after, 4)}  max={srep.max_extent_after:.4f} m  "
            f"mode={srep.mode}",
        )
        pts, nrm = sample_mesh_points(mesh, args.num_points, args.seed + i)
        prepared.append({
            "obj_id": oid,
            "mesh_path": str(mpath),
            "points": pts,
            "normals": nrm,
            "scale_report": srep,
            "pre_rotation_x_deg": None if args.no_pre_rotate_x else float(args.pre_rotate_x_deg),
            "pre_rotation_matrix": pre_rot_R,
        })
        scale_rows.append({
            "obj_id": oid,
            "mesh_path": str(mpath),
            "scale_json": srep.scale_json_path,
            "pre_rotation_x_deg": None if args.no_pre_rotate_x else float(args.pre_rotate_x_deg),
            "target_height_m": srep.target_height_m,
            "height_axis": srep.height_axis,
            "height_before_m": srep.height_before,
            "height_after_m": srep.height_after,
            "max_extent_before_m": srep.max_extent_before,
            "max_extent_after_m": srep.max_extent_after,
            "scale_applied": srep.scale_applied,
            "skipped_scale": srep.skipped_scale,
            "scale_mode": srep.mode,
            "centered": srep.centered,
        })

    obj_ids_order = [x["obj_id"] for x in prepared]
    batch_size = max(1, args.batch_size)

    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        pts_b = np.stack([c["points"] for c in chunk], axis=0)
        nrm_b = np.stack([c["normals"] for c in chunk], axis=0)
        preds = predict_heatmap_batch(model, pts_b, nrm_b, device)
        if preds.ndim == 1:
            preds = preds[np.newaxis, :]
        print(f"\n  [batch] forward {len(chunk)} mesh(es)")
        for j, item in enumerate(chunk):
            oid = item["obj_id"]
            pred = preds[j]
            srep = item["scale_report"]
            print(
                f"  {oid}: pred max={pred.max():.3f} mean={pred.mean():.4f}",
            )
            npz_kw = dict(
                points=item["points"],
                normals=item["normals"],
                pred=pred,
                gt=None,
                obj_id=oid,
                threshold=thresh,
                mesh_path=item["mesh_path"],
                scale_applied=srep.scale_applied,
                max_extent_before_m=srep.max_extent_before,
                max_extent_after_m=srep.max_extent_after,
                extent_before_m=srep.extent_before,
                extent_after_m=srep.extent_after,
                centered=srep.centered,
                scale_mode=srep.mode,
                skipped_scale=srep.skipped_scale,
            )
            if item["pre_rotation_x_deg"] is not None:
                npz_kw["pre_rotation_x_deg"] = item["pre_rotation_x_deg"]
            if item["pre_rotation_matrix"] is not None:
                npz_kw["pre_rotation_matrix"] = item["pre_rotation_matrix"]
            srep = item["scale_report"]
            if srep.target_height_m is not None:
                npz_kw["target_height_m"] = float(srep.target_height_m)
                npz_kw["height_axis"] = srep.height_axis
                npz_kw["height_before_m"] = srep.height_before
                npz_kw["height_after_m"] = srep.height_after
            if srep.scale_json_path:
                npz_kw["scale_json_path"] = srep.scale_json_path
            np.savez(npz_dir / f"{oid}.npz", **npz_kw)
            print(f"    → npz/{oid}.npz")
            if not args.no_vis:
                title_suffix = (
                    f"  scale×{srep.scale_applied:.3f}"
                    if not srep.skipped_scale
                    else "  scale=1 (in band)"
                )
                save_vis_png(
                    str(png_dir / f"{oid}.png"),
                    item["points"],
                    None,
                    pred,
                    f"{oid}{title_suffix}",
                    thresh,
                )
                print(f"    → png/{oid}.png")

    scale_json = save_dir / "scale_report.json"
    with open(scale_json, "w") as f:
        json.dump(
            {
                "target_max_extent_m": args.target_max_extent,
                "scale_mode": args.scale_mode,
                "min_scale_factor": args.min_scale_factor,
                "auto_extent_lo": args.auto_extent_lo,
                "auto_extent_hi": args.auto_extent_hi,
                "train_max_extent_median": TRAIN_MAX_EXTENT_MEDIAN,
                "train_max_extent_p10": TRAIN_MAX_EXTENT_P10,
                "train_max_extent_p90": TRAIN_MAX_EXTENT_P90,
                "center_mesh": not args.no_center,
                "pre_rotation_x_deg": None if args.no_pre_rotate_x else float(args.pre_rotate_x_deg),
                "objects": scale_rows,
            },
            f,
            indent=2,
        )
    print(f"\n  Scale report: {scale_json}")

    if not args.no_vis and not args.no_grid:
        save_inference_montage(
            str(save_dir),
            obj_ids_order,
            cols=args.grid_cols,
            max_cell_width=512,
            max_per_page=0,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
