#!/usr/bin/env python3
"""
One-stop pipeline: triangle mesh (GLB/OBJ/PLY) → affordance v6 → PDM grasp candidates.

Steps per mesh:
  1. Load mesh, optional +90° about +X, scale via ``{stem}.scale.json`` or auto band, center.
  2. Surface-sample 4096 points + normals; run PointNet++ v6 affordance.
  3. Build PDM condition (xyz + normal + affordance heatmap).
  4. DDIM-sample ``--n-samples`` poses; write ``{obj_id}[_yaw###]_grasp.hdf5``.

Yaw (default PDM ckpt: ``output/pdm/checkpoints_yaw``, ``use_yaw_condition=True``):
  - Omit ``--z-yaw-deg``: object z-yaw = 0° (same as ``model.pdm.sample``).
  - Set ``--z-yaw-deg DEG``: condition on that z-yaw in degrees.
  - For checkpoints without yaw (``output/pdm/checkpoints``), do not pass ``--z-yaw-deg``.

Usage::

    python tools/glb_to_pdm_grasp.py

    python tools/glb_to_pdm_grasp.py \\
        --mesh data_hub/real_machine/sam3d_glb/IMG_4475.glb \\
        --n-samples 30 \\
        --output-dir output/real_machine/pdm/candidates

    python tools/glb_to_pdm_grasp.py --mesh-dir data_hub/real_machine/sam3d_glb --z-yaw-deg 90

Default writes:
  - affordance npz/png + scale report under ``<output-dir>/../affordance/``
  - grasp HDF5 under ``--output-dir``
  - PDM overlay PNGs under ``<output-dir>/../vis/`` (``--vis``)

Both affordance and PDM point clouds use the shared ``jet`` colormap.
Affordance PNG/PDM overlay use max-normalized predictions (peak → 1); PDM conditioning
uses raw v6 outputs. Normalized arrays are also written under ``affordance/npz_norm/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_MESH_DIR = PROJ / "data_hub" / "real_machine" / "sam3d_glb"
DEFAULT_AFF_CKPT = (
    PROJ / "output" / "affordance_no_rot_executed" / "min20" / "checkpoints_v6" / "best_v6_model.pth"
)
DEFAULT_PDM_CKPT = PROJ / "output" / "pdm" / "checkpoints_yaw_v6cond" / "best_model.pth"
DEFAULT_OUT_DIR = PROJ / "output" / "real_machine" / "pdm" / "candidates"
DEFAULT_AFF_OUT_DIR = PROJ / "output" / "real_machine" / "affordance"

sys.path.insert(0, str(PROJ))

from evaluation.affordance_ckpt import (  # noqa: E402
    add_affordance_checkpoint_args,
    resolve_affordance_checkpoint,
)
from model.inference_v6 import (  # noqa: E402
    default_threshold,
    load_model,
    normalize_affordance_pred,
    predict_heatmap_batch,
    save_inference_montage,
)
from model.pdm.dataset import yaw_feature_from_deg  # noqa: E402
from model.pdm.model import PDM  # noqa: E402
from model.pdm.pose_codec import pose9_to_command  # noqa: E402
from model.pdm.sample import write_candidates_hdf5  # noqa: E402
from model.pdm.visualize import make_overview, save_candidate_overlay  # noqa: E402
from tools.infer_mesh_v6 import (  # noqa: E402
    TRAIN_MAX_EXTENT_MEDIAN,
    TRAIN_MAX_EXTENT_P10,
    TRAIN_MAX_EXTENT_P90,
    apply_pre_rotation_x,
    discover_meshes,
    load_triangle_mesh,
    obj_id_from_path,
    rescale_mesh_with_optional_json,
    rescale_mesh_for_v6,
    sample_mesh_points,
)
from tools.affordance_pointcloud_vis import save_scalar_pointcloud_png  # noqa: E402


def build_condition_tensor(
    points: np.ndarray,
    normals: np.ndarray,
    affordance: np.ndarray,
) -> torch.Tensor:
    """(N,3) + (N,3) + (N,) → (1, N, 7) float32."""
    aff = np.asarray(affordance, dtype=np.float32).reshape(-1, 1)
    channels = np.concatenate(
        [
            np.asarray(points, dtype=np.float32),
            np.asarray(normals, dtype=np.float32),
            aff,
        ],
        axis=-1,
    )
    return torch.from_numpy(channels).unsqueeze(0)


def resolve_yaw_tensor(
    z_yaw_deg: float | None,
    *,
    use_yaw_condition: bool,
    device: torch.device,
) -> torch.Tensor | None:
    if z_yaw_deg is None:
        return None
    if not use_yaw_condition:
        raise ValueError(
            "PDM checkpoint was not trained with yaw conditioning (use_yaw_condition=False). "
            "Remove --z-yaw-deg or retrain with --use-yaw-condition."
        )
    feat = yaw_feature_from_deg(float(z_yaw_deg))
    return torch.from_numpy(feat).unsqueeze(0).to(device=device, dtype=torch.float32)


def output_hdf5_name(obj_id: str, z_yaw_deg: float | None) -> str:
    if z_yaw_deg is None:
        return f"{obj_id}_grasp.hdf5"
    tag = int(round(float(z_yaw_deg))) % 360
    return f"{obj_id}_yaw{tag:03d}_grasp.hdf5"


def prepare_mesh_item(
    mpath: Path,
    *,
    obj_id_override: str | None,
    dataset: str | None,
    sam3d_rotated_mesh: bool,
    no_pre_rotate_x: bool,
    pre_rotate_x_deg: float,
    scale_mode: str,
    ignore_scale_json: bool,
    target_max_extent: float,
    auto_extent_lo: float,
    auto_extent_hi: float,
    min_scale_factor: float,
    no_center: bool,
    num_points: int,
    seed: int,
    index: int,
) -> dict:
    oid = obj_id_override or obj_id_from_path(mpath)
    mesh = load_triangle_mesh(mpath)
    pre_rot_R = None
    if sam3d_rotated_mesh:
        no_pre_rotate_x = True
        scale_mode = "never"
        no_center = True
        try:
            from tools import random_grasp_sampler as rgs

            ds = dataset or None
            sf = rgs.read_scale_factor(oid, ds)
            if rgs.apply_metric_scale_to_mesh(oid, ds) and abs(sf - 1.0) > 1e-8:
                mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) * float(sf)).astype(np.float64)
        except Exception as exc:
            print(f"  WARNING: rotated SAM3D scale lookup failed for {oid}: {exc}")

    if not no_pre_rotate_x:
        pre_rot_R = apply_pre_rotation_x(mesh, pre_rotate_x_deg)

    if sam3d_rotated_mesh:
        mesh, srep = rescale_mesh_for_v6(
            mesh,
            target_max_extent=target_max_extent,
            scale_mode="never",
            extent_lo=auto_extent_lo,
            extent_hi=auto_extent_hi,
            min_scale_factor=min_scale_factor,
            center_mesh=False,
        )
        srep.mode = "sam3d_rotated_metric"
    else:
        mesh, srep = rescale_mesh_with_optional_json(
            mesh,
            mpath,
            scale_mode=scale_mode,
            target_max_extent=target_max_extent,
            extent_lo=auto_extent_lo,
            extent_hi=auto_extent_hi,
            min_scale_factor=min_scale_factor,
            center_mesh=not no_center,
            prefer_scale_json=not ignore_scale_json,
        )
    srep.obj_id = oid
    pts, nrm = sample_mesh_points(mesh, num_points, seed)
    return {
        "obj_id": oid,
        "mesh_path": str(mpath.resolve()),
        "points": pts,
        "normals": nrm,
        "scale_report": srep,
        "pre_rotation_x_deg": None if no_pre_rotate_x else float(pre_rotate_x_deg),
        "pre_rotation_matrix": pre_rot_R,
    }


def run_pdm_sample(
    pdm: PDM,
    stats: dict,
    condition: torch.Tensor,
    *,
    n_samples: int,
    ddim_steps: int,
    z_yaw_deg: float | None,
    device: torch.device,
    reject_upward: bool,
    max_approach_z: float,
) -> np.ndarray:
    pose_mean = stats["pose_mean"].to(device)
    pose_std = stats["pose_std"].to(device)
    yaw = resolve_yaw_tensor(
        z_yaw_deg,
        use_yaw_condition=pdm.config.use_yaw_condition,
        device=device,
    )
    cond = condition.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pose_norm = pdm.sample(
            cond,
            yaw=yaw,
            n_samples=n_samples,
            ddim_steps=ddim_steps,
        )
        pose = pose_norm * pose_std.unsqueeze(0) + pose_mean.unsqueeze(0)
    poses_np = pose.cpu().numpy().astype(np.float32)

    if reject_upward:
        kept = []
        for p in poses_np:
            cmd = pose9_to_command(p)
            if cmd.rotation[:, 2][2] <= max_approach_z:
                kept.append(p)
        poses_np = np.asarray(kept, dtype=np.float32)
    return poses_np


def save_affordance_outputs(
    *,
    affordance_dir: Path,
    item: dict,
    pred: np.ndarray,
    pred_norm: np.ndarray,
    pred_norm_scale: float,
    threshold: float,
    no_aff_vis: bool,
) -> tuple[str, str, str]:
    """Write raw + normalized affordance npz and PNG; return (raw_npz, norm_npz, png)."""
    oid = item["obj_id"]
    srep = item["scale_report"]
    npz_dir = affordance_dir / "npz"
    npz_norm_dir = affordance_dir / "npz_norm"
    png_dir = affordance_dir / "png"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_norm_dir.mkdir(parents=True, exist_ok=True)
    if not no_aff_vis:
        png_dir.mkdir(parents=True, exist_ok=True)

    base_kw = dict(
        points=item["points"],
        normals=item["normals"],
        obj_id=oid,
        threshold=threshold,
        mesh_path=item["mesh_path"],
        scale_applied=srep.scale_applied,
        max_extent_before_m=srep.max_extent_before,
        max_extent_after_m=srep.max_extent_after,
        extent_before_m=srep.extent_before,
        extent_after_m=srep.extent_after,
        centered=srep.centered,
        scale_mode=srep.mode,
        skipped_scale=srep.skipped_scale,
        pred_norm_scale=float(pred_norm_scale),
    )
    if item["pre_rotation_x_deg"] is not None:
        base_kw["pre_rotation_x_deg"] = item["pre_rotation_x_deg"]
    if item["pre_rotation_matrix"] is not None:
        base_kw["pre_rotation_matrix"] = item["pre_rotation_matrix"]
    if srep.target_height_m is not None:
        base_kw["target_height_m"] = float(srep.target_height_m)
        base_kw["height_axis"] = srep.height_axis
        base_kw["height_before_m"] = srep.height_before
        base_kw["height_after_m"] = srep.height_after
    if srep.scale_json_path:
        base_kw["scale_json_path"] = srep.scale_json_path

    raw_npz_path = str(npz_dir / f"{oid}.npz")
    np.savez(raw_npz_path, pred=pred, gt=None, **base_kw)

    norm_npz_path = str(npz_norm_dir / f"{oid}.npz")
    np.savez(
        norm_npz_path,
        pred=pred_norm,
        pred_raw=pred,
        gt=None,
        **base_kw,
    )

    png_path = ""
    if not no_aff_vis:
        png_path = str(png_dir / f"{oid}.png")
        save_scalar_pointcloud_png(
            Path(png_path),
            item["points"],
            pred_norm,
            vmax=1.0,
        )
    return raw_npz_path, norm_npz_path, png_path


def scale_report_row(item: dict) -> dict:
    srep = item["scale_report"]
    return {
        "obj_id": item["obj_id"],
        "mesh_path": item["mesh_path"],
        "scale_json": srep.scale_json_path,
        "pre_rotation_x_deg": item["pre_rotation_x_deg"],
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
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="GLB/mesh → affordance v6 → PDM grasp candidate HDF5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
  # meshes
    p.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    p.add_argument("--mesh", type=Path, action="append", default=None)
    p.add_argument("--obj-id", default=None, help="Override obj_id for a single --mesh input")
    p.add_argument("--glob", default="*.glb")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--dataset", default="real_machine", help="metadata.dataset in HDF5")
  # checkpoints
    add_affordance_checkpoint_args(p)
    p.add_argument("--pdm-checkpoint", type=Path, default=DEFAULT_PDM_CKPT)
    p.add_argument("--pose-stats", type=Path, default=None, help="If PDM ckpt lacks pose_stats")
    p.add_argument("--dataset-dir", type=Path, default=None, help="For affordance default threshold")
  # sampling
    p.add_argument("--n-samples", type=int, default=50, help="PDM poses per mesh")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--num-points", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42, help="Fixed RNG seed for mesh sampling (+ obj index)")
    p.add_argument(
        "--random-seed",
        action="store_true",
        help="Use a fresh nondeterministic seed for mesh sampling / PDM",
    )
    p.add_argument("--aff-batch-size", type=int, default=8, help="Affordance forward batch size")
    p.add_argument("--threshold", type=float, default=None, help="Affordance viz threshold only")
    p.add_argument("--gripper-width", type=float, default=0.06)
    p.add_argument(
        "--z-yaw-deg",
        type=float,
        default=None,
        help="Object z-yaw in degrees (yaw-conditioned PDM only; default 0° if omitted)",
    )
    p.add_argument("--reject-upward", action="store_true")
    p.add_argument("--max-approach-z", type=float, default=0.3)
  # mesh align (same as infer_mesh_v6)
    p.add_argument("--scale-mode", choices=("auto", "always", "never"), default="auto")
    p.add_argument("--ignore-scale-json", action="store_true")
    p.add_argument("--target-max-extent", type=float, default=0.30)
    p.add_argument("--min-scale-factor", type=float, default=0.45)
    p.add_argument("--auto-extent-lo", type=float, default=0.08)
    p.add_argument("--auto-extent-hi", type=float, default=0.38)
    p.add_argument("--no-center", action="store_true")
    p.add_argument("--no-pre-rotate-x", action="store_true")
    p.add_argument("--pre-rotate-x-deg", type=float, default=90.0)
    p.add_argument(
        "--sam3d-rotated-mesh",
        action="store_true",
        help="Input mesh is data_hub/meshes/SAM3DMesh/rotated_mesh/{dataset}/{obj}/mesh.ply; skip +X rotation, preserve frame, apply metric scale.",
    )
  # outputs
    p.add_argument(
        "--affordance-dir",
        type=Path,
        default=None,
        help="Affordance npz/png root (default: <output-dir>/../../affordance or DEFAULT_AFF_OUT_DIR)",
    )
    p.add_argument("--no-affordance-output", action="store_true", help="Skip affordance npz/png/scale report")
    p.add_argument("--no-aff-vis", action="store_true", help="Skip affordance PNG + grid (still writes npz)")
    p.add_argument("--aff-grid-cols", type=int, default=4, help="Columns in affordance montage")
    p.add_argument(
        "--vis",
        action="store_true",
        default=True,
        help="Write PDM overlay PNG per object (default: on)",
    )
    p.add_argument("--no-vis", dest="vis", action="store_false")
    p.add_argument(
        "--vis-dir",
        type=Path,
        default=None,
        help="PDM overlay PNG directory (default: <output-dir>/../vis)",
    )
    p.add_argument("--vis-top", type=int, default=20, help="Max grippers drawn per PNG")
    p.add_argument("--device", default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    def _run_seed(index: int) -> int:
        if args.random_seed or args.seed is None:
            import secrets

            return secrets.randbelow(2**31 - 1) + index
        return int(args.seed) + index

    if args.mesh:
        mesh_paths = [Path(m).expanduser().resolve() for m in args.mesh]
    else:
        mesh_dir = args.mesh_dir.expanduser().resolve()
        if not mesh_dir.is_dir():
            raise SystemExit(f"--mesh-dir not found: {mesh_dir}")
        mesh_paths = discover_meshes(mesh_dir, (args.glob,))
    if not mesh_paths:
        raise SystemExit("No mesh files found")
    if args.obj_id and len(mesh_paths) != 1:
        raise SystemExit("--obj-id can only be used with exactly one --mesh")

    aff_ckpt = resolve_affordance_checkpoint(
        hp_affordance=bool(args.hp_affordance),
        affordance_checkpoint=args.affordance_checkpoint,
    )
    pdm_ckpt = args.pdm_checkpoint.expanduser().resolve()
    if not pdm_ckpt.is_file():
        raise SystemExit(f"PDM checkpoint not found: {pdm_ckpt}")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.vis_dir.expanduser().resolve() if args.vis_dir else (out_dir.parent / "vis")
    if args.vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if args.affordance_dir is not None:
        affordance_dir = args.affordance_dir.expanduser().resolve()
    elif out_dir.name == "candidates" and out_dir.parent.name == "pdm":
        affordance_dir = out_dir.parent.parent / "affordance"
    else:
        affordance_dir = DEFAULT_AFF_OUT_DIR
    if not args.no_affordance_output:
        affordance_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device or ("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")),
    )

    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        dataset_dir = aff_ckpt.parent.parent
    aff_thresh = args.threshold
    if aff_thresh is None:
        aff_thresh = default_threshold(str(aff_ckpt), str(dataset_dir.expanduser().resolve()))

    aff_model, _ = load_model(str(aff_ckpt), device)
    pdm_model, pdm_ckpt_meta = PDM.load(str(pdm_ckpt), device=device)
    stats = pdm_ckpt_meta.get("pose_stats")
    if stats is None:
        if args.pose_stats is None:
            raise SystemExit("PDM checkpoint has no pose_stats; pass --pose-stats")
        stats = torch.load(args.pose_stats.expanduser().resolve(), map_location=device, weights_only=False)

    print("=" * 72)
    print("glb_to_pdm_grasp")
    print(f"  Meshes:              {len(mesh_paths)}")
    print(f"  Affordance ckpt:     {aff_ckpt}")
    print(f"  PDM ckpt:            {pdm_ckpt}")
    print(f"  PDM use_yaw:         {pdm_model.config.use_yaw_condition}")
    print(f"  z_yaw_deg:           {args.z_yaw_deg if args.z_yaw_deg is not None else '(none)'}")
    print(f"  n_samples / ddim:    {args.n_samples} / {args.ddim_steps}")
    print(f"  Output dir:          {out_dir}")
    print(f"  Affordance out:      {affordance_dir if not args.no_affordance_output else '(off)'}")
    print(f"  PDM visualization:   {args.vis}  -> {vis_dir if args.vis else '(off)'}")
    print(f"  Mesh sample seed:    {'random' if args.random_seed else args.seed}")
    print(f"  Device:              {device}")
    print("=" * 72)

    prepared: list[dict] = []
    for i, mpath in enumerate(mesh_paths):
        oid = obj_id_from_path(mpath)
        print(f"\n[{i + 1}/{len(mesh_paths)}] prepare {oid}  ←  {mpath.name}")
        item = prepare_mesh_item(
            mpath,
            obj_id_override=args.obj_id if len(mesh_paths) == 1 else None,
            dataset=args.dataset,
            sam3d_rotated_mesh=args.sam3d_rotated_mesh,
            no_pre_rotate_x=args.no_pre_rotate_x,
            pre_rotate_x_deg=args.pre_rotate_x_deg,
            scale_mode=args.scale_mode,
            ignore_scale_json=args.ignore_scale_json,
            target_max_extent=args.target_max_extent,
            auto_extent_lo=args.auto_extent_lo,
            auto_extent_hi=args.auto_extent_hi,
            min_scale_factor=args.min_scale_factor,
            no_center=args.no_center,
            num_points=args.num_points,
            seed=_run_seed(i),
            index=i,
        )
        srep = item["scale_report"]
        if not args.no_pre_rotate_x:
            print(f"  pre-rotate +X {args.pre_rotate_x_deg:.1f}°")
        print(
            f"  scale mode={srep.mode}  max_extent {srep.max_extent_before:.4f} → "
            f"{srep.max_extent_after:.4f} m",
        )
        prepared.append(item)

    batch_size = max(1, args.aff_batch_size)
    summary_rows: list[dict] = []
    vis_png_paths: list[str] = []
    scale_rows: list[dict] = []
    obj_ids_order: list[str] = []

    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        pts_b = np.stack([c["points"] for c in chunk], axis=0)
        nrm_b = np.stack([c["normals"] for c in chunk], axis=0)
        preds = predict_heatmap_batch(aff_model, pts_b, nrm_b, device)
        if preds.ndim == 1:
            preds = preds[np.newaxis, :]

        for j, item in enumerate(chunk):
            oid = item["obj_id"]
            pred = preds[j].astype(np.float32)
            pred_norm, pred_norm_stats = normalize_affordance_pred(pred)
            pred_norm_scale = float(pred_norm_stats.get("pred_span", 0.0))
            print(
                f"\n  {oid}: affordance max={pred.max():.3f} mean={pred.mean():.4f}  "
                f"norm_scale={pred_norm_scale:.4f}",
            )
            obj_ids_order.append(oid)
            scale_rows.append(scale_report_row(item))

            aff_npz_path = None
            aff_npz_norm_path = None
            aff_png_path = None
            if not args.no_affordance_output:
                aff_npz_path, aff_npz_norm_path, aff_png_path = save_affordance_outputs(
                    affordance_dir=affordance_dir,
                    item=item,
                    pred=pred,
                    pred_norm=pred_norm,
                    pred_norm_scale=pred_norm_scale,
                    threshold=aff_thresh,
                    no_aff_vis=args.no_aff_vis,
                )
                print(f"  → affordance npz/{oid}.npz")
                print(f"  → affordance npz_norm/{oid}.npz")
                if aff_png_path:
                    print(f"  → affordance png/{oid}.png")

            condition = build_condition_tensor(item["points"], item["normals"], pred)
            poses_np = run_pdm_sample(
                pdm_model,
                stats,
                condition,
                n_samples=args.n_samples,
                ddim_steps=args.ddim_steps,
                z_yaw_deg=args.z_yaw_deg,
                device=device,
                reject_upward=args.reject_upward,
                max_approach_z=args.max_approach_z,
            )

            out_path = None
            vis_png = None
            n_candidates = 0
            if poses_np.size == 0:
                print(f"  WARNING: no poses left after filters for {oid}")
            else:
                n_candidates = int(len(poses_np))
                out_name = output_hdf5_name(oid, args.z_yaw_deg)
                out_path = out_dir / out_name
                write_candidates_hdf5(
                    str(out_path),
                    oid,
                    poses_np,
                    mesh_path=item["mesh_path"],
                    gripper_width=args.gripper_width,
                    dataset=args.dataset,
                )
                print(f"  → {out_path}  ({n_candidates} candidates)")

                if args.vis:
                    yaw_tag = ""
                    if args.z_yaw_deg is not None:
                        yaw_tag = f"  yaw={int(round(float(args.z_yaw_deg))) % 360}°"
                    vis_name = out_name.replace(
                        "_grasp.hdf5",
                        f"_pdm_overlay_top{min(n_candidates, args.vis_top)}.png",
                    )
                    vis_png = str(vis_dir / vis_name)
                    save_candidate_overlay(
                        str(out_path),
                        item["points"],
                        vis_png,
                        top=args.vis_top,
                        affordance=pred_norm,
                        affordance_vmax_fixed=1.0,
                        title_suffix=yaw_tag,
                    )
                    vis_png_paths.append(vis_png)
                    print(f"  → {vis_png}")

            srep = item["scale_report"]
            summary_rows.append({
                "obj_id": oid,
                "mesh_path": item["mesh_path"],
                "output_hdf5": str(out_path) if out_path is not None else None,
                "n_candidates": n_candidates,
                "affordance_max": float(pred.max()),
                "affordance_mean": float(pred.mean()),
                "affordance_norm_scale": float(pred_norm_scale),
                "affordance_norm_stats": dict(pred_norm_stats),
                "affordance_npz": aff_npz_path,
                "affordance_npz_norm": aff_npz_norm_path,
                "affordance_png": aff_png_path or None,
                "z_yaw_deg": args.z_yaw_deg,
                "pdm_use_yaw_condition": bool(pdm_model.config.use_yaw_condition),
                "scale_mode": srep.mode,
                "scale_applied": srep.scale_applied,
                "max_extent_after_m": srep.max_extent_after,
                "target_height_m": srep.target_height_m,
                "vis_png": vis_png,
                "scale_json": srep.scale_json_path,
            })

    if not args.no_affordance_output:
        scale_json = affordance_dir / "scale_report.json"
        with open(scale_json, "w", encoding="utf-8") as f:
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
                    "seed": None if args.random_seed else int(args.seed),
                    "objects": scale_rows,
                },
                f,
                indent=2,
            )
        print(f"\n  Affordance scale report: {scale_json}")
        if not args.no_aff_vis and obj_ids_order:
            save_inference_montage(
                str(affordance_dir),
                obj_ids_order,
                cols=args.aff_grid_cols,
                max_cell_width=512,
                max_per_page=0,
            )
            print(f"  Affordance grid: {affordance_dir / 'all_objects_grid.png'}")

    if args.vis and len(vis_png_paths) > 1:
        overview_path = str(vis_dir / "overview.png")
        make_overview(vis_png_paths, overview_path, cols=min(3, len(vis_png_paths)))
        print(f"\n  Overview -> {overview_path}")

    summary_path = out_dir / "glb_to_pdm_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meshes": len(mesh_paths),
                "affordance_checkpoint": str(aff_ckpt),
                "pdm_checkpoint": str(pdm_ckpt),
                "affordance_dir": str(affordance_dir) if not args.no_affordance_output else None,
                "n_samples": args.n_samples,
                "z_yaw_deg": args.z_yaw_deg,
                "vis_dir": str(vis_dir) if args.vis else None,
                "seed": None if args.random_seed else int(args.seed),
                "objects": summary_rows,
            },
            f,
            indent=2,
        )
    print(f"\nSummary -> {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
