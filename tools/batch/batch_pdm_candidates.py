#!/usr/bin/env python3
"""Batch PDM candidate generation for evaluation pool.

This script is intentionally separate from glb_to_pdm_grasp.py so the existing
single-mesh generation path keeps its current behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from evaluation.affordance_ckpt import (  # noqa: E402
    add_affordance_checkpoint_args,
    resolve_affordance_checkpoint,
)
from model.inference_v6 import default_threshold, load_model, predict_heatmap_batch  # noqa: E402
from model.pdm.dataset import yaw_feature_from_deg  # noqa: E402
from model.pdm.model import PDM  # noqa: E402
from model.pdm.pose_codec import (  # noqa: E402
    R_ADAPT,
    TCP_OFFSET,
    CommandPose,
    command_to_executed,
    pose9_to_command,
    rotation_to_6d,
)
from model.pdm.sample import write_candidates_hdf5  # noqa: E402
from model.pdm.mesh_points import resolve_metric_dataset as _resolve_metric_dataset_mp  # noqa: E402
from model.pdm.mesh_points import resolve_mesh_path as _resolve_mesh_path_mp  # noqa: E402
from tools.infer_mesh_v6 import (  # noqa: E402
    apply_pre_rotation_x,
    load_triangle_mesh,
    rescale_mesh_for_v6,
    rescale_mesh_with_optional_json,
    sample_mesh_points,
)


DEFAULT_AFF_CKPT = (
    PROJ / "output" / "affordance_no_rot_executed" / "min20" / "checkpoints_v6" / "best_v6_model.pth"
)
DEFAULT_PDM_CKPT = PROJ / "output" / "pdm" / "checkpoints_yaw" / "best_model.pth"
TABLE_TOP_Z = 0.80
OBJECT_POSITION = [0.0, 0.55, TABLE_TOP_Z]
OBJECT_ORIENTATION = [0.0, 0.0, 0.0]
HARD_GATE_TABLE_MARGIN = 0.005
DEFAULT_GRIPPER_WIDTH = 0.06

# Filtering (hard gate + scoring) defaults.
FILTER_POOL_MULTIPLIER = 2
FILTER_SURFACE_SIGMA_M = 0.003
FILTER_WIDTH_CENTER_M = 0.035
FILTER_WIDTH_SIGMA_M = 0.020
FILTER_WIDTH_MIN_M = 0.005
FILTER_WIDTH_MAX_M = 0.080
FILTER_CLEARANCE_LAMBDA_M = 0.020
FILTER_APPROACH_Z_MIN = -1.0
FILTER_APPROACH_Z_MAX = 0.30

# Soft scoring weights (sum not required to be 1.0).
FILTER_W_SURF = 0.35
FILTER_W_WIDTH = 0.20
FILTER_W_ANTI = 0.25
FILTER_W_APPROACH = 0.10
FILTER_W_CLEAR = 0.10


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _find_obj_usd_path(obj_id: str) -> str | None:
    obj_usd_root = PROJ / "output" / "obj_usd"
    datasets_order = ["oakink", "ycb", "arctic", "dexycb", "egocentric", "ho3d_v3"]
    paths = [obj_usd_root / ds / f"{obj_id}.usd" for ds in datasets_order]
    paths.append(PROJ / "sim" / "assets" / f"{obj_id}.usd")
    return next((str(p) for p in paths if p.exists()), None)


def _object_rotation_overrides() -> dict:
    path = PROJ / "sim" / "object_rotation_overrides.json"
    try:
        data = _load_json(path)
    except Exception:
        return {}
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


OBJECT_ROTATION_OVERRIDES = _object_rotation_overrides()


def _resolve_object_placement(obj_id: str, object_scale: float, sim_z_yaw_deg: float) -> dict:
    override = OBJECT_ROTATION_OVERRIDES.get(obj_id)
    obj_orientation = list(OBJECT_ORIENTATION)
    usd_path = _find_obj_usd_path(obj_id)
    meta_path = usd_path.replace(".usd", "_meta.json") if usd_path else ""
    if meta_path and os.path.exists(meta_path):
        meta = _load_json(Path(meta_path))
        z_offset = float(meta.get("z_offset_m", 0.075 * object_scale))
    elif isinstance(override, dict) and "z_offset" in override:
        z_offset = float(override["z_offset"])
    else:
        z_offset = 0.075 * object_scale
    if isinstance(override, dict) and "rotation" in override:
        obj_orientation = list(override["rotation"])
    obj_orientation[2] = float(obj_orientation[2]) + float(sim_z_yaw_deg)
    pos = list(OBJECT_POSITION)
    pos[2] += z_offset
    return {"pos": pos, "ori": obj_orientation, "z_offset": z_offset, "usd_path": usd_path or ""}


def _make_transform(pos, euler_xyz_deg) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = Rotation.from_euler("xyz", euler_xyz_deg, degrees=True).as_matrix()
    t[:3, 3] = np.asarray(pos, dtype=np.float64)
    return t


def _resolve_metric_dataset(obj_id: str, dataset: str | None) -> str | None:
    """Infer oakink/ycb/... for scale.json; never use placeholder 'evaluation'."""
    return _resolve_metric_dataset_mp(obj_id, dataset)


def _resolve_mesh_path(obj_id: str, mesh_root: str, dataset: str | None) -> Path:
    return _resolve_mesh_path_mp(obj_id, mesh_root, dataset)


def _prepare_mesh(
    *,
    obj_id: str,
    mesh_path: Path,
    dataset: str | None,
    num_points: int,
    seed: int,
    target_max_extent: float,
    auto_extent_lo: float,
    auto_extent_hi: float,
    min_scale_factor: float,
) -> dict:
    mesh = load_triangle_mesh(mesh_path)
    try:
        from tools import random_grasp_sampler as rgs

        sf = rgs.read_scale_factor(obj_id, dataset or None)
        if rgs.apply_metric_scale_to_mesh(obj_id, dataset or None) and abs(sf - 1.0) > 1e-8:
            mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) * float(sf)).astype(np.float64)
    except Exception as exc:
        print(f"  WARNING: rotated SAM3D scale lookup failed for {obj_id}: {exc}", flush=True)

    mesh, scale_report = rescale_mesh_for_v6(
        mesh,
        target_max_extent=target_max_extent,
        scale_mode="never",
        extent_lo=auto_extent_lo,
        extent_hi=auto_extent_hi,
        min_scale_factor=min_scale_factor,
        center_mesh=False,
    )
    scale_report.mode = "sam3d_rotated_metric"
    points, normals = sample_mesh_points(mesh, num_points, seed)
    return {
        "mesh": mesh,
        "points": points,
        "normals": normals,
        "scale_report": scale_report,
    }


def _build_condition_tensor(points: np.ndarray, normals: np.ndarray, affordance: np.ndarray) -> torch.Tensor:
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


def _sample_pdm_batch(
    pdm: PDM,
    stats: dict,
    condition: torch.Tensor,
    *,
    n_samples: int,
    ddim_steps: int,
    z_yaw_deg: float,
    device: torch.device,
) -> np.ndarray:
    pose_mean = stats["pose_mean"].to(device)
    pose_std = stats["pose_std"].to(device)
    if not pdm.config.use_yaw_condition:
        raise ValueError("batch_pdm_candidates expects a yaw-conditioned PDM checkpoint")
    yaw = torch.from_numpy(yaw_feature_from_deg(float(z_yaw_deg))).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    cond = condition.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pose_norm = pdm.sample(cond, yaw=yaw, n_samples=n_samples, ddim_steps=ddim_steps)
        pose = pose_norm * pose_std.unsqueeze(0) + pose_mean.unsqueeze(0)
    return pose.cpu().numpy().astype(np.float32)


def _grasp_footprint_object(cmd, gripper_width: float) -> dict[str, np.ndarray]:
    """TCP, wrist, and finger tips in object_mesh frame (same as metric mesh)."""
    tcp = np.asarray(cmd.position, dtype=np.float64)
    r_cmd = np.asarray(cmd.rotation, dtype=np.float64)
    executed = command_to_executed(cmd.position, cmd.rotation)
    wrist = np.asarray(executed.position, dtype=np.float64)
    finger = r_cmd[:, 0]
    half_width = max(float(gripper_width) * 0.5, 0.001)
    left_tip = tcp - finger * half_width
    right_tip = tcp + finger * half_width
    return {
        "tcp": tcp,
        "wrist": wrist,
        "finger": finger,
        "left_tip": left_tip,
        "right_tip": right_tip,
    }


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _score_surface_nearness(d_left: float, d_right: float, sigma: float) -> float:
    sigma = max(1e-6, float(sigma))
    return float(np.exp(-((d_left * d_left + d_right * d_right) / (2.0 * sigma * sigma))))


def _score_width(width: float, *, center: float, sigma: float, wmin: float, wmax: float) -> float:
    w = float(width)
    if not (float(wmin) <= w <= float(wmax)):
        return 0.0
    sigma = max(1e-6, float(sigma))
    return float(np.exp(-((w - float(center)) * (w - float(center))) / (2.0 * sigma * sigma)))


def _score_antipodal(n_left: np.ndarray, n_right: np.ndarray, finger_dir: np.ndarray) -> float:
    nl = np.asarray(n_left, dtype=np.float64).reshape(3)
    nr = np.asarray(n_right, dtype=np.float64).reshape(3)
    f = np.asarray(finger_dir, dtype=np.float64).reshape(3)
    alpha = _clamp01(float(-np.dot(nl, nr)))
    beta = _clamp01(float(np.dot(nl, f))) * _clamp01(float(-np.dot(nr, f)))
    return float(np.sqrt(alpha * beta))


def _score_approach(approach_world_z: float, *, zmin: float, zmax: float) -> float:
    denom = float(zmax) - float(zmin)
    if abs(denom) < 1e-9:
        return 0.0
    return _clamp01((float(zmax) - float(approach_world_z)) / denom)


def _score_clearance(delta_z: float, *, lam: float) -> float:
    if float(delta_z) <= 0.0:
        return 0.0
    lam = max(1e-6, float(lam))
    return float(1.0 - np.exp(-float(delta_z) / lam))


def _nearest_points_and_normals(
    mesh: trimesh.Trimesh, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nearest_points, normals, distances) for each query point."""
    q = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    closest, dist, tri_id = trimesh.proximity.closest_point(mesh, q)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[np.asarray(tri_id, dtype=np.int64)]
    return np.asarray(closest, dtype=np.float64), normals, np.asarray(dist, dtype=np.float64)


def _object_points_to_world(
    points_obj: np.ndarray, t_world_obj: np.ndarray, object_scale: float
) -> np.ndarray:
    scaled = np.asarray(points_obj, dtype=np.float64) * float(object_scale)
    out = []
    for p in scaled:
        out.append((t_world_obj @ np.append(p, 1.0))[:3])
    return np.stack(out, axis=0)


def _hard_gate_pose(
    *,
    mesh: trimesh.Trimesh,
    pose9: np.ndarray,
    obj_id: str,
    z_yaw_deg: float,
    object_scale: float,
    gripper_width: float,
    table_margin: float,
    t_world_obj: np.ndarray | None = None,
) -> tuple[bool, str, dict, np.ndarray]:
    cmd = pose9_to_command(pose9)
    pose9_out = np.asarray(pose9, dtype=np.float32).copy()
    if t_world_obj is None:
        placement = _resolve_object_placement(obj_id, object_scale, z_yaw_deg)
        t_world_obj = _make_transform(placement["pos"], placement["ori"])

    def _check(cmd_pose) -> tuple[bool, str, dict]:
        fp = _grasp_footprint_object(cmd_pose, gripper_width)
        tcp_w, wrist_w, left_w, right_w = _object_points_to_world(
            np.stack([fp["tcp"], fp["wrist"], fp["left_tip"], fp["right_tip"]]),
            t_world_obj,
            object_scale,
        )
        rot_world_cmd = t_world_obj[:3, :3] @ np.asarray(cmd_pose.rotation, dtype=np.float64)
        hand_up_world = (rot_world_cmd @ R_ADAPT)[:, 1]
        approach_world = rot_world_cmd[:, 2]
        z_values = {
            "tcp_z": float(tcp_w[2]),
            "wrist_z": float(wrist_w[2]),
            "left_tip_z": float(left_w[2]),
            "right_tip_z": float(right_w[2]),
        }
        min_z = TABLE_TOP_Z + float(table_margin)
        if min(z_values.values()) < min_z:
            return False, "pose_pokes_table", {
                **z_values,
            }
        meta = {
            "hand_up_z": float(hand_up_world[2]),
            "approach_z": float(approach_world[2]),
            **z_values,
        }
        if float(hand_up_world[2]) <= 0.0:
            return False, "hand_upside_down", meta
        return True, "", meta

    ok, reason, meta = _check(cmd)
    if ok:
        return True, "", meta, pose9_out

    if reason == "hand_upside_down":
        flip = np.diag([-1.0, -1.0, 1.0])
        rot_cmd_fixed = np.asarray(cmd.rotation, dtype=np.float64) @ flip
        cmd_fixed = CommandPose(
            position=cmd.position,
            rotation=rot_cmd_fixed.astype(np.float32),
        )
        ok2, reason2, meta2 = _check(cmd_fixed)
        if ok2:
            pose9_out[:3] = np.asarray(cmd.position, dtype=np.float32)
            pose9_out[3:9] = rotation_to_6d(rot_cmd_fixed).astype(np.float32)
            meta2 = dict(meta2)
            meta2["hard_gate_pose_flipped"] = True
            meta2["pre_flip_hand_up_z"] = float(meta.get("hand_up_z", 0.0))
            return True, "", meta2, pose9_out
        return False, reason2, meta2, pose9_out

    return False, reason, meta, pose9_out


def _postprocess_hdf5(path: Path, stats: dict, selected_meta: list[dict]) -> None:
    with h5py.File(path, "a") as f:
        meta = f.require_group("metadata")
        for key, value in stats.items():
            if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                meta.attrs[key] = value
        cg = f.get("candidates")
        if cg is not None:
            for idx, row in enumerate(selected_meta):
                key = f"candidate_{idx}"
                if key not in cg:
                    continue
                ci = cg[key]
                ci.attrs["source_batch"] = int(row.get("source_batch", -1))
                ci.attrs["source_index"] = int(row.get("source_index", -1))
                ci.attrs["hard_gate_pass"] = bool(row.get("hard_gate_pass", False))
                ci.attrs["hard_gate_forced_fill"] = bool(row.get("hard_gate_forced_fill", False))
                ci.attrs["hard_gate_pose_flipped"] = bool(row.get("hard_gate_pose_flipped", False))
                if row.get("hard_gate_reject_reason"):
                    ci.attrs["hard_gate_reject_reason"] = str(row["hard_gate_reject_reason"])


def _task_eval_seed(args: argparse.Namespace, obj_id: str, yaw: float) -> int | None:
    if getattr(args, "eval_seed", None) is None:
        return None
    from evaluation.randomness import mix_eval_seed

    return mix_eval_seed(int(args.eval_seed), "pdm_task", obj_id, int(round(float(yaw))) % 360)


def _generate_one(task: dict, args: argparse.Namespace, models: dict, device: torch.device) -> dict:
    obj_id = str(task["obj_id"])
    yaw = float(task["z_yaw_deg"])
    target = int(task["target_candidates"])
    out_path = Path(task["output_hdf5"]).expanduser().resolve()
    metric_ds = _resolve_metric_dataset(obj_id, args.dataset)
    mesh_path = Path(task.get("mesh_path") or _resolve_mesh_path(obj_id, args.mesh_root, metric_ds))
    task_seed = _task_eval_seed(args, obj_id, yaw)
    seed = int(task_seed) if task_seed is not None else secrets.randbits(31)
    prepared = _prepare_mesh(
        obj_id=obj_id,
        mesh_path=mesh_path,
        dataset=metric_ds,
        num_points=args.num_points,
        seed=seed,
        target_max_extent=args.target_max_extent,
        auto_extent_lo=args.auto_extent_lo,
        auto_extent_hi=args.auto_extent_hi,
        min_scale_factor=args.min_scale_factor,
    )
    pred = predict_heatmap_batch(
        models["affordance"],
        prepared["points"][None, ...],
        prepared["normals"][None, ...],
        device,
    )
    if pred.ndim == 2:
        pred = pred[0]
    condition = _build_condition_tensor(prepared["points"], prepared["normals"], pred.astype(np.float32))
    batch_size = max(1, int(args.batch_multiplier) * target)
    accepted: list[tuple[np.ndarray, dict]] = []
    rejected: list[tuple[np.ndarray, dict]] = []
    all_sampled = 0
    batches_run = 0
    placement = _resolve_object_placement(obj_id, float(args.object_scale), yaw)
    t_world_obj = _make_transform(placement["pos"], placement["ori"])

    max_batches = int(args.max_batches)
    filtering_enabled = not bool(args.no_filtering)
    pool_target = (
        int(max(1, int(args.filter_pool_multiplier)) * target) if filtering_enabled else int(target)
    )
    for batch_idx in range(max_batches):
        batches_run = batch_idx + 1
        batch_no = batch_idx + 1
        print(
            f"[batch-candidates] obj={obj_id} yaw={yaw:.0f} "
            f"batch={batch_no}/{max_batches} pdm_sample_start n={batch_size}",
            flush=True,
        )
        t_pdm0 = time.perf_counter()
        if task_seed is not None:
            torch.manual_seed(int(task_seed) % (2**31 - 1))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(task_seed) % (2**31 - 1))
        poses = _sample_pdm_batch(
            models["pdm"],
            models["stats"],
            condition,
            n_samples=batch_size,
            ddim_steps=args.ddim_steps,
            z_yaw_deg=yaw,
            device=device,
        )
        pdm_elapsed_s = time.perf_counter() - t_pdm0
        print(
            f"[batch-candidates] obj={obj_id} yaw={yaw:.0f} "
            f"batch={batch_no}/{max_batches} pdm_done n={len(poses)} "
            f"elapsed_s={pdm_elapsed_s:.1f}",
            flush=True,
        )
        batch_pass = 0
        t_gate0 = time.perf_counter()
        use_hard_gate = filtering_enabled and (not bool(args.no_hard_gate))
        for local_idx, pose9 in enumerate(poses):
            all_sampled += 1
            if use_hard_gate:
                ok, reason, gate_meta, pose9_checked = _hard_gate_pose(
                    mesh=prepared["mesh"],
                    pose9=pose9,
                    obj_id=obj_id,
                    z_yaw_deg=yaw,
                    object_scale=float(args.object_scale),
                    gripper_width=float(args.gripper_width),
                    table_margin=float(args.table_margin),
                    t_world_obj=t_world_obj,
                )
            else:
                ok = True
                reason = ""
                gate_meta = {}
                pose9_checked = np.asarray(pose9, dtype=np.float32)
            row = {
                "source_batch": batch_idx,
                "source_index": local_idx,
                "hard_gate_pass": ok if use_hard_gate else True,
                "hard_gate_forced_fill": False,
                "hard_gate_reject_reason": reason,
                **gate_meta,
            }
            if ok:
                batch_pass += 1
                accepted.append((pose9_checked, row))
            else:
                rejected.append((pose9, row))
        gate_elapsed_s = time.perf_counter() - t_gate0
        gate_tag = "gate_done" if use_hard_gate else "gate_skipped"
        print(
            f"[batch-candidates] obj={obj_id} yaw={yaw:.0f} "
            f"batch={batch_no}/{max_batches} {gate_tag} batch_pass={batch_pass}/{len(poses)} "
            f"total_pass={len(accepted)}/{pool_target} sampled={all_sampled} "
            f"gate_elapsed_s={gate_elapsed_s:.1f} pdm_elapsed_s={pdm_elapsed_s:.1f}",
            flush=True,
        )
        if len(accepted) >= pool_target:
            break

    rng = np.random.default_rng(task_seed) if task_seed is not None else np.random.default_rng()
    chosen: list[tuple[np.ndarray, dict]] = []
    if not filtering_enabled:
        # Legacy behavior: do not filter or score; take the first target poses.
        chosen = accepted[:target]
        if len(chosen) < target:
            need = target - len(chosen)
            if need > 0 and rejected:
                indices = rng.choice(len(rejected), size=min(need, len(rejected)), replace=False)
                for i in indices:
                    pose9, row = rejected[int(i)]
                    row = dict(row)
                    row["hard_gate_forced_fill"] = True
                    chosen.append((pose9, row))
    else:
        # Filtering: pool accepted poses, then score and take top-K.
        pool = accepted[:pool_target]
        scored: list[tuple[float, np.ndarray, dict]] = []
        for pose9, row in pool:
            cmd = pose9_to_command(pose9)
            fp = _grasp_footprint_object(cmd, float(args.gripper_width))
            pts_obj = np.stack([fp["left_tip"], fp["right_tip"]], axis=0)
            closest, normals, dists = _nearest_points_and_normals(prepared["mesh"], pts_obj)
            d_left, d_right = float(dists[0]), float(dists[1])
            n_left, n_right = normals[0], normals[1]
            width = float(np.linalg.norm(np.asarray(closest[0]) - np.asarray(closest[1])))

            finger_dir_obj = np.asarray(cmd.rotation[:, 0], dtype=np.float64)
            surf = _score_surface_nearness(d_left, d_right, float(args.filter_surface_sigma))
            sw = _score_width(
                width,
                center=float(args.filter_width_center),
                sigma=float(args.filter_width_sigma),
                wmin=float(args.filter_width_min),
                wmax=float(args.filter_width_max),
            )
            anti = _score_antipodal(n_left, n_right, finger_dir_obj)
            sa = _score_approach(
                float(row.get("approach_z", 0.0)),
                zmin=float(args.filter_approach_z_min),
                zmax=float(args.filter_approach_z_max),
            )
            dz = float(
                min(
                    row.get("tcp_z", 0.0),
                    row.get("wrist_z", 0.0),
                    row.get("left_tip_z", 0.0),
                    row.get("right_tip_z", 0.0),
                )
                - (TABLE_TOP_Z + float(args.table_margin))
            )
            sc = _score_clearance(dz, lam=float(args.filter_clearance_lambda))
            total = (
                float(args.filter_w_surf) * surf
                + float(args.filter_w_width) * sw
                + float(args.filter_w_anti) * anti
                + float(args.filter_w_approach) * sa
                + float(args.filter_w_clear) * sc
            )
            row2 = dict(row)
            row2.update(
                {
                    "filter_score": float(total),
                    "filter_surf": float(surf),
                    "filter_width": float(sw),
                    "filter_anti": float(anti),
                    "filter_approach": float(sa),
                    "filter_clearance": float(sc),
                    "filter_width_m": float(width),
                    "filter_tip_dist_left_m": float(d_left),
                    "filter_tip_dist_right_m": float(d_right),
                }
            )
            scored.append((float(total), pose9, row2))

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [(pose9, row) for _s, pose9, row in scored[:target]]

        # If pool was short (max_batches hit), allow sorting whatever we have.
        if len(chosen) < target:
            need = target - len(chosen)
            if need > 0 and rejected:
                indices = rng.choice(len(rejected), size=min(need, len(rejected)), replace=False)
                for i in indices:
                    pose9, row = rejected[int(i)]
                    row = dict(row)
                    row["hard_gate_forced_fill"] = True
                    chosen.append((pose9, row))

        if len(chosen) < target and chosen:
            while len(chosen) < target:
                pose9, row = chosen[int(rng.integers(0, len(chosen)))]
                row = dict(row)
                row["hard_gate_forced_fill"] = True
                chosen.append((pose9.copy(), row))

    if not chosen:
        raise RuntimeError(f"no PDM candidates generated for {obj_id} yaw={yaw}")

    poses_np = np.stack([x[0] for x in chosen], axis=0).astype(np.float32)
    selected_meta = [x[1] for x in chosen]
    reject_counts: dict[str, int] = {}
    for _pose, row in rejected:
        reason = str(row.get("hard_gate_reject_reason") or "unknown")
        reject_counts[reason] = reject_counts.get(reason, 0) + 1
    write_candidates_hdf5(
        str(out_path),
        obj_id,
        poses_np,
        mesh_path=str(mesh_path),
        gripper_width=float(args.gripper_width),
        dataset=metric_ds or "oakink",
    )
    forced = sum(1 for row in selected_meta if row.get("hard_gate_forced_fill"))
    stats = {
        "z_yaw_deg": yaw,
        "n_target": target,
        "n_selected": len(chosen),
        "hard_gate_enabled": bool(use_hard_gate),
        "filtering_enabled": bool(filtering_enabled),
        "filter_pool_multiplier": int(args.filter_pool_multiplier),
        "filter_pool_target": int(pool_target),
        "hard_gate_pass_count": len(accepted) if use_hard_gate else len(chosen),
        "n_batches_used": batches_run,
        "forced_fill_count": forced,
        "all_sampled_count": all_sampled,
        "batch_multiplier": int(args.batch_multiplier),
        "max_batches": int(args.max_batches),
        "reject_counts_json": json.dumps(reject_counts, sort_keys=True),
    }
    _postprocess_hdf5(out_path, stats, selected_meta)
    return {
        "obj_id": obj_id,
        "z_yaw_deg": yaw,
        "output_hdf5": str(out_path),
        "n_selected": len(chosen),
        "hard_gate_pass_count": len(accepted),
        "forced_fill_count": forced,
        "n_batches_used": stats["n_batches_used"],
        "reject_counts": reject_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch PDM candidate generation for eval_pool")
    p.add_argument("--tasks-json", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument("--mesh-root", required=True)
    p.add_argument(
        "--dataset",
        default=None,
        help="Dataset for scale.json lookup (oakink/ycb/...). Omit to infer per obj_id.",
    )
    add_affordance_checkpoint_args(p)
    p.add_argument("--pdm-checkpoint", type=Path, default=DEFAULT_PDM_CKPT)
    p.add_argument("--pose-stats", type=Path, default=None)
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--num-points", type=int, default=4096)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--batch-multiplier", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=10)
    p.add_argument("--gripper-width", type=float, default=DEFAULT_GRIPPER_WIDTH)
    p.add_argument("--object-scale", type=float, default=1.0)
    p.add_argument("--table-margin", type=float, default=HARD_GATE_TABLE_MARGIN)
    p.add_argument("--target-max-extent", type=float, default=0.28)
    p.add_argument("--auto-extent-lo", type=float, default=0.02)
    p.add_argument("--auto-extent-hi", type=float, default=0.80)
    p.add_argument("--min-scale-factor", type=float, default=1e-6)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="When set, derive per-task mesh/PDM/forced-fill seeds from this eval master seed.",
    )
    p.add_argument(
        "--no-hard-gate",
        action="store_true",
        help="Skip the two hard gates (table poke + hand-up). Only effective when filtering is enabled.",
    )
    p.add_argument(
        "--no-filtering",
        action="store_true",
        help="Disable filtering (hard gates + scoring). Default enables filtering.",
    )
    p.add_argument("--filter-pool-multiplier", type=int, default=FILTER_POOL_MULTIPLIER)
    p.add_argument("--filter-surface-sigma", type=float, default=FILTER_SURFACE_SIGMA_M)
    p.add_argument("--filter-width-center", type=float, default=FILTER_WIDTH_CENTER_M)
    p.add_argument("--filter-width-sigma", type=float, default=FILTER_WIDTH_SIGMA_M)
    p.add_argument("--filter-width-min", type=float, default=FILTER_WIDTH_MIN_M)
    p.add_argument("--filter-width-max", type=float, default=FILTER_WIDTH_MAX_M)
    p.add_argument("--filter-clearance-lambda", type=float, default=FILTER_CLEARANCE_LAMBDA_M)
    p.add_argument("--filter-approach-z-min", type=float, default=FILTER_APPROACH_Z_MIN)
    p.add_argument("--filter-approach-z-max", type=float, default=FILTER_APPROACH_Z_MAX)
    p.add_argument("--filter-w-surf", type=float, default=FILTER_W_SURF)
    p.add_argument("--filter-w-width", type=float, default=FILTER_W_WIDTH)
    p.add_argument("--filter-w-anti", type=float, default=FILTER_W_ANTI)
    p.add_argument("--filter-w-approach", type=float, default=FILTER_W_APPROACH)
    p.add_argument("--filter-w-clear", type=float, default=FILTER_W_CLEAR)
    return p


def main() -> None:
    args = build_parser().parse_args()
    tasks = _load_json(args.tasks_json).get("tasks", [])
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    aff_ckpt = resolve_affordance_checkpoint(
        hp_affordance=bool(args.hp_affordance),
        affordance_checkpoint=args.affordance_checkpoint,
    )
    dataset_dir = args.dataset_dir or aff_ckpt.parents[1]
    threshold = default_threshold(str(aff_ckpt), str(dataset_dir.expanduser().resolve()))
    affordance_model, _ = load_model(str(aff_ckpt), device)
    pdm_model, ckpt = PDM.load(str(args.pdm_checkpoint.expanduser().resolve()), device=device)
    stats = ckpt.get("pose_stats")
    if stats is None:
        if args.pose_stats is None:
            raise RuntimeError("PDM checkpoint has no pose_stats; pass --pose-stats")
        stats = torch.load(args.pose_stats.expanduser().resolve(), map_location=device, weights_only=False)
    models = {"affordance": affordance_model, "pdm": pdm_model, "stats": stats}
    rows = []
    print(
        f"[batch-candidates] tasks={len(tasks)} device={device} "
        f"aff_thresh={threshold:.3f}",
        flush=True,
    )
    for task in tasks:
        rows.append(_generate_one(task, args, models, device))
    _write_json(args.output_manifest, {"version": 1, "tasks": rows})
    print(f"[batch-candidates] wrote {args.output_manifest}", flush=True)


if __name__ == "__main__":
    main()

