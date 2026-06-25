#!/usr/bin/env python3
"""
prepare_affordance_executed.py — merged executed pose → PointNet++ 训练 HDF5
============================================================================

接触点优先级（方法 C → B → A）:
  C: approach×finger_dir 平面内沿指长扫描；每站位沿 finger_dir 测表面宽度（**相邻**交点对，
     选 |w−指宽| 最小的站位）
  B: finger_mid 沿 ±finger_dir raycast（C 与记录指宽差太多时 fallback）
  A: finger_mid ± finger_dir * width/2 解析（B 仍失败时）

Human prior（默认开启）:
  - 来源: data_hub/ProcessedData/train_fp_rotated/{dataset}/{obj}.hdf5
  - 旋转: 与 rotated_mesh 相同 Rx(+90°)（见 tools/rotate_training_fp.py）
  - 尺度: OakInk HP 盘内未 scale → × scale.json；ycb_dex_* 盘内已 metric → 不重复乘

每物体 1 样本；不修改 build_dataset.py / gen_m5_training_data.py。

用法:
    python3 tools/prepare_affordance_executed.py --workers 8
    # 默认: affordance_all.h5 + affordance_all_soft.h5 + objects_train_val_split.json
    python3 tools/prepare_affordance_executed.py --write-split   # 额外 4 个 train/val h5
    python3 tools/prepare_affordance_executed.py --obj A01001 --qc-vis
    python3 tools/prepare_affordance_executed.py --no-hp --no-soft
    # 仅从已有 binary h5 重算 soft（不跑 merged）:
    python3 tools/prepare_affordance_executed.py --export-soft-only --dataset-dir output/affordance_no_rot_executed --overwrite
    # 已有 train/val 合并为 all（不改动原文件）:
    python3 tools/merge_affordance_h5_splits.py --dataset-dir output/affordance_no_rot_executed
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import h5py
import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "tools"))

from mesh_utils import infer_dataset  # noqa: E402
import random_grasp_sampler as rgs  # noqa: E402
from tools.soft_affordance_gt import export_soft_dataset_dir, write_soft_h5  # noqa: E402

DEFAULT_MERGED_DIR = os.path.join(PROJ, "output", "grasp_collect_no_rot", "merged")
DEFAULT_OUT_DIR = os.path.join(PROJ, "output", "affordance_no_rot_executed")

TCP_OFFSET = 0.105
MIN_GRIPPER_WIDTH = 0.02
MAX_GRIPPER_OPEN = 0.08
WIDTH_DROP_MAX = MAX_GRIPPER_OPEN * 1.5
WIDTH_DROP_MIN = MIN_GRIPPER_WIDTH * 0.5
AXIS_ORTH_WARN = 0.2
RAY_ALONG_APPROACH_MAX = 0.05

# 方法 C：指长扫描
SCAN_STEP_M = 0.003
FINGER_TIP_EXTENT_M = 0.045
FINGER_ROOT_MARGIN_M = 0.01
WIDTH_MISMATCH_ABS = 0.02
WIDTH_MISMATCH_REL = 0.35


@dataclass
class ExecutedGrasp:
    key: str
    name: str
    score: float
    round_id: str
    source_file: str
    wrist: np.ndarray
    approach: np.ndarray
    finger_dir: np.ndarray
    width: float
    finger_mid: np.ndarray
    tip_l: np.ndarray | None = None
    tip_r: np.ndarray | None = None


@dataclass
class GraspContacts:
    grasp_key: str
    name: str
    L: np.ndarray
    R: np.ndarray
    finger_mid: np.ndarray
    width_meas: float
    width_target: float
    width_err: float
    method: str
    source_L: str
    source_R: str
    status: str


@dataclass
class ObjectSample:
    obj_id: str
    dataset: str
    points: np.ndarray
    normals: np.ndarray
    labels: np.ndarray
    human_priors: np.ndarray
    force_center: np.ndarray
    n_grasps_trusted: int = 0
    n_grasps_used: int = 0
    n_grasps_dropped: int = 0
    n_method_c: int = 0
    n_method_b: int = 0
    n_method_a: int = 0
    n_contact_pts: int = 0
    positive_ratio: float = 0.0
    width_meas_mean: float = 0.0
    width_err_mean: float = 0.0
    hp_path: str = ""
    hp_scale_applied: bool = False
    hp_nn_median_cm: float = float("nan")
    hp_positive_ratio: float = 0.0
    contact_rows: list = field(default_factory=list)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def _obj_seed(obj_id: str) -> int:
    h = hashlib.md5(obj_id.encode()).hexdigest()[:8]
    return int(h, 16) % (2**31)


def _round_from_source(path: str) -> str:
    m = re.search(r"round_(\d+)", path or "")
    return m.group(1) if m else "?"


def _width_mismatch(width_meas: float, width_target: float) -> bool:
    err = abs(width_meas - width_target)
    tol = max(WIDTH_MISMATCH_ABS, WIDTH_MISMATCH_REL * max(width_target, 1e-6))
    return err > tol


def is_trusted_grasp(g: h5py.Group) -> bool:
    if "gripper_tips_loc" not in g:
        return False
    if bool(g.attrs.get("gripper_tips_trusted", False)):
        return True
    if str(g.attrs.get("gripper_tips_source", "")) == "legacy_post_lift":
        return False
    return str(g.attrs.get("gripper_tips_snapshot", "at_close")) == "at_close"


def load_rotated_mesh_metric(obj_id: str, dataset: str | None = None) -> tuple[trimesh.Trimesh, str, float]:
    path, sf, ds, apply_scale = rgs.find_obj_mesh(obj_id, dataset, use_legacy_assets=False)
    if path is None:
        raise FileNotFoundError(f"rotated_mesh not found for {obj_id}")
    mesh = trimesh.load(path, force="mesh", process=False)
    if apply_scale and abs(sf - 1.0) > 1e-8:
        mesh = mesh.copy()
        mesh.vertices = (mesh.vertices * float(sf)).astype(np.float64)
    return mesh, ds or infer_dataset(obj_id, dataset), float(sf)


def load_executed_grasps(merged_path: str) -> list[ExecutedGrasp]:
    rows: list[ExecutedGrasp] = []
    with h5py.File(merged_path, "r") as f:
        if "successful_grasps" not in f:
            return rows
        for key in f["successful_grasps"].keys():
            g = f["successful_grasps"][key]
            if not is_trusted_grasp(g):
                continue
            sub = "executed_panda_hand_at_close"
            if sub not in g or "position" not in g[sub]:
                continue
            eg = g[sub]
            width = float(g.attrs.get("finger_width_actual", g.attrs.get("gripper_width", 0.04)))
            if width <= 0:
                continue
            approach = _unit(eg["approach_dir"][:])
            finger_dir = _unit(eg["finger_dir"][:])
            wrist = np.asarray(eg["position"][:], dtype=np.float64)
            finger_mid = wrist + approach * TCP_OFFSET
            tip_l = tip_r = None
            if "gripper_tips_loc" in g:
                tips = np.asarray(g["gripper_tips_loc"][:], dtype=np.float64)
                tip_l, tip_r = tips[0], tips[1]
            rows.append(
                ExecutedGrasp(
                    key=str(key),
                    name=str(g.attrs.get("name", key)),
                    score=float(g.attrs.get("score", 0.0)),
                    round_id=_round_from_source(str(g.attrs.get("source_file", ""))),
                    source_file=str(g.attrs.get("source_file", "")),
                    wrist=wrist,
                    approach=approach,
                    finger_dir=finger_dir,
                    width=width,
                    finger_mid=finger_mid,
                    tip_l=tip_l,
                    tip_r=tip_r,
                )
            )
    return rows


def count_trusted_grasps(merged_path: str) -> int:
    """Fast count for discovery (no full grasp struct load)."""
    with h5py.File(merged_path, "r") as f:
        if "successful_grasps" not in f:
            return 0
        grp = f["successful_grasps"]
        return sum(1 for key in grp.keys() if is_trusted_grasp(grp[key]))


def _prepare_worker(payload: dict) -> tuple[str, ObjectSample | None, str | None]:
    """ProcessPool worker: one object → sample or skip reason."""
    obj_id = payload["obj_id"]
    merged_path = payload["merged_path"]
    try:
        sample = prepare_object(
            obj_id,
            merged_path,
            num_points=int(payload["num_points"]),
            contact_radius=float(payload["contact_radius"]),
            with_hp=bool(payload["with_hp"]),
        )
    except Exception as e:
        return obj_id, None, str(e)
    if sample is None:
        return obj_id, None, "no trusted grasps or all dropped"
    return obj_id, sample, None


def _run_prepare_jobs(
    jobs: list[tuple[str, str, int]],
    merged_dir: str,
    *,
    num_points: int,
    contact_radius: float,
    with_hp: bool,
    workers: int,
) -> tuple[list[ObjectSample], list[str]]:
    """Sequential (workers=1) or per-object multiprocessing."""
    n_jobs = len(jobs)
    samples: list[ObjectSample] = []
    skipped: list[str] = []

    if workers <= 1:
        for i, (obj_id, _ds, _) in enumerate(jobs):
            merged_path = os.path.join(merged_dir, f"{obj_id}_robot_gt_merged.hdf5")
            if not os.path.isfile(merged_path):
                merged_path = os.path.join(merged_dir, f"{obj_id}_merged.hdf5")
            if not os.path.isfile(merged_path):
                skipped.append(f"{obj_id}: no merged file")
                continue
            obj_id_out, sample, err = _prepare_worker({
                "obj_id": obj_id,
                "merged_path": merged_path,
                "num_points": num_points,
                "contact_radius": contact_radius,
                "with_hp": with_hp,
            })
            if err:
                skipped.append(f"{obj_id_out}: {err}")
                continue
            samples.append(sample)
            print(
                f"  [{i+1}/{n_jobs}] {sample.obj_id} ({sample.dataset}): "
                f"used={sample.n_grasps_used}/{sample.n_grasps_trusted} "
                f"C/B/A={sample.n_method_c}/{sample.n_method_b}/{sample.n_method_a} "
                f"pos={sample.positive_ratio*100:.2f}%",
            )
        return samples, skipped

    payloads = []
    for obj_id, _ds, _ in jobs:
        merged_path = os.path.join(merged_dir, f"{obj_id}_robot_gt_merged.hdf5")
        if not os.path.isfile(merged_path):
            merged_path = os.path.join(merged_dir, f"{obj_id}_merged.hdf5")
        if not os.path.isfile(merged_path):
            skipped.append(f"{obj_id}: no merged file")
            continue
        payloads.append({
            "obj_id": obj_id,
            "merged_path": merged_path,
            "num_points": num_points,
            "contact_radius": contact_radius,
            "with_hp": with_hp,
        })

    print(f"  Parallel prepare: {len(payloads)} objects, workers={workers}")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_prepare_worker, p) for p in payloads]
        for fut in as_completed(futures):
            obj_id_out, sample, err = fut.result()
            done += 1
            if err:
                skipped.append(f"{obj_id_out}: {err}")
                continue
            samples.append(sample)
            print(
                f"  [{done}/{len(payloads)}] {sample.obj_id} ({sample.dataset}): "
                f"used={sample.n_grasps_used}/{sample.n_grasps_trusted} "
                f"C/B/A={sample.n_method_c}/{sample.n_method_b}/{sample.n_method_a} "
                f"pos={sample.positive_ratio*100:.2f}%",
            )
    return samples, skipped


def _plane_scan_axis(g: ExecutedGrasp) -> np.ndarray:
    """指长方向：在 approach×finger_dir 平面内，从指尖侧指向指根侧。"""
    x = np.cross(g.approach, g.finger_dir)
    x = _unit(x)
    toward_root = g.wrist - g.finger_mid
    if float(np.dot(toward_root, x)) < 0:
        x = -x
    return x


def _scan_s_interval(g: ExecutedGrasp, x_scan: np.ndarray) -> tuple[float, float]:
    """只在手指长度那一段 [s_lo, s_hi]（相对 finger_mid）。"""
    anchors = [g.finger_mid, g.wrist]
    if g.tip_l is not None and g.tip_r is not None:
        anchors.extend([g.tip_l, g.tip_r])
    projs = [float(np.dot(a - g.finger_mid, x_scan)) for a in anchors]
    s_lo = min(projs + [-FINGER_TIP_EXTENT_M])
    s_hi = max(projs + [FINGER_ROOT_MARGIN_M])
    s_lo = min(s_lo, -FINGER_TIP_EXTENT_M)
    s_hi = max(s_hi, FINGER_ROOT_MARGIN_M)
    if s_hi - s_lo < SCAN_STEP_M * 4:
        s_lo, s_hi = -FINGER_TIP_EXTENT_M, FINGER_ROOT_MARGIN_M + 0.03
    return s_lo, s_hi


def _surface_width_at_station(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    finger_dir: np.ndarray,
    width_target: float,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """过 origin、沿 ±finger_dir 与 mesh 求交；仅 **相邻** 交点对，选 |w−width_target| 最小。"""
    origin = np.asarray(origin, dtype=np.float64)
    fd = _unit(finger_dir)
    hits_list = []
    for d in (fd, -fd):
        try:
            locs, _, _ = mesh.ray.intersects_location(
                ray_origins=[origin],
                ray_directions=[d],
            )
        except Exception:
            continue
        if len(locs):
            hits_list.append(np.asarray(locs, dtype=np.float64))
    if not hits_list:
        return None, None, 0.0
    hits = np.vstack(hits_list)
    if len(hits) < 2:
        return None, None, 0.0

    t = np.dot(hits - origin, fd)
    order = np.argsort(t)
    hits = hits[order]

    best_err = float("inf")
    best_l = best_r = None
    best_w = 0.0
    for i in range(len(hits) - 1):
        p_l, p_r = hits[i], hits[i + 1]
        w3 = float(np.linalg.norm(p_r - p_l))
        if w3 < WIDTH_DROP_MIN or w3 > WIDTH_DROP_MAX:
            continue
        err = abs(w3 - width_target)
        if err < best_err:
            best_err = err
            best_w = w3
            best_l, best_r = p_l, p_r
    if best_l is None:
        return None, None, 0.0
    return best_l.astype(np.float32), best_r.astype(np.float32), best_w


def contacts_method_c(mesh: trimesh.Trimesh, g: ExecutedGrasp) -> GraspContacts | None:
    x_scan = _plane_scan_axis(g)
    s_lo, s_hi = _scan_s_interval(g, x_scan)
    stations = np.arange(s_lo, s_hi + SCAN_STEP_M * 0.5, SCAN_STEP_M)

    best_err = float("inf")
    best_l = best_r = None
    best_w = 0.0
    for s in stations:
        q = g.finger_mid + x_scan * s
        l_s, r_s, w_s = _surface_width_at_station(mesh, q, g.finger_dir, g.width)
        if l_s is None or w_s <= 0:
            continue
        err = abs(w_s - g.width)
        if err < best_err:
            best_err = err
            best_w = w_s
            best_l, best_r = l_s, r_s

    if best_l is None:
        return None

    return GraspContacts(
        grasp_key=g.key,
        name=g.name,
        L=best_l,
        R=best_r,
        finger_mid=g.finger_mid.astype(np.float32),
        width_meas=best_w,
        width_target=g.width,
        width_err=abs(best_w - g.width),
        method="C",
        source_L="surface_scan",
        source_R="surface_scan",
        status="ok",
    )


def contacts_method_a(g: ExecutedGrasp) -> GraspContacts:
    half_w = g.width / 2.0
    l_a = (g.finger_mid + g.finger_dir * half_w).astype(np.float32)
    r_a = (g.finger_mid - g.finger_dir * half_w).astype(np.float32)
    w = float(np.linalg.norm(l_a - r_a))
    return GraspContacts(
        grasp_key=g.key,
        name=g.name,
        L=l_a,
        R=r_a,
        finger_mid=g.finger_mid.astype(np.float32),
        width_meas=w,
        width_target=g.width,
        width_err=abs(w - g.width),
        method="A",
        source_L="analytic",
        source_R="analytic",
        status="ok",
    )


def _raycast_contact(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    direction: np.ndarray,
    ref: np.ndarray,
    approach: np.ndarray,
) -> np.ndarray | None:
    direction = _unit(direction)
    try:
        locs, _, _ = mesh.ray.intersects_location(
            ray_origins=[np.asarray(origin, dtype=np.float64)],
            ray_directions=[direction],
        )
    except Exception:
        return None
    if len(locs) == 0:
        return None
    locs = np.asarray(locs, dtype=np.float64)
    dists = np.linalg.norm(locs - ref, axis=1)
    mask = dists > 1e-4
    if not np.any(mask):
        return None
    locs = locs[mask]
    dists = dists[mask]
    along = np.abs(np.dot(locs - ref, approach))
    in_band = along <= RAY_ALONG_APPROACH_MAX
    if np.any(in_band):
        idx = int(np.argmin(dists[in_band]))
        return locs[in_band][idx].astype(np.float32)
    return locs[int(np.argmin(dists))].astype(np.float32)


def contacts_method_b(mesh: trimesh.Trimesh, g: ExecutedGrasp) -> GraspContacts:
    half_w = g.width / 2.0
    l_a = (g.finger_mid + g.finger_dir * half_w).astype(np.float32)
    r_a = (g.finger_mid - g.finger_dir * half_w).astype(np.float32)

    l_r = _raycast_contact(mesh, g.finger_mid, g.finger_dir, g.finger_mid, g.approach)
    r_r = _raycast_contact(mesh, g.finger_mid, -g.finger_dir, g.finger_mid, g.approach)

    src_l = "raycast" if l_r is not None else "analytic"
    src_r = "raycast" if r_r is not None else "analytic"
    l_f = l_r if l_r is not None else l_a
    r_f = r_r if r_r is not None else r_a

    w = float(np.linalg.norm(l_f - r_f))
    return GraspContacts(
        grasp_key=g.key,
        name=g.name,
        L=l_f,
        R=r_f,
        finger_mid=g.finger_mid.astype(np.float32),
        width_meas=w,
        width_target=g.width,
        width_err=abs(w - g.width),
        method="B",
        source_L=src_l,
        source_R=src_r,
        status="ok",
    )


def _finalize_contact_status(c: GraspContacts) -> GraspContacts:
    orth = 0.0
    if c.status == "ok":
        pass
    if c.width_meas > WIDTH_DROP_MAX or c.width_meas < WIDTH_DROP_MIN:
        c.status = "dropped"
    return c


def contacts_for_grasp(mesh: trimesh.Trimesh, g: ExecutedGrasp) -> GraspContacts:
    """C 优先；指宽差过大 → B；B 不可用 → A。"""
    c_res = contacts_method_c(mesh, g)
    if c_res is not None and not _width_mismatch(c_res.width_meas, g.width):
        return _finalize_contact_status(c_res)

    if c_res is not None:
        b_try = contacts_method_b(mesh, g)
        b_try.method = "B_fallback_from_C"
        out = _finalize_contact_status(b_try)
        if out.status != "dropped":
            return out

    b_res = contacts_method_b(mesh, g)
    out = _finalize_contact_status(b_res)
    if out.status != "dropped":
        if c_res is None:
            out.method = "B"
        else:
            out.method = "B_fallback_from_C"
        return out

    a_res = contacts_method_a(g)
    a_res.method = "A_fallback"
    return _finalize_contact_status(a_res)


def load_human_prior_metric_frame(
    obj_id: str,
    dataset: str,
    mesh: trimesh.Trimesh,
    scale_factor: float,
) -> tuple[np.ndarray | None, np.ndarray | None, str, bool, float]:
    """
    HP 与 load_rotated_mesh_metric 同一物体系（rotated + metric）。

    约定（与 random_grasp_sampler / vis_rotated_mesh_hp 一致）:
      - 读 train_fp_rotated（已 Rx+90°，与 SAM3D rotated_mesh 同旋转）
      - OakInk: HP 点云再 × scale.json
      - ycb_dex_*: HP 在盘上已是 metric，不再乘 scale
    """
    hp_pc, hp_labels, hp_path = rgs.load_human_prior(
        obj_id, dataset=dataset, use_rotated_hp=True,
    )
    if hp_pc is None or hp_labels is None:
        return None, None, "", False, float("nan")

    scale_applied = bool(rgs.apply_metric_scale_to_hp_on_load(obj_id, dataset))
    if scale_applied:
        hp_pc = rgs.scale_hp_to_metric(hp_pc, scale_factor)

    dists_m = cKDTree(mesh.vertices.astype(np.float64)).query(
        hp_pc.astype(np.float64), k=1,
    )[0]
    nn_median_cm = float(np.median(dists_m) * 100.0)
    return hp_pc, hp_labels, hp_path or "", scale_applied, nn_median_cm


def map_human_prior_to_points(
    obj_id: str,
    dataset: str,
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    scale_factor: float,
) -> tuple[np.ndarray, dict]:
    """KNN：mesh 采样点 ← 最近 HP 点的 human_prior 标量。"""
    meta = {
        "hp_path": "",
        "hp_scale_applied": False,
        "hp_nn_median_cm": float("nan"),
        "hp_positive_ratio": 0.0,
        "hp_missing": True,
    }
    hp_pc, hp_labels, hp_path, scale_applied, nn_cm = load_human_prior_metric_frame(
        obj_id, dataset, mesh, scale_factor,
    )
    if hp_pc is None:
        return np.zeros(len(points), dtype=np.float32), meta

    tree = cKDTree(hp_pc.astype(np.float64))
    _, nn = tree.query(points.astype(np.float64), k=1)
    mapped = hp_labels[nn].astype(np.float32)
    meta.update({
        "hp_path": hp_path,
        "hp_scale_applied": scale_applied,
        "hp_nn_median_cm": nn_cm,
        "hp_positive_ratio": float((mapped > 0.05).mean()),
        "hp_missing": False,
    })
    return mapped, meta


def prepare_object(
    obj_id: str,
    merged_path: str,
    *,
    num_points: int,
    contact_radius: float,
    with_hp: bool,
    mesh: trimesh.Trimesh | None = None,
) -> ObjectSample | None:
    grasps = load_executed_grasps(merged_path)
    n_trusted = len(grasps)
    if n_trusted == 0:
        return None

    if mesh is None:
        mesh, dataset, sf = load_rotated_mesh_metric(obj_id)
    else:
        dataset = infer_dataset(obj_id)
        sf = rgs.read_scale_factor(obj_id, dataset)

    contacts: list[GraspContacts] = []
    for g in grasps:
        contacts.append(contacts_for_grasp(mesh, g))

    used = [c for c in contacts if c.status != "dropped"]
    dropped = len(contacts) - len(used)
    if not used:
        return None

    contact_pts = np.vstack([np.stack([c.L, c.R]) for c in used]).astype(np.float64)
    contact_mids = np.stack(
        [0.5 * (c.L.astype(np.float64) + c.R.astype(np.float64)) for c in used],
        axis=0,
    )

    points, face_idx = trimesh.sample.sample_surface(mesh, num_points, seed=_obj_seed(obj_id))
    points = points.astype(np.float32)
    normals = mesh.face_normals[face_idx].astype(np.float32)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    tree = cKDTree(contact_pts)
    dists, _ = tree.query(points.astype(np.float64))
    labels = (dists < contact_radius).astype(np.float32)

    force_center = contact_mids.mean(axis=0).astype(np.float32)
    hp_meta = {
        "hp_path": "",
        "hp_scale_applied": False,
        "hp_nn_median_cm": float("nan"),
        "hp_positive_ratio": 0.0,
    }
    if with_hp:
        human_priors, hp_meta = map_human_prior_to_points(
            obj_id, dataset, points, mesh, sf,
        )
    else:
        human_priors = np.zeros(num_points, dtype=np.float32)

    n_c = sum(1 for c in used if c.method.startswith("C"))
    n_b = sum(1 for c in used if "B" in c.method)
    n_a = sum(1 for c in used if "A" in c.method)

    return ObjectSample(
        obj_id=obj_id,
        dataset=dataset,
        points=points,
        normals=normals,
        labels=labels,
        human_priors=human_priors,
        force_center=force_center,
        n_grasps_trusted=n_trusted,
        n_grasps_used=len(used),
        n_grasps_dropped=dropped,
        n_method_c=n_c,
        n_method_b=n_b,
        n_method_a=n_a,
        n_contact_pts=len(contact_pts),
        positive_ratio=float(labels.mean()),
        width_meas_mean=float(np.mean([c.width_meas for c in used])),
        width_err_mean=float(np.mean([c.width_err for c in used])),
        hp_path=hp_meta.get("hp_path", ""),
        hp_scale_applied=bool(hp_meta.get("hp_scale_applied", False)),
        hp_nn_median_cm=float(hp_meta.get("hp_nn_median_cm", float("nan"))),
        hp_positive_ratio=float(hp_meta.get("hp_positive_ratio", 0.0)),
        contact_rows=used,
    )


def discover_trainable(merged_dir: str) -> list[tuple[str, str, int]]:
    import glob

    jobs: list[tuple[str, str, int]] = []
    for path in sorted(glob.glob(os.path.join(merged_dir, "*_merged.hdf5"))):
        obj_id = os.path.basename(path).replace("_robot_gt_merged.hdf5", "").replace("_merged.hdf5", "")
        n = count_trusted_grasps(path)
        if n > 0:
            jobs.append((obj_id, infer_dataset(obj_id), n))
    return jobs


def pick_qc_vis_objects(
    samples: list[ObjectSample],
    *,
    min_oakink: int = 6,
    min_dexycb: int = 6,
    max_total: int = 16,
) -> list[ObjectSample]:
    """QC 图必须同时包含 oakink 与 dexycb。"""
    oak = sorted(
        [s for s in samples if s.dataset == "oakink"],
        key=lambda s: -s.n_grasps_used,
    )
    dex = sorted(
        [s for s in samples if s.dataset == "dexycb"],
        key=lambda s: -s.n_grasps_used,
    )
    picked: list[ObjectSample] = []
    seen: set[str] = set()
    for s in oak[:min_oakink]:
        picked.append(s)
        seen.add(s.obj_id)
    for s in dex[:min_dexycb]:
        if s.obj_id not in seen:
            picked.append(s)
            seen.add(s.obj_id)
    rest = sorted(
        [s for s in samples if s.obj_id not in seen],
        key=lambda s: -s.n_grasps_used,
    )
    for s in rest:
        if len(picked) >= max_total:
            break
        picked.append(s)
    return picked[:max_total]


def split_objects(
    obj_ids: list[str],
    train_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    ids = sorted(obj_ids)
    if len(ids) == 0:
        return [], []
    if len(ids) == 1:
        return ids, ids
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * (1.0 - train_ratio))))
    n_val = min(n_val, len(ids) - 1)
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids


def write_affordance_h5(
    path: str,
    samples: list[ObjectSample],
    *,
    num_points: int,
    contact_radius: float,
    extra_meta: dict,
) -> None:
    if not samples:
        raise ValueError(f"no samples to write: {path}")

    points = np.stack([s.points for s in samples], axis=0)
    normals = np.stack([s.normals for s in samples], axis=0)
    labels = np.stack([s.labels for s in samples], axis=0)
    hp = np.stack([s.human_priors for s in samples], axis=0)
    fc = np.stack([s.force_center for s in samples], axis=0)
    obj_ids = np.array([s.obj_id for s in samples], dtype="S32")
    categories = np.array([s.dataset for s in samples], dtype="S20")
    intents = np.array(["grasp"] * len(samples), dtype="S20")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["num_samples"] = len(samples)
        meta.attrs["num_points"] = num_points
        meta.attrs["contact_radius"] = contact_radius
        meta.attrs["augmentations"] = 1
        for k, v in extra_meta.items():
            try:
                meta.attrs[k] = v
            except TypeError:
                meta.attrs[k] = str(v)

        grp = f.create_group("data")
        grp.create_dataset("points", data=points, compression="gzip", compression_opts=4)
        grp.create_dataset("normals", data=normals, compression="gzip", compression_opts=4)
        grp.create_dataset("human_priors", data=hp, compression="gzip", compression_opts=4)
        grp.create_dataset("labels", data=labels, compression="gzip", compression_opts=4)
        grp.create_dataset("force_centers", data=fc, compression="gzip", compression_opts=4)
        grp.create_dataset("obj_ids", data=obj_ids)
        grp.create_dataset("categories", data=categories)
        grp.create_dataset("intents", data=intents)


def _samples_to_stacked(samples: list[ObjectSample]) -> dict[str, np.ndarray]:
    return {
        "points": np.stack([s.points for s in samples], axis=0),
        "normals": np.stack([s.normals for s in samples], axis=0),
        "labels": np.stack([s.labels for s in samples], axis=0),
        "human_priors": np.stack([s.human_priors for s in samples], axis=0),
        "force_centers": np.stack([s.force_center for s in samples], axis=0),
        "obj_ids": np.array([s.obj_id for s in samples], dtype="S32"),
        "categories": np.array([s.dataset for s in samples], dtype="S20"),
        "intents": np.array(["grasp"] * len(samples), dtype="S20"),
    }


def write_soft_h5_from_samples(
    path: str,
    samples: list[ObjectSample],
    *,
    heatmap_sigma_ratio: float,
    label_threshold: float,
    extra_meta: dict,
    source_h5: str | None = None,
) -> dict:
    if not samples:
        raise ValueError(f"no samples to write: {path}")
    stacked = _samples_to_stacked(samples)
    extra = {
        "normals": stacked["normals"],
        "human_priors": stacked["human_priors"],
        "force_centers": stacked["force_centers"],
        "categories": stacked["categories"],
        "intents": stacked["intents"],
    }
    return write_soft_h5(
        path,
        stacked["points"],
        stacked["labels"],
        stacked["obj_ids"],
        extra,
        heatmap_sigma_ratio=heatmap_sigma_ratio,
        label_threshold=label_threshold,
        source_h5=source_h5,
        src_meta=extra_meta,
        overwrite=True,
    )


def write_qc_summary(path: str, samples: list[ObjectSample]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "obj_id", "dataset", "n_grasps_trusted", "n_grasps_used", "n_grasps_dropped",
        "n_method_c", "n_method_b", "n_method_a",
        "n_contact_pts", "positive_ratio", "width_meas_mean", "width_err_mean",
        "force_center_x", "force_center_y", "force_center_z",
        "hp_scale_applied", "hp_nn_median_cm", "hp_positive_ratio",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in samples:
            w.writerow({
                "obj_id": s.obj_id,
                "dataset": s.dataset,
                "n_grasps_trusted": s.n_grasps_trusted,
                "n_grasps_used": s.n_grasps_used,
                "n_grasps_dropped": s.n_grasps_dropped,
                "n_method_c": s.n_method_c,
                "n_method_b": s.n_method_b,
                "n_method_a": s.n_method_a,
                "n_contact_pts": s.n_contact_pts,
                "positive_ratio": round(s.positive_ratio, 5),
                "width_meas_mean": round(s.width_meas_mean, 4),
                "width_err_mean": round(s.width_err_mean, 4),
                "force_center_x": round(float(s.force_center[0]), 5),
                "force_center_y": round(float(s.force_center[1]), 5),
                "force_center_z": round(float(s.force_center[2]), 5),
                "hp_scale_applied": int(s.hp_scale_applied),
                "hp_nn_median_cm": round(s.hp_nn_median_cm, 3)
                if np.isfinite(s.hp_nn_median_cm) else "",
                "hp_positive_ratio": round(s.hp_positive_ratio, 5),
            })


def qc_visualize(sample: ObjectSample, out_png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = sample.points
    pos = sample.labels > 0.5
    fig = plt.figure(figsize=(8, 7), facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")
    neg = pts[~pos]
    if len(neg):
        ax.scatter(neg[:, 0], neg[:, 1], neg[:, 2], c="#5ab4d4", s=2, alpha=0.35, linewidths=0)
    if np.any(pos):
        ax.scatter(
            pts[pos, 0], pts[pos, 1], pts[pos, 2],
            c="#2ca02c", s=8, alpha=0.9, linewidths=0, label="label=1",
        )
    method_colors = {"C": "#17becf", "B": "#ff7f0e", "A": "#e74c3c"}
    for c in sample.contact_rows[:40]:
        mc = "C"
        if "A" in c.method:
            mc = "A"
        elif "B" in c.method:
            mc = "B"
        col = method_colors.get(mc, "#aaa")
        ax.scatter(*c.L, c=col, s=40, marker="s", edgecolors="white", linewidths=0.3)
        ax.scatter(*c.R, c=col, s=40, marker="s", edgecolors="white", linewidths=0.3)
        ax.plot([c.L[0], c.R[0]], [c.L[1], c.R[1]], [c.L[2], c.R[2]], color=col, linewidth=1.0, alpha=0.7)
    ax.scatter(*sample.force_center, c="#ffdd57", s=60, marker="D", edgecolors="#666", linewidths=0.3)
    b = pts.min(0), pts.max(0)
    c = (b[0] + b[1]) / 2
    r = (b[1] - b[0]).max() * 0.55
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(b[0][2] - r * 0.1, b[0][2] + 2 * r)
    ax.set_title(
        f"{sample.obj_id} ({sample.dataset})  used={sample.n_grasps_used}  "
        f"C/B/A={sample.n_method_c}/{sample.n_method_b}/{sample.n_method_a}  "
        f"pos={sample.positive_ratio*100:.1f}%",
        color="#ddd",
        fontsize=9,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


def run_qc_checks(samples: list[ObjectSample]) -> list[str]:
    warnings: list[str] = []
    for s in samples:
        if s.hp_positive_ratio <= 0 and not s.hp_path:
            warnings.append(f"{s.obj_id}: human_prior missing (all zeros)")
        elif np.isfinite(s.hp_nn_median_cm) and s.hp_nn_median_cm > 2.5:
            warnings.append(
                f"{s.obj_id}: HP–mesh median nn={s.hp_nn_median_cm:.2f}cm (>2.5cm, check rotation/scale)"
            )
        if s.positive_ratio < 0.005:
            warnings.append(f"{s.obj_id}: positive_ratio={s.positive_ratio:.4f} (<0.5%)")
        if s.positive_ratio > 0.25:
            warnings.append(f"{s.obj_id}: positive_ratio={s.positive_ratio:.4f} (>25%)")
        if s.n_method_c == 0 and s.n_grasps_used > 0:
            warnings.append(f"{s.obj_id}: no method C contacts (all fallback)")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare affordance HDF5 (method C→B→A, executed pose)",
    )
    parser.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR)
    parser.add_argument("--outdir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--obj", help="只处理单个物体（调试）")
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--contact-radius", type=float, default=0.005)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--write-split",
        action="store_true",
        help="Also write affordance_train/val.h5 (+ soft); default is affordance_all*.h5 only",
    )
    parser.add_argument(
        "--heatmap-sigma-ratio",
        type=float,
        default=0.03,
        help="Gaussian σ ratio for soft labels (when soft export enabled)",
    )
    parser.add_argument(
        "--no-soft",
        action="store_true",
        help="Skip soft heatmap HDF5 (binary only)",
    )
    parser.add_argument(
        "--export-soft-only",
        action="store_true",
        help="Only export *_soft.h5 from existing affordance_*.h5 (no merged prepare)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing HDF5 when writing (export-soft-only or prepare)",
    )
    parser.add_argument(
        "--with-hp",
        action="store_true",
        default=True,
        help="KNN 映射 train_fp_rotated human_prior（默认开启）",
    )
    parser.add_argument(
        "--no-hp",
        action="store_false",
        dest="with_hp",
        help="不写 human_priors（全 0）",
    )
    parser.add_argument("--qc-vis", action="store_true", help="写 qc/vis PNG（含 oakink+dexycb）")
    parser.add_argument("--qc-vis-oakink", type=int, default=6)
    parser.add_argument("--qc-vis-dexycb", type=int, default=6)
    parser.add_argument("--qc-vis-max", type=int, default=16)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Per-object parallel workers (1=sequential). Try 4–8 on multi-core CPU.",
    )
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    if args.export_soft_only:
        export_soft_dataset_dir(
            outdir,
            heatmap_sigma_ratio=args.heatmap_sigma_ratio,
            label_threshold=0.5,
            overwrite=args.overwrite,
        )
        return

    merged_dir = os.path.abspath(args.merged_dir)
    qc_dir = os.path.join(outdir, "qc")
    vis_dir = os.path.join(qc_dir, "vis")
    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    if args.obj:
        jobs = [(args.obj, infer_dataset(args.obj), -1)]
    else:
        jobs = discover_trainable(merged_dir)
        print(f"Discovered {len(jobs)} trainable objects under {merged_dir}")

    with open(os.path.join(outdir, "objects_trainable.txt"), "w") as f:
        for obj_id, ds, n in jobs:
            if n < 0:
                p = os.path.join(merged_dir, f"{obj_id}_robot_gt_merged.hdf5")
                if not os.path.isfile(p):
                    p = os.path.join(merged_dir, f"{obj_id}_merged.hdf5")
                n = count_trusted_grasps(p)
            f.write(f"{obj_id},{ds},{n}\n")

    workers = max(1, int(args.workers))
    if args.obj:
        workers = 1
    samples, skipped = _run_prepare_jobs(
        jobs,
        merged_dir,
        num_points=args.num_points,
        contact_radius=args.contact_radius,
        with_hp=args.with_hp,
        workers=workers,
    )

    if not samples:
        print("❌ No samples produced.")
        for s in skipped[:20]:
            print(f"  {s}")
        sys.exit(1)

    if args.qc_vis:
        for s in pick_qc_vis_objects(
            samples,
            min_oakink=args.qc_vis_oakink,
            min_dexycb=args.qc_vis_dexycb,
            max_total=args.qc_vis_max,
        ):
            qc_visualize(s, os.path.join(vis_dir, f"{s.obj_id}.png"))
        n_oak = sum(1 for s in samples if s.dataset == "oakink")
        n_dex = sum(1 for s in samples if s.dataset == "dexycb")
        print(f"QC vis: {vis_dir} (dataset totals: oakink={n_oak} dexycb={n_dex})")

    write_qc_summary(os.path.join(qc_dir, "summary.csv"), samples)
    warnings = run_qc_checks(samples)
    if warnings:
        print("\n⚠️  QC warnings:")
        for w in warnings[:30]:
            print(f"  {w}")

    train_ids, val_ids = split_objects([s.obj_id for s in samples], args.train_ratio, args.split_seed)
    with open(os.path.join(outdir, "objects_train_val_split.json"), "w") as f:
        json.dump({"train": train_ids, "val": val_ids, "seed": args.split_seed}, f, indent=2)

    by_id = {s.obj_id: s for s in samples}
    all_samples = sorted(samples, key=lambda s: s.obj_id)
    train_samples = [by_id[o] for o in train_ids if o in by_id]
    val_samples = [by_id[o] for o in val_ids if o in by_id]

    meta = {
        "schema": "no_rot_executed_cba_v2",
        "contact_method": "C_width_match_adjacent_B_A_fallback",
        "tcp_offset": TCP_OFFSET,
        "width_mismatch_abs": WIDTH_MISMATCH_ABS,
        "width_mismatch_rel": WIDTH_MISMATCH_REL,
        "merged_dir": merged_dir,
        "with_hp": bool(args.with_hp),
        "n_objects": len(samples),
        "total_grasps_trusted": int(sum(s.n_grasps_trusted for s in samples)),
        "total_grasps_used": int(sum(s.n_grasps_used for s in samples)),
        "total_method_c": int(sum(s.n_method_c for s in samples)),
        "total_method_b": int(sum(s.n_method_b for s in samples)),
        "total_method_a": int(sum(s.n_method_a for s in samples)),
    }

    all_bin = os.path.join(outdir, "affordance_all.h5")
    all_soft = os.path.join(outdir, "affordance_all_soft.h5")
    write_affordance_h5(
        all_bin,
        all_samples,
        num_points=args.num_points,
        contact_radius=args.contact_radius,
        extra_meta=meta,
    )
    print(f"   {all_bin}  ({len(all_samples)} objects)")

    soft_exports: list[dict] = []
    if not args.no_soft:
        soft_exports.append(
            write_soft_h5_from_samples(
                all_soft,
                all_samples,
                heatmap_sigma_ratio=args.heatmap_sigma_ratio,
                label_threshold=0.5,
                extra_meta=meta,
                source_h5=all_bin,
            )
        )
        print(f"   {all_soft}")

    if args.write_split:
        train_bin = os.path.join(outdir, "affordance_train.h5")
        val_bin = os.path.join(outdir, "affordance_val.h5")
        write_affordance_h5(
            train_bin,
            train_samples,
            num_points=args.num_points,
            contact_radius=args.contact_radius,
            extra_meta=meta,
        )
        write_affordance_h5(
            val_bin,
            val_samples,
            num_points=args.num_points,
            contact_radius=args.contact_radius,
            extra_meta=meta,
        )
        print(f"   {train_bin}  ({len(train_samples)} objects)")
        print(f"   {val_bin}  ({len(val_samples)} objects)")
        if not args.no_soft:
            soft_exports.append(
                write_soft_h5_from_samples(
                    os.path.join(outdir, "affordance_train_soft.h5"),
                    train_samples,
                    heatmap_sigma_ratio=args.heatmap_sigma_ratio,
                    label_threshold=0.5,
                    extra_meta=meta,
                    source_h5=train_bin,
                )
            )
            soft_exports.append(
                write_soft_h5_from_samples(
                    os.path.join(outdir, "affordance_val_soft.h5"),
                    val_samples,
                    heatmap_sigma_ratio=args.heatmap_sigma_ratio,
                    label_threshold=0.5,
                    extra_meta=meta,
                    source_h5=val_bin,
                )
            )
            print(f"   {outdir}/affordance_train_soft.h5")
            print(f"   {outdir}/affordance_val_soft.h5")

    info = {
        **meta,
        "train_objects": len(train_samples),
        "val_objects": len(val_samples),
        "all_objects": len(all_samples),
        "num_points": args.num_points,
        "contact_radius": args.contact_radius,
        "train_ratio": args.train_ratio,
        "write_split": bool(args.write_split),
        "heatmap_sigma_ratio": args.heatmap_sigma_ratio,
        "with_soft": not args.no_soft,
        "outputs": {
            "affordance_all_h5": all_bin,
            "affordance_all_soft_h5": all_soft if not args.no_soft else None,
            "affordance_train_h5": os.path.join(outdir, "affordance_train.h5")
            if args.write_split
            else None,
            "affordance_val_h5": os.path.join(outdir, "affordance_val.h5")
            if args.write_split
            else None,
        },
        "soft_exports": soft_exports,
        "avg_positive_ratio": float(np.mean([s.positive_ratio for s in samples])),
        "generation_time_seconds": round(time.time() - t0, 1),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
    }
    with open(os.path.join(outdir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Done in {info['generation_time_seconds']}s")
    print(f"   objects: {len(samples)}  train={len(train_samples)}  val={len(val_samples)}")
    print(f"   methods C/B/A totals: {meta['total_method_c']}/{meta['total_method_b']}/{meta['total_method_a']}")
    if not args.write_split:
        print("   (train/val HDF5 skipped; use --write-split for 4 extra split files)")
    if skipped:
        print(f"   skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
