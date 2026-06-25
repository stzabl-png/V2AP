#!/usr/bin/env python3
"""
mesh_utils.py — 统一 mesh 加载工具
=====================================
所有脚本调用同一入口，确保 PLY mesh 的朝向与 USD/Sim 完全一致。

坐标系基准 (Isaac Sim 世界坐标系):
  +Z : 垂直桌面向上
  +Y : 机械臂前进方向
  +X : 横向

canonical rotation 来源: data_hub/ProcessedData/obj_meshes/{dataset}/{obj_id}/rotation.json
  → 由 estimate_obj_rotation.py 生成
  → convert_obj_usd.py 已正确将其 bake 进 USD 顶点
  → 本文件让 PLY 也应用同样旋转，使两者完全对齐
"""
import os, json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

# ── 路径 ─────────────────────────────────────────────────────────────────────
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_MESH_DIR = os.path.join(_PROJ, "data_hub", "ProcessedData", "obj_meshes")
DATASETS = ["oakink", "dexycb", "arctic"]


def get_scale_factor(obj_id: str, dataset: str) -> float:
    """读取 scale.json，无则返回 1.0。"""
    path = os.path.join(PROC_MESH_DIR, dataset, obj_id, "scale.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return float(d.get("scale_factor", 1.0))
    return 1.0


def get_canonical_euler(obj_id: str, dataset: str = "oakink") -> list:
    """
    返回 canonical rotation (euler_xyz_deg)。
    来源：obj_meshes/{dataset}/{obj_id}/rotation.json
    无文件或旋转≈0 时返回 [0.0, 0.0, 0.0]。
    """
    rot_path = os.path.join(PROC_MESH_DIR, dataset, obj_id, "rotation.json")
    if os.path.exists(rot_path):
        with open(rot_path) as f:
            data = json.load(f)
        euler = data.get("euler_xyz_deg", [0.0, 0.0, 0.0])
        if any(abs(e) > 0.5 for e in euler):
            return [float(e) for e in euler]
    return [0.0, 0.0, 0.0]


def get_canonical_matrix(obj_id: str, dataset: str = "oakink") -> np.ndarray:
    """返回 3×3 旋转矩阵，无旋转时为 I。"""
    euler = get_canonical_euler(obj_id, dataset)
    if any(abs(e) > 0.5 for e in euler):
        return Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
    return np.eye(3, dtype=np.float64)


def infer_dataset(obj_id: str, dataset: str | None = None) -> str:
    """根据 obj_id / 显式参数推断 obj_meshes 子目录名。"""
    if dataset:
        return dataset
    if obj_id.startswith("ycb_"):
        return "dexycb"
    return "oakink"


def identity_mesh_prerotation_record(obj_id: str, dataset: str | None = None) -> dict:
    """未应用 mesh 预旋转（no_rotation / 单位阵）。"""
    ds = infer_dataset(obj_id, dataset)
    return {
        "dataset": ds,
        "euler_xyz_deg": [0.0, 0.0, 0.0],
        "matrix": np.eye(3, dtype=np.float64),
        "method": "identity",
        "rotation_json_path": "",
    }


def applied_mesh_prerotation_record(
    obj_id: str, dataset: str | None = None, *, no_rotation: bool = False,
) -> dict:
    """
    本次生成/测试 pose 时实际 bake 到 mesh 上的预旋转（与 sampler / USD 一致）。
    no_rotation=True → 单位旋转；否则与 get_canonical_euler 相同逻辑。
    """
    if no_rotation:
        return identity_mesh_prerotation_record(obj_id, dataset)
    ds = infer_dataset(obj_id, dataset)
    euler = get_canonical_euler(obj_id, ds)
    matrix = get_canonical_matrix(obj_id, ds)
    method = ""
    rot_path = os.path.join(PROC_MESH_DIR, ds, obj_id, "rotation.json")
    if os.path.exists(rot_path):
        with open(rot_path) as f:
            method = str(json.load(f).get("method", ""))
    if not any(abs(e) > 0.5 for e in euler):
        return identity_mesh_prerotation_record(obj_id, ds)
    return {
        "dataset": ds,
        "euler_xyz_deg": euler,
        "matrix": matrix.astype(np.float64),
        "method": method or "rotation.json",
        "rotation_json_path": rot_path,
    }


def load_rotation_json_record(obj_id: str, dataset: str | None = None) -> dict:
    """
    读取 rotation.json 中的欧拉角与 3×3 矩阵（磁盘定义，不表示 pipeline 是否应用）。
    无文件时 euler=[0,0,0]，matrix=I。
    """
    ds = infer_dataset(obj_id, dataset)
    rot_path = os.path.join(PROC_MESH_DIR, ds, obj_id, "rotation.json")
    euler = [0.0, 0.0, 0.0]
    method = ""
    json_path = ""
    if os.path.exists(rot_path):
        json_path = rot_path
        with open(rot_path) as f:
            data = json.load(f)
        euler = [float(e) for e in data.get("euler_xyz_deg", [0.0, 0.0, 0.0])]
        method = str(data.get("method", ""))
    matrix = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
    return {
        "dataset": ds,
        "euler_xyz_deg": euler,
        "matrix": matrix.astype(np.float64),
        "method": method,
        "rotation_json_path": json_path,
    }


def stable_orientations_path(obj_id: str, dataset: str) -> str:
    return os.path.join(PROC_MESH_DIR, dataset, obj_id, "stable_orientations.json")


def load_stable_orientations(obj_id: str, dataset: str | None = None) -> dict | None:
    """读取 stable_orientations.json；不存在返回 None。"""
    ds = infer_dataset(obj_id, dataset)
    path = stable_orientations_path(obj_id, ds)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    doc.setdefault("dataset", ds)
    doc.setdefault("obj", obj_id)
    return doc


def placement_seed(dataset: str, obj_id: str, round_idx: int) -> int:
    """Deterministic seed for per-object placement in a batch round."""
    import hashlib

    key = f"{dataset}:{obj_id}:round_{round_idx:04d}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def sample_placement_id(doc: dict, seed: int) -> int:
    """Uniform random over orientations[] (includes id=0 identity)."""
    orientations = doc.get("orientations") or []
    if not orientations:
        return 0
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(orientations)))
    return int(orientations[idx]["id"])


def get_orientation_entry(doc: dict, placement_id: int) -> dict | None:
    for o in doc.get("orientations") or []:
        if int(o["id"]) == int(placement_id):
            return o
    return None


def placement_record(
    R: np.ndarray,
    *,
    obj_id: str,
    dataset: str,
    placement_id: int = 0,
    method: str = "identity",
    probability: float | None = None,
    source: str = "stable_orientations",
) -> dict:
    """Record for HDF5 mesh_prerotation/ (compatible with legacy readers)."""
    R = np.asarray(R, dtype=np.float64)
    euler = [float(x) for x in Rotation.from_matrix(R).as_euler("xyz", degrees=True)]
    return {
        "dataset": dataset,
        "euler_xyz_deg": euler,
        "matrix": R,
        "placement_id": int(placement_id),
        "method": str(method),
        "probability": probability,
        "source": str(source),
        "rotation_json_path": "",
    }


def resolve_placement(
    obj_id: str,
    dataset: str | None = None,
    *,
    placement_id: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Resolve R_placement (v' = R @ v on identity-scaled mesh).

    No stable_orientations.json → identity fallback.
    placement_id set → use that entry; else sample with seed.
    """
    ds = infer_dataset(obj_id, dataset)
    doc = load_stable_orientations(obj_id, ds)

    if doc is None:
        rec = placement_record(
            np.eye(3, dtype=np.float64),
            obj_id=obj_id,
            dataset=ds,
            placement_id=0,
            method="fallback_identity",
            probability=None,
            source="fallback_identity",
        )
        return rec["matrix"], rec

    if placement_id is not None:
        entry = get_orientation_entry(doc, placement_id)
        if entry is None:
            raise ValueError(
                f"placement_id={placement_id} not in stable_orientations for {obj_id} ({ds})"
            )
    else:
        if seed is None:
            seed = 0
        placement_id = sample_placement_id(doc, seed)
        entry = get_orientation_entry(doc, placement_id)
        if entry is None:
            raise ValueError(f"sampled placement_id={placement_id} missing for {obj_id}")

    R = np.asarray(entry["matrix"], dtype=np.float64)
    rec = placement_record(
        R,
        obj_id=obj_id,
        dataset=ds,
        placement_id=int(entry["id"]),
        method=str(entry.get("method", "trimesh_stable_pose")),
        probability=entry.get("probability"),
        source="stable_orientations",
    )
    return R, rec


def rotate_vertices(mesh: trimesh.Trimesh, R: np.ndarray) -> None:
    R = np.asarray(R, dtype=np.float64)
    mesh.vertices = (R @ mesh.vertices.T).T


def rotate_points(points: np.ndarray | None, R: np.ndarray) -> np.ndarray | None:
    if points is None:
        return None
    R = np.asarray(R, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        return (R @ pts).astype(np.float32)
    return (R @ pts.T).T.astype(np.float32)


def list_ready_objects(dataset: str) -> list[str]:
    """Objects with mesh.ply + scale.json (same readiness as stable orientations estimate)."""
    ds_dir = os.path.join(PROC_MESH_DIR, dataset)
    if not os.path.isdir(ds_dir):
        return []
    return sorted(
        o for o in os.listdir(ds_dir)
        if os.path.isfile(os.path.join(ds_dir, o, "mesh.ply"))
        and os.path.isfile(os.path.join(ds_dir, o, "scale.json"))
        and "_" not in o.split("_")[0]
    )


def write_mesh_prerotation_hdf5(parent, record: dict) -> None:
    """在 pose group（或文件根）下写入 mesh_prerotation/（euler + matrix）。"""
    import h5py

    if "mesh_prerotation" in parent:
        del parent["mesh_prerotation"]
    if "mesh_placement" in parent:
        del parent["mesh_placement"]
    g = parent.create_group("mesh_prerotation")
    g.create_dataset(
        "euler_xyz_deg",
        data=np.asarray(record["euler_xyz_deg"], dtype=np.float64),
    )
    g.create_dataset("matrix", data=np.asarray(record["matrix"], dtype=np.float64))
    g.attrs["dataset"] = str(record.get("dataset", ""))
    if record.get("method"):
        g.attrs["method"] = str(record["method"])
    if record.get("rotation_json_path"):
        g.attrs["rotation_json_path"] = str(record["rotation_json_path"])
    if "placement_id" in record:
        g.attrs["placement_id"] = int(record["placement_id"])
    if record.get("source"):
        g.attrs["source"] = str(record["source"])
    if record.get("probability") is not None:
        g.attrs["probability"] = float(record["probability"])


def read_mesh_prerotation_hdf5(parent) -> dict | None:
    """从 HDF5 group 读取 mesh_prerotation/ 或 mesh_placement/；不存在则返回 None。"""
    gname = None
    if "mesh_prerotation" in parent:
        gname = "mesh_prerotation"
    elif "mesh_placement" in parent:
        gname = "mesh_placement"
    if gname is None:
        return None
    g = parent[gname]
    rec = {
        "dataset": str(g.attrs.get("dataset", "")),
        "euler_xyz_deg": [float(x) for x in g["euler_xyz_deg"][:]],
        "matrix": np.array(g["matrix"][:], dtype=np.float64),
        "method": str(g.attrs.get("method", "")),
        "rotation_json_path": str(g.attrs.get("rotation_json_path", "")),
    }
    if "placement_id" in g.attrs:
        rec["placement_id"] = int(g.attrs["placement_id"])
    if "source" in g.attrs:
        rec["source"] = str(g.attrs["source"])
    if "probability" in g.attrs:
        rec["probability"] = float(g.attrs["probability"])
    return rec


def read_mesh_prerotation_hdf5_pose(pose_group, file_group=None) -> dict | None:
    """优先读 pose 级 mesh_prerotation/，否则回退到文件根（兼容旧格式）。"""
    rec = read_mesh_prerotation_hdf5(pose_group)
    if rec is not None:
        return rec
    if file_group is not None:
        return read_mesh_prerotation_hdf5(file_group)
    return None


def resolve_mesh_prerotation_record(
    obj_id: str,
    *,
    dataset: str | None = None,
    hdf5_path: str | None = None,
    no_rotation: bool = True,
) -> dict:
    """优先从已有 HDF5 读 mesh_prerotation，否则按 no_rotation 推断实际应用的旋转。"""
    if hdf5_path and os.path.isfile(hdf5_path):
        import h5py

        with h5py.File(hdf5_path, "r") as f:
            rec = read_mesh_prerotation_hdf5(f)
            if rec is not None:
                return rec
            meta = f.get("metadata")
            if meta is not None and "no_rotation" in meta.attrs:
                no_rotation = bool(meta.attrs["no_rotation"])
    return applied_mesh_prerotation_record(obj_id, dataset, no_rotation=no_rotation)


def find_ply(obj_id: str, dataset: str = None) -> tuple:
    """
    查找 mesh.ply 文件路径。
    返回 (ply_path, dataset_found) 或 (None, None)。
    """
    search = [dataset] if dataset else DATASETS
    for ds in search:
        p = os.path.join(PROC_MESH_DIR, ds, obj_id, "mesh.ply")
        if os.path.exists(p):
            return p, ds
    return None, None


def load_mesh_canonical(
    obj_id: str,
    dataset: str = None,
    apply_scale: bool = True,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """
    加载 PLY mesh 并应用 scale + canonical rotation，
    使其朝向与 Isaac Sim 中的 USD 完全一致。

    Args:
        obj_id:       物体 ID
        dataset:      'oakink' / 'dexycb' / 'arctic'，None 则自动搜索
        apply_scale:  是否应用 scale.json 转换为米制
        verbose:      是否打印旋转信息

    Returns:
        trimesh.Trimesh，已旋转到 canonical 朝向（与 Sim/USD 一致）
    """
    ply_path, ds = find_ply(obj_id, dataset)
    if ply_path is None:
        raise FileNotFoundError(f"mesh.ply not found for {obj_id} (searched: {dataset or DATASETS})")

    mesh = trimesh.load(ply_path, force="mesh")

    # ── 1. 应用 scale ──────────────────────────────────────────────────────
    if apply_scale:
        scale = get_scale_factor(obj_id, ds)
        if abs(scale - 1.0) > 1e-6:
            mesh.vertices = mesh.vertices * scale

    # ── 2. 应用 canonical rotation (使朝向与 USD/Sim 对齐) ─────────────────
    euler = get_canonical_euler(obj_id, ds)
    if any(abs(e) > 0.5 for e in euler):
        R_mat = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
        mesh.vertices = (R_mat @ mesh.vertices.T).T
        if not mesh.is_watertight:
            trimesh.repair.fix_normals(mesh)
        if verbose:
            print(f"  [mesh_utils] {obj_id}: canonical rot {[round(e,1) for e in euler]}°")
    else:
        if verbose:
            print(f"  [mesh_utils] {obj_id}: no rotation (identity)")

    return mesh


def load_mesh_raw(
    obj_id: str,
    dataset: str = None,
    apply_scale: bool = True,
) -> trimesh.Trimesh:
    """
    加载原始 PLY（不应用 canonical rotation）。
    仅在需要与旧数据对比时使用。
    """
    ply_path, ds = find_ply(obj_id, dataset)
    if ply_path is None:
        raise FileNotFoundError(f"mesh.ply not found for {obj_id}")
    mesh = trimesh.load(ply_path, force="mesh")
    if apply_scale:
        scale = get_scale_factor(obj_id, ds)
        if abs(scale - 1.0) > 1e-6:
            mesh.vertices = mesh.vertices * scale
    return mesh
