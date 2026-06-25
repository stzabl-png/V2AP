#!/usr/bin/env python3
"""
convert_obj_usd.py
==================
将物体 mesh 转为 Isaac Sim USD，输出:
    output/obj_usd/{dataset}/{obj_id}.usd

默认几何来源（与 random_grasp_sampler 一致）:
    data_hub/meshes/SAM3DMesh/rotated_mesh/{dataset|ycb}/{obj}/mesh.ply
    × scale.json（仍从 data_hub/ProcessedData/obj_meshes/ 读取）

rotated_mesh 已含 +90°X canonical 旋转，导出时不再叠 rotation.json（请用 --no-rotation）。

用法:
    python3 tools/convert_obj_usd.py --obj A01001 --dataset oakink --no-rotation --force
    python3 tools/convert_obj_usd.py --dataset oakink --no-rotation --force
    python3 tools/convert_obj_usd.py --dataset ycb --no-rotation --force

    # 回退旧 obj_meshes 源:
    python3 tools/convert_obj_usd.py --obj A01001 --legacy-assets --no-rotation --force
"""
import os, sys, json, argparse
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJ_MESHES_DIR = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
SAM3D_ROTATED_MESH_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'SAM3DMesh', 'rotated_mesh')
OBJ_USD_DIR = os.path.join(PROJ, 'output', 'obj_usd')
CANONICAL_ROT_JSON = os.path.join(PROJ, 'sim', 'canonical_rotation.json')

DATASETS = ['oakink', 'ycb', 'arctic', 'dexycb', 'egocentric', 'ho3d_v3']
# 当前 rotated_mesh + scale.json 就绪的数据集
ROTATED_MESH_DATASETS = ['oakink', 'ycb']


def infer_obj_dataset(obj_id: str, dataset: str | None = None) -> str:
    if dataset:
        return dataset
    if obj_id.startswith('ycb_dex_'):
        return 'dexycb'
    if obj_id.startswith('arctic_'):
        return 'arctic'
    return 'oakink'


def mesh_scale_json_path(obj_id: str, dataset: str | None = None) -> str | None:
    ds = infer_obj_dataset(obj_id, dataset)
    if obj_id.startswith('ycb_dex_'):
        path = os.path.join(OBJ_MESHES_DIR, 'ycb', obj_id, 'scale.json')
    else:
        path = os.path.join(OBJ_MESHES_DIR, ds, obj_id, 'scale.json')
    return path if os.path.isfile(path) else None


def read_scale_factor(obj_id: str, dataset: str | None = None) -> float:
    path = mesh_scale_json_path(obj_id, dataset)
    if path is None:
        return 1.0
    with open(path) as f:
        return float(json.load(f).get('scale_factor', 1.0))


def find_mesh_for_convert(obj_id: str, dataset: str | None = None, *, use_legacy_assets: bool = False):
    """
    查找转换用 mesh + scale。

    默认: SAM3DMesh/rotated_mesh + obj_meshes/scale.json
    legacy: obj_meshes/mesh.ply

    Returns:
        mesh_path, scale_factor, usd_dataset (USD 输出子目录名)
    """
    ds = infer_obj_dataset(obj_id, dataset)

    if use_legacy_assets:
        search_ds = [dataset] if dataset else DATASETS
        for sub in search_ds:
            mesh_path = os.path.join(OBJ_MESHES_DIR, sub, obj_id, 'mesh.ply')
            if os.path.isfile(mesh_path):
                return mesh_path, read_scale_factor(obj_id, sub), sub
        return None, read_scale_factor(obj_id, ds), None

    if obj_id.startswith('ycb_dex_') or ds in ('ycb', 'dexycb'):
        mesh_path = os.path.join(SAM3D_ROTATED_MESH_DIR, 'ycb', obj_id, 'mesh.ply')
        usd_ds = 'ycb'
    else:
        mesh_path = os.path.join(SAM3D_ROTATED_MESH_DIR, ds, obj_id, 'mesh.ply')
        usd_ds = ds

    if not os.path.isfile(mesh_path):
        return None, read_scale_factor(obj_id, ds), None
    return mesh_path, read_scale_factor(obj_id, ds), usd_ds


def load_obj_rotation(obj_id, dataset):
    """Legacy: per-object rotation.json（仅 --legacy-assets 且未 --no-rotation 时使用）。"""
    rot_path = os.path.join(OBJ_MESHES_DIR, dataset, obj_id, 'rotation.json')
    if os.path.exists(rot_path):
        data = json.load(open(rot_path))
        euler = data.get('euler_xyz_deg', None)
        if euler is not None:
            return euler, f'rotation.json ({data.get("method", "?")})'
    return None, 'none'


def list_dataset_objs(dataset: str, *, use_legacy_assets: bool = False):
    """列出可转换物体（mesh + scale.json）。"""
    if use_legacy_assets:
        ds_dir = os.path.join(OBJ_MESHES_DIR, dataset)
        if not os.path.isdir(ds_dir):
            return []
        return sorted(
            o for o in os.listdir(ds_dir)
            if os.path.isfile(os.path.join(ds_dir, o, 'mesh.ply'))
            and mesh_scale_json_path(o, dataset)
        )

    out: list[str] = []
    if dataset in ('ycb', 'dexycb'):
        sam_dir = os.path.join(SAM3D_ROTATED_MESH_DIR, 'ycb')
        if not os.path.isdir(sam_dir):
            return []
        for obj_id in sorted(os.listdir(sam_dir)):
            if not os.path.isfile(os.path.join(sam_dir, obj_id, 'mesh.ply')):
                continue
            if not obj_id.startswith('ycb_dex_'):
                continue
            if mesh_scale_json_path(obj_id, 'dexycb'):
                out.append(obj_id)
        return out

    sam_dir = os.path.join(SAM3D_ROTATED_MESH_DIR, dataset)
    if not os.path.isdir(sam_dir):
        return []
    for obj_id in sorted(os.listdir(sam_dir)):
        if not os.path.isfile(os.path.join(sam_dir, obj_id, 'mesh.ply')):
            continue
        if mesh_scale_json_path(obj_id, dataset):
            out.append(obj_id)
    return out


def convert_one(
    obj_id,
    mesh_path,
    scale_factor,
    dataset,
    force=False,
    no_rotation=False,
    *,
    use_legacy_assets: bool = False,
):
    """将单个物体 PLY → USD."""
    out_dir = os.path.join(OBJ_USD_DIR, dataset)
    out_path = os.path.join(out_dir, f'{obj_id}.usd')

    if os.path.exists(out_path) and not force:
        print(f'  ⏭️  已存在: {out_path}')
        return True

    os.makedirs(out_dir, exist_ok=True)

    try:
        import trimesh
        from pxr import Usd, UsdGeom, Vt, Gf
    except ImportError as e:
        print(f'  ❌ 缺少依赖: {e}')
        return False

    mesh = trimesh.load(mesh_path, force='mesh')

    if abs(scale_factor - 1.0) > 1e-6:
        mesh.vertices = mesh.vertices * scale_factor

    mesh_source = (
        'obj_meshes' if use_legacy_assets
        else 'SAM3DMesh/rotated_mesh'
    )

    if no_rotation or not use_legacy_assets:
        rot_euler, rot_source = None, 'skipped (--no-rotation; +90° in rotated_mesh)' if not use_legacy_assets else 'skipped (--no-rotation)'
        print(f'     [rotation: identity  source={rot_source}]')
    else:
        rot_euler, rot_source = load_obj_rotation(obj_id, dataset)
        if rot_euler is not None and any(abs(e) > 0.5 for e in rot_euler):
            from scipy.spatial.transform import Rotation as _R
            R_mat = _R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
            mesh.vertices = (R_mat @ mesh.vertices.T).T
            print(f'     [rotation {rot_euler}  source={rot_source}]')
        else:
            print(f'     [rotation: identity  source={rot_source}]')

    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata('metersPerUnit', 1.0)

    root_xform = UsdGeom.Xform.Define(stage, '/Root')
    stage.SetDefaultPrim(root_xform.GetPrim())

    mesh_prim = UsdGeom.Mesh.Define(stage, '/Root/Mesh')

    verts = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)

    mesh_prim.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
    )
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))

    if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
        normals = mesh.face_normals.astype(np.float32)
        mesh_prim.GetNormalsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in normals])
        )
        mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.uniform)

    stage.Save()

    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    z_offset = float(-vmin[2]) if vmin[2] < 0 else 0.0

    _tools = os.path.dirname(os.path.abspath(__file__))
    if _tools not in sys.path:
        sys.path.insert(0, _tools)
    from mesh_utils import applied_mesh_prerotation_record

    prerot_no_rot = no_rotation or not use_legacy_assets
    _pr = applied_mesh_prerotation_record(obj_id, dataset, no_rotation=prerot_no_rot)
    meta = {
        'obj': obj_id,
        'dataset': dataset,
        'mesh_source': mesh_source,
        'scale_factor': scale_factor,
        'z_offset_m': round(z_offset, 6),
        'bbox_min': [round(float(v), 6) for v in vmin],
        'bbox_max': [round(float(v), 6) for v in vmax],
        'bbox_extent_cm': [round(float(v) * 100, 2) for v in (vmax - vmin)],
        'no_rotation': bool(prerot_no_rot),
        'mesh_prerotation_euler_xyz_deg': _pr['euler_xyz_deg'],
        'mesh_prerotation_matrix': [list(row) for row in _pr['matrix'].tolist()],
        'mesh_prerotation_method': _pr.get('method', ''),
    }
    meta_path = os.path.join(out_dir, f'{obj_id}_meta.json')
    with open(meta_path, 'w') as mf:
        json.dump(meta, mf, indent=2)

    ext = vmax - vmin
    print(f'  ✅  {out_path}')
    print(f'      mesh: {os.path.relpath(mesh_path, PROJ)}')
    print(f'      尺寸 (cm): {ext[0]*100:.1f}×{ext[1]*100:.1f}×{ext[2]*100:.1f}   '
          f'scale={scale_factor:.6f}   z_offset={z_offset*100:.1f}cm')
    return True


def main():
    parser = argparse.ArgumentParser(
        description='rotated SAM3D mesh (默认) 或 obj_meshes → USD',
    )
    parser.add_argument('--obj', help='单个物体 ID')
    parser.add_argument('--dataset', help='oakink / ycb / …')
    parser.add_argument('--all', action='store_true', help='转换 ROTATED_MESH_DATASETS 全部')
    parser.add_argument('--force', action='store_true', help='覆盖已有 USD')
    parser.add_argument(
        '--legacy-assets',
        action='store_true',
        help='使用 obj_meshes/mesh.ply（旧路径）',
    )
    parser.add_argument(
        '--no-rotation',
        action='store_true',
        help='不应用 rotation.json（rotated_mesh 默认已 canonical，务必开启）',
    )
    args = parser.parse_args()

    use_legacy = args.legacy_assets
    if not use_legacy and not args.no_rotation:
        print('⚠️  默认使用 rotated_mesh（已含 +90°X）；自动启用 --no-rotation')
        args.no_rotation = True

    src = 'obj_meshes' if use_legacy else SAM3D_ROTATED_MESH_DIR
    print(f'Mesh 来源: {src}')
    print(f'no_rotation: {args.no_rotation}')

    todo = []

    if args.obj:
        mesh_path, sf, usd_ds = find_mesh_for_convert(
            args.obj, dataset=args.dataset, use_legacy_assets=use_legacy,
        )
        if mesh_path is None:
            print(f'❌ 未找到 mesh: {args.obj}  (dataset={args.dataset})')
            return
        todo.append((args.obj, mesh_path, sf, usd_ds))

    elif args.dataset or args.all:
        target_ds = ROTATED_MESH_DATASETS if args.all else [args.dataset]
        for ds in target_ds:
            objs = list_dataset_objs(ds, use_legacy_assets=use_legacy)
            print(f'  {ds}: {len(objs)} 个物体')
            for obj_id in objs:
                mp, sf, usd_ds = find_mesh_for_convert(
                    obj_id, dataset=ds, use_legacy_assets=use_legacy,
                )
                if mp:
                    todo.append((obj_id, mp, sf, usd_ds))
        print(f'待转换: {len(todo)} 个物体 (数据集: {target_ds})')

    else:
        parser.print_help()
        return

    ok = err = 0
    for i, (obj_id, mesh_path, sf, usd_ds) in enumerate(todo):
        print(f'\n[{i+1}/{len(todo)}] {obj_id}  (usd_dataset={usd_ds})')
        if convert_one(
            obj_id, mesh_path, sf, usd_ds,
            force=args.force,
            no_rotation=args.no_rotation,
            use_legacy_assets=use_legacy,
        ):
            ok += 1
        else:
            err += 1

    print(f'\n{"="*50}')
    print(f'  完成: {ok} 成功  {err} 失败')
    print(f'  输出: {OBJ_USD_DIR}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
