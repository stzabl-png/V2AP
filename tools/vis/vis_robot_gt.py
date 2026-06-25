#!/usr/bin/env python3
"""
Robot GT 可视化: 显示成功抓取的夹爪位姿叠加在物体上.
用法:
    python3 tools/vis_robot_gt.py --obj A01001        # 单个物体
    python3 tools/vis_robot_gt.py --all                # 汇总统计
"""
import os, sys, glob, argparse
import numpy as np
import trimesh
import h5py

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'tools'))
from mesh_utils import load_mesh_canonical, find_ply

MESH_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'v1')          # legacy .obj
PROC_MESH_DIR = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')  # .ply
GT_DIRS = [
    os.path.join(PROJ, 'output', 'robot_gt_merged_oakink'),  # ★ 最新合并 (优先)
    os.path.join(PROJ, 'output', 'robot_gt_merged_dexycb'),  # ★ 最新合并 (优先)
    os.path.join(PROJ, 'output', 'robot_gt_r3'),
    os.path.join(PROJ, 'output', 'robot_gt_verified'),
    os.path.join(PROJ, 'output', 'robot_gt_v1_manual'),
    os.path.join(PROJ, 'output', 'robot_gt_v2_raycast'),
    os.path.join(PROJ, 'output', 'robot_gt'),
    os.path.join(PROJ, 'output', 'robot_gt_random'),
]


def find_mesh(obj_id):
    """先找 .obj，再找 ProcessedData .ply。"""
    p = os.path.join(MESH_DIR, f'{obj_id}.obj')
    if os.path.exists(p): return p
    for ds in ['oakink', 'dexycb', 'egodex']:
        p = os.path.join(PROC_MESH_DIR, ds, obj_id, 'mesh.ply')
        if os.path.exists(p): return p
    return None


def vis_single(obj_id):
    """可视化单个物体的 Robot GT: 用 canonical mesh (与 USD/Sim 一致) 做 raycast."""
    import open3d as o3d

    # 加载 canonical mesh（与 USD/Sim 完全对齐）
    _dataset = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
    _ply, _ds = find_ply(obj_id, _dataset)
    if _ply is not None:
        mesh = load_mesh_canonical(obj_id, _ds, verbose=True)
    else:
        legacy = os.path.join(MESH_DIR, f'{obj_id}.obj')
        if not os.path.exists(legacy):
            print(f'❌ mesh 不存在: {obj_id}'); return
        mesh = trimesh.load(legacy, force='mesh')


    grasp_data = []  # list of {mid, c1, c2, name}
    
    for gt_dir in GT_DIRS:
        gt_path = os.path.join(gt_dir, f'{obj_id}_robot_gt.hdf5')
        if not os.path.exists(gt_path): continue
        
        with h5py.File(gt_path, 'r') as f:
            if not f.attrs.get('success', False): continue
            if 'successful_grasps' not in f: continue
            
            src = os.path.basename(gt_dir)
            ns = f.attrs.get('n_successful', 0)
            print(f"📂 {src}: ✅ {ns} 成功")
            
            for key in f['successful_grasps'].keys():
                g = f[f'successful_grasps/{key}']
                name = g.attrs.get('name', key)
                gp = g['grasp_point'][:]  # TCP 位置
                ad = g['approach_dir'][:] if 'approach_dir' in g else None
                fd = g['finger_dir'][:] if 'finger_dir' in g else None
                w = g.attrs.get('gripper_width', 0.04)

                if ad is None or fd is None:
                    print(f"    ⚠️ {name}: 缺少方向数据, 跳过")
                    continue

                # 手指中点 = TCP + approach × 10.5cm
                finger_mid = gp + ad * 0.105
                
                # Raycast: 从中点沿 ±finger_dir 射出, 找表面交点
                c1, c2 = None, None
                
                # 方向1: +finger_dir
                locs1, _, _ = mesh.ray.intersects_location(
                    ray_origins=[finger_mid],
                    ray_directions=[fd]
                )
                if len(locs1) > 0:
                    dists = np.linalg.norm(locs1 - finger_mid, axis=1)
                    c1 = locs1[np.argmin(dists)]
                
                # 方向2: -finger_dir
                locs2, _, _ = mesh.ray.intersects_location(
                    ray_origins=[finger_mid],
                    ray_directions=[-fd]
                )
                if len(locs2) > 0:
                    dists = np.linalg.norm(locs2 - finger_mid, axis=1)
                    c2 = locs2[np.argmin(dists)]
                
                if c1 is not None and c2 is not None:
                    print(f"    🤖 {name}: w={w*100:.1f}cm ✅ 两个接触点")
                elif c1 is not None or c2 is not None:
                    print(f"    🤖 {name}: w={w*100:.1f}cm ⚠️ 只找到1个接触点")
                else:
                    print(f"    🤖 {name}: w={w*100:.1f}cm ❌ 无接触点")
                
                grasp_data.append({
                    'mid':          finger_mid,
                    'c1':           c1,
                    'c2':           c2,
                    'name':         name,
                    'approach_dir': ad,
                    'finger_dir':   fd,
                    'gripper_width': w,
                })
        
        if grasp_data:
            break  # 优先使用第一个有数据的 GT 源
    
    if not grasp_data:
        print(f"⚠️ {obj_id} 没有成功的 Robot GT"); return
    
    print(f"\n打开 Open3D ({len(grasp_data)} 个GT)...")

    geometries = []

    # 物体线框 (canonical 坐标系，与 USD/Sim 完全一致)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
    wireframe.paint_uniform_color([0.6, 0.6, 0.6])
    geometries.append(wireframe)

    def add_sphere(pos, r, color):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        s.translate(pos); s.paint_uniform_color(color); s.compute_vertex_normals()
        geometries.append(s)

    def add_line(p0, p1, color):
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector([p0, p1])
        ls.lines  = o3d.utility.Vector2iVector([[0, 1]])
        ls.colors = o3d.utility.Vector3dVector([color])
        geometries.append(ls)

    for gd in grasp_data:
        mid  = gd['mid']   # 指尖中点 (TCP + approach*0.105)
        ad   = gd.get('approach_dir')
        fd   = gd.get('finger_dir')
        w    = gd.get('gripper_width', 0.04)
        c1, c2 = gd['c1'], gd['c2']

        # 🟡 黄色球: 指尖中点 (force_center)
        add_sphere(mid, 0.004, [1.0, 0.85, 0.0])

        # 🔴🔵 左右接触点
        if c1 is not None: add_sphere(c1, 0.003, [0.9, 0.15, 0.15])
        if c2 is not None: add_sphere(c2, 0.003, [0.15, 0.3, 0.95])

        # ━ 夹爪横杆 (接触点之间)
        if c1 is not None and c2 is not None:
            add_line(c1, c2, [0.2, 0.85, 0.2])

        # ↑ approach 方向箭头 (绿色, 从 TCP 指向物体)
        if ad is not None:
            tcp = mid - ad * 0.105          # 手腕位置
            add_sphere(tcp, 0.003, [0.9, 0.6, 0.1])  # 🟠 手腕
            add_line(tcp, mid, [0.1, 0.9, 0.3])       # 绿色 approach 线

        # ← finger_dir 横向 (青色, 表示夹爪方向)
        if fd is not None:
            tl = mid - fd * w / 2
            tr = mid + fd * w / 2
            add_line(tl, tr, [0.2, 0.85, 0.85])

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Robot GT — {obj_id} ({len(grasp_data)} grasps)  "
                    f"🟡指尖中心  🔴🔵接触点  🟢approach  🟠手腕  青=夹爪宽"
    )


def summary_all():
    """统计所有物体的 Robot GT 状态（从 merged 目录读）."""
    all_obj_ids = set()
    for d in GT_DIRS:
        for fp in glob.glob(os.path.join(d, '*_robot_gt.hdf5')):
            all_obj_ids.add(os.path.basename(fp).replace('_robot_gt.hdf5', ''))
    all_obj_ids = sorted(all_obj_ids)
    results = {'success': [], 'fail': []}

    for obj_id in all_obj_ids:
        best_status = 'fail'
        for gt_dir in GT_DIRS:
            gt_path = os.path.join(gt_dir, f'{obj_id}_robot_gt.hdf5')
            if os.path.exists(gt_path):
                try:
                    with h5py.File(gt_path, 'r') as hf:
                        if int(hf.attrs.get('n_successful', 0)) > 0:
                            best_status = 'success'
                            break
                except:
                    pass
        results[best_status].append(obj_id)

    total = len(all_obj_ids)
    print('=' * 60)
    print('  Robot GT 汇总 (merged 目录)')
    print('=' * 60)
    print(f'  ✅ 成功: {len(results["success"])}/{total}')
    print(f'  ❌ 全轮失败: {len(results["fail"])}/{total}')
    print()

    if results['success']:
        print('✅ 成功的物体:')
        for i, oid in enumerate(results['success']):
            print(f'  {oid}', end='')
            if (i + 1) % 8 == 0: print()
        print()

    if results['fail']:
        print('\n❌ 全轮失败 ({len(results["fail"])}):')
        for oid in results['fail']:
            print(f'  {oid}')

def main():
    parser = argparse.ArgumentParser(description='Robot GT 可视化')
    parser.add_argument('--obj',    help='可视化单个物体')
    parser.add_argument('--all',    action='store_true', help='汇总所有统计')
    parser.add_argument('--gt_dir', help='指定 robot_gt 目录 (默认搜索所有标准目录)')
    args = parser.parse_args()

    if args.gt_dir:
        # 临时覆盖全局 GT_DIRS
        global GT_DIRS
        GT_DIRS = [os.path.abspath(args.gt_dir)]

    if args.all:
        summary_all()
    elif args.obj:
        vis_single(args.obj)
    else:
        print("用法: --all 查看汇总, --obj ID 可视化单个, --gt_dir 指定目录")


if __name__ == '__main__':
    main()
