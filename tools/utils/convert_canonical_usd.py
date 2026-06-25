#!/usr/bin/env python3
"""
convert_canonical_usd.py — canonical PLY → USD (Isaac Sim 5.0)
===============================================================
将 data_hub/meshes/canonical/{dataset}/{obj}/mesh.ply
转换为 output/canonical_assets/{obj}.usd

必须用 sim5 运行:
    sim5 tools/convert_canonical_usd.py --dataset oakink --obj A01001
    sim5 tools/convert_canonical_usd.py --dataset oakink --all
"""

import os, sys, asyncio, argparse

# ── 路径初始化 ────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJ          = os.path.dirname(SCRIPT_DIR)
CANONICAL_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'canonical')
OUTPUT_DIR    = os.path.join(PROJ, 'output', 'canonical_assets')
DATASETS      = ['oakink', 'ycb', 'egodex']


async def convert_one(src_ply: str, dst_usd: str):
    import omni.kit.asset_converter as assetConverter
    from pxr import Usd, UsdGeom

    ctx = assetConverter.AssetConverterContext()
    ctx.ignore_animations     = True
    ctx.merge_all_meshes      = True
    ctx.use_meter_as_world_unit = True
    ctx.up_axis               = "Z"

    task = assetConverter.get_instance().create_converter_task(
        src_ply, dst_usd, None, ctx
    )
    success = await task.wait_until_finished()
    if not success:
        print(f"  ❌ conversion failed: {task.get_error_message()}")
        return False

    # 确保 upAxis=Z
    stage = Usd.Stage.Open(dst_usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.Save()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--obj',  help='只处理指定 obj_id')
    parser.add_argument('--all',  action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    # 收集要转换的文件
    datasets = DATASETS if args.all else ([args.dataset] if args.dataset else ['oakink'])
    tasks = []
    for ds in datasets:
        ds_dir = os.path.join(CANONICAL_DIR, ds)
        if not os.path.isdir(ds_dir): continue
        for obj_id in sorted(os.listdir(ds_dir)):
            if args.obj and args.obj not in obj_id: continue
            src = os.path.join(ds_dir, obj_id, 'mesh.ply')
            dst = os.path.join(OUTPUT_DIR, f'{obj_id}.usd')
            if not os.path.exists(src): continue
            if os.path.exists(dst) and not args.force:
                print(f"  ⏭  {obj_id}: already exists (use --force)")
                continue
            tasks.append((obj_id, src, dst))

    if not tasks:
        print("没有需要转换的文件。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n将转换 {len(tasks)} 个物体 → {OUTPUT_DIR}\n")

    # Isaac Sim 环境启动后运行
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    import carb

    ok, fail = 0, 0
    for obj_id, src, dst in tasks:
        print(f"  [{ok+fail+1}/{len(tasks)}] {obj_id} ...")
        try:
            result = asyncio.get_event_loop().run_until_complete(convert_one(src, dst))
            if result:
                ok += 1
                print(f"         ✅ → {dst}")
            else:
                fail += 1
        except Exception as e:
            print(f"         ❌ {e}")
            fail += 1

    app.close()
    print(f"\n完成: ✅{ok}  ❌{fail}")
    print(f"canonical USDs → {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
