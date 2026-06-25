#!/usr/bin/env python3
"""
批量将 OakInk (OBJ) 和 DexYCB/YCB (PLY) 的 mesh 转换为 Isaac Sim USD 格式

来源:
  - OakInkObjectsV2:      ~/Project/OakInk/shape/OakInkObjectsV2/*/align/textured_simple.obj  (101个)
  - OakInkVirtualObjectsV2: ~/Project/OakInk/shape/OakInkVirtualObjectsV2/*/align/design.obj  (1700个)
  - DexYCB meshes:        data_hub/ProcessedData/obj_meshes/dexycb/*/mesh.ply  (3个)
  - YCB meshes:           data_hub/ProcessedData/obj_meshes/ycb/*/mesh.ply     (28个)

输出:
  - assets/usd/oakink_real/{obj_name}.usd
  - assets/usd/oakink_virtual/{obj_name}.usd
  - assets/usd/dexycb/{session_name}.usd
  - assets/usd/ycb/{obj_name}.usd

使用方式 (Isaac Sim 环境):
    sim45 assets/convert_oakink_dexycb_usd.py
    sim45 assets/convert_oakink_dexycb_usd.py --source oakink_real
    sim45 assets/convert_oakink_dexycb_usd.py --source dexycb
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os
import sys
import asyncio
import argparse
import tempfile
import trimesh
import omni.kit.asset_converter as converter

# ── 路径配置 ────────────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
PROJECT = os.path.join(HOME, "Project", "V2AP")

OAKINK_REAL_DIR    = os.path.join(HOME, "Project", "OakInk", "shape", "OakInkObjectsV2")
OAKINK_VIRTUAL_DIR = os.path.join(HOME, "Project", "OakInk", "shape", "OakInkVirtualObjectsV2")
DEXYCB_MESH_DIR    = os.path.join(PROJECT, "data_hub", "ProcessedData", "obj_meshes", "dexycb")
YCB_MESH_DIR       = os.path.join(PROJECT, "data_hub", "ProcessedData", "obj_meshes", "ycb")

USD_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd")

# ── Isaac Sim 转换核心 ──────────────────────────────────────────────────────
async def convert_one(input_path: str, output_path: str) -> bool:
    task_manager = converter.get_instance()
    context = converter.AssetConverterContext()
    context.ignore_materials = False
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True
    context.single_mesh = True
    context.smooth_normals = True
    context.export_preview_surface = False
    context.use_meter_as_world_unit = True
    context.embed_textures = True

    task = task_manager.create_converter_task(
        input_path, output_path, progress_callback=None, asset_converter_context=context
    )
    success = await task.wait_until_finished()
    if not success:
        print(f"    ❌ {task.get_status()} - {task.get_detailed_error()}")
    return success


def ply_to_obj_tmp(ply_path: str) -> str:
    """用 trimesh 把 PLY 转成临时 OBJ，返回临时文件路径"""
    mesh = trimesh.load(ply_path, force='mesh')
    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    tmp.close()
    mesh.export(tmp.name)
    return tmp.name


def run_convert(input_path: str, output_path: str) -> bool:
    return asyncio.get_event_loop().run_until_complete(convert_one(input_path, output_path))


# ── 各数据集批量入口 ────────────────────────────────────────────────────────
def collect_oakink_real():
    """OakInkObjectsV2: align/textured_simple.obj"""
    items = []
    if not os.path.isdir(OAKINK_REAL_DIR):
        print(f"[WARN] OakInkObjectsV2 not found: {OAKINK_REAL_DIR}")
        return items
    for obj_name in sorted(os.listdir(OAKINK_REAL_DIR)):
        obj_path = os.path.join(OAKINK_REAL_DIR, obj_name, "align", "textured_simple.obj")
        if os.path.isfile(obj_path):
            items.append((obj_name, obj_path, "obj"))
    return items


def collect_oakink_virtual():
    """OakInkVirtualObjectsV2: align/design.obj"""
    items = []
    if not os.path.isdir(OAKINK_VIRTUAL_DIR):
        print(f"[WARN] OakInkVirtualObjectsV2 not found: {OAKINK_VIRTUAL_DIR}")
        return items
    for obj_name in sorted(os.listdir(OAKINK_VIRTUAL_DIR)):
        obj_path = os.path.join(OAKINK_VIRTUAL_DIR, obj_name, "align", "design.obj")
        if os.path.isfile(obj_path):
            items.append((obj_name, obj_path, "obj"))
    return items


def collect_dexycb():
    """dexycb: session_name/mesh.ply"""
    items = []
    if not os.path.isdir(DEXYCB_MESH_DIR):
        print(f"[WARN] DexYCB mesh dir not found: {DEXYCB_MESH_DIR}")
        return items
    for session in sorted(os.listdir(DEXYCB_MESH_DIR)):
        ply_path = os.path.join(DEXYCB_MESH_DIR, session, "mesh.ply")
        if os.path.isfile(ply_path):
            items.append((session, ply_path, "ply"))
    return items


def collect_ycb():
    """ycb: obj_name/mesh.ply"""
    items = []
    if not os.path.isdir(YCB_MESH_DIR):
        print(f"[WARN] YCB mesh dir not found: {YCB_MESH_DIR}")
        return items
    for obj_name in sorted(os.listdir(YCB_MESH_DIR)):
        ply_path = os.path.join(YCB_MESH_DIR, obj_name, "mesh.ply")
        if os.path.isfile(ply_path):
            items.append((obj_name, ply_path, "ply"))
    return items


SOURCE_MAP = {
    "oakink_real":    (collect_oakink_real,    "oakink_real"),
    "oakink_virtual": (collect_oakink_virtual, "oakink_virtual"),
    "dexycb":         (collect_dexycb,         "dexycb"),
    "ycb":            (collect_ycb,            "ycb"),
}


def process_source(source_key: str):
    collect_fn, usd_subdir = SOURCE_MAP[source_key]
    items = collect_fn()
    usd_dir = os.path.join(USD_BASE_DIR, usd_subdir)
    os.makedirs(usd_dir, exist_ok=True)

    total = len(items)
    print(f"\n{'='*60}")
    print(f"[{source_key}] {total} meshes → {usd_dir}")
    print(f"{'='*60}")

    success_cnt, skip_cnt, fail_list = 0, 0, []

    for i, (name, src_path, fmt) in enumerate(items, 1):
        usd_path = os.path.join(usd_dir, f"{name}.usd")
        prefix = f"[{i:4d}/{total}] {name}"

        if os.path.exists(usd_path):
            print(f"{prefix}: skip ✓")
            skip_cnt += 1
            continue

        # PLY → 临时 OBJ
        tmp_obj = None
        if fmt == "ply":
            print(f"{prefix}: PLY→OBJ ...", end=" ", flush=True)
            try:
                tmp_obj = ply_to_obj_tmp(src_path)
                convert_input = tmp_obj
                print("ok, converting...", end=" ", flush=True)
            except Exception as e:
                print(f"❌ PLY load error: {e}")
                fail_list.append(name)
                continue
        else:
            convert_input = src_path
            print(f"{prefix}: converting...", end=" ", flush=True)

        ok = run_convert(convert_input, usd_path)

        # 清理临时文件
        if tmp_obj and os.path.exists(tmp_obj):
            os.unlink(tmp_obj)

        if ok:
            print("✅")
            success_cnt += 1
        else:
            fail_list.append(name)

    print(f"\n[{source_key}] Done: {success_cnt} converted, {skip_cnt} skipped, {len(fail_list)} failed")
    if fail_list:
        print(f"  Failed items: {fail_list[:20]}{'...' if len(fail_list) > 20 else ''}")

    return success_cnt, skip_cnt, fail_list


def main():
    parser = argparse.ArgumentParser(description="批量转换 OakInk/DexYCB mesh → USD")
    parser.add_argument(
        "--source",
        choices=list(SOURCE_MAP.keys()) + ["all"],
        default="all",
        help="要转换的数据集 (默认: all)"
    )
    args = parser.parse_args()

    sources = list(SOURCE_MAP.keys()) if args.source == "all" else [args.source]

    total_ok, total_skip, total_fail = 0, 0, []
    for src in sources:
        ok, skip, fail = process_source(src)
        total_ok += ok
        total_skip += skip
        total_fail += fail

    print(f"\n{'='*60}")
    print(f"全部完成: {total_ok} 转换成功, {total_skip} 跳过, {len(total_fail)} 失败")
    if total_fail:
        print(f"失败列表: {total_fail}")
    print(f"USD 输出根目录: {USD_BASE_DIR}/")


main()
simulation_app.close()
