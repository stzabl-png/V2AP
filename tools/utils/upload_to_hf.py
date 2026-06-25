#!/usr/bin/env python3
"""
upload_to_hf.py — 上传 ProcessedData 到 HuggingFace
=====================================================
目标 repo: UCBProject/ProcessedData

上传结构:
  third_depth/{dataset}/{seq}/        depths.npz, K.txt, frame_ids.txt
  third_mano/{dataset}/{seq}.npz      MANO verts
  obj_poses/{dataset}/{seq}/          ob_in_cam/*.txt, track_vis/
  obj_meshes/{dataset}/{obj}/         mesh.ply, scale.json, rotation.json
  obj_recon_input/{dataset}/          SAM2 masks
  training_fp/{dataset}/{obj}.hdf5    Phase 2 training data   [已在HF]
  human_prior_fp/{obj}.hdf5           Phase 3 inference prior [已在HF]

用法:
  python3 tools/upload_to_hf.py --category third_depth --dataset oakink
  python3 tools/upload_to_hf.py --category obj_meshes --dataset oakink
  python3 tools/upload_to_hf.py --category all
  python3 tools/upload_to_hf.py --category obj_poses --dataset dexycb --dry-run
"""

import os, sys, argparse, glob
from pathlib import Path

PROJ      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE      = os.path.join(PROJ, 'data_hub', 'ProcessedData')
HF_REPO   = 'UCBProject/ProcessedData'
HF_TYPE   = 'dataset'

# 各类别本地目录名 → HF路径名（保持一致）
CATEGORIES = {
    'third_depth':     'third_depth',
    'third_mano':      'third_mano',
    'obj_poses':       'obj_poses',
    'obj_meshes':      'obj_meshes',
    'obj_recon_input': 'obj_recon_input',
    # 以下已在HF，默认跳过
    # 'training_fp':   'training_fp',
    # 'human_prior_fp':'human_prior_fp',
}

# 每个类别默认上传的数据集
CATEGORY_DATASETS = {
    'third_depth':     ['oakink', 'dexycb', 'arctic', 'ho3d_v3', 'egodex'],
    'third_mano':      ['oakink', 'dexycb', 'arctic', 'ho3d_v3'],
    'obj_poses':       ['dexycb', 'arctic', 'ho3d_v3'],
    'obj_meshes':      ['oakink', 'ycb', 'arctic', 'ho3d_v3', 'dexycb', 'egocentric'],
    'obj_recon_input': ['oakink', 'ycb', 'arctic', 'egocentric'],
}


def get_local_files(category, dataset):
    """列出要上传的所有本地文件，返回 (local_path, hf_path) 列表."""
    local_dir = os.path.join(BASE, category, dataset)
    if not os.path.isdir(local_dir):
        print(f'  ⚠️  本地目录不存在: {local_dir}')
        return []

    files = []
    for root, dirs, fnames in os.walk(local_dir):
        for fname in fnames:
            local_path = os.path.join(root, fname)
            # HF 路径 = category/dataset/...
            rel = os.path.relpath(local_path, BASE)
            hf_path = rel  # 保持相对路径
            files.append((local_path, hf_path))
    return files


def upload_category(category, dataset, dry_run=False, token=None):
    """上传单个 category/dataset 下的所有文件."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)

    files = get_local_files(category, dataset)
    if not files:
        return 0

    print(f'\n📦 上传 {category}/{dataset}  ({len(files)} 文件)')

    if dry_run:
        for lp, hp in files[:5]:
            print(f'  [DRY] {hp}')
        if len(files) > 5:
            print(f'  ... 共 {len(files)} 文件')
        return len(files)

    # 用 upload_folder 批量上传（效率高）
    local_dir = os.path.join(BASE, category, dataset)
    hf_dir    = f'{category}/{dataset}'

    try:
        api.upload_folder(
            folder_path=local_dir,
            path_in_repo=hf_dir,
            repo_id=HF_REPO,
            repo_type=HF_TYPE,
            commit_message=f'add {category}/{dataset}',
            ignore_patterns=['__pycache__', '*.pyc', '.DS_Store'],
        )
        print(f'  ✅ {category}/{dataset} 上传完成')
        return len(files)
    except Exception as e:
        print(f'  ❌ 上传失败: {e}')
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='all',
                        help='上传类别: third_depth/third_mano/obj_poses/obj_meshes/obj_recon_input/all')
    parser.add_argument('--dataset',  default=None,
                        help='指定数据集: oakink/dexycb/arctic/ho3d_v3/ycb/egocentric')
    parser.add_argument('--dry-run',  action='store_true', help='只列出文件，不实际上传')
    parser.add_argument('--token',    default=os.environ.get('HF_TOKEN'),
                        help='HuggingFace token (或设置 HF_TOKEN 环境变量)')
    args = parser.parse_args()

    if not args.token and not args.dry_run:
        print('❌ 需要 HuggingFace token: --token <token> 或 export HF_TOKEN=<token>')
        sys.exit(1)

    # 确定要上传的 (category, dataset) 对
    todo = []
    cats = list(CATEGORIES.keys()) if args.category == 'all' else [args.category]
    for cat in cats:
        if cat not in CATEGORIES:
            print(f'❌ 未知类别: {cat}')
            continue
        datasets = [args.dataset] if args.dataset else CATEGORY_DATASETS.get(cat, [])
        for ds in datasets:
            todo.append((cat, ds))

    print(f'待上传: {len(todo)} 个 (category, dataset) 对')
    if args.dry_run:
        print('[DRY RUN 模式 — 不实际上传]\n')

    total_files = 0
    for cat, ds in todo:
        n = upload_category(cat, ds, dry_run=args.dry_run, token=args.token)
        total_files += n

    print(f'\n✅ 完成: 共 {total_files} 个文件')
    print(f'   HF: https://huggingface.co/datasets/{HF_REPO}')


if __name__ == '__main__':
    main()
