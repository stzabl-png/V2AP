# prepare_affordance_executed — 训练数据（executed + C/B/A + HP）

独立脚本：`tools/prepare_affordance_executed.py`（不修改 `build_dataset.py` / `gen_m5_training_data.py`）。

## 输入

- `output/grasp_collect_no_rot/merged/{obj}_robot_gt_merged.hdf5`（仅 `gripper_tips_trusted` + `executed_panda_hand_at_close`）
- `data_hub/meshes/SAM3DMesh/rotated_mesh/.../mesh.ply` × `scale.json`
- **Human prior（默认）**: `data_hub/ProcessedData/train_fp_rotated/{dataset}/{obj}.hdf5`

## Human prior 坐标系（旋转 + 尺度）

与 `random_grasp_sampler` / `vis_rotated_mesh_hp.py` 一致：

| 步骤 | 内容 |
|------|------|
| **旋转** | `train_fp_rotated` 由 `tools/rotate_training_fp.py` 对 `training_fp` 做 **Rx(+90°)**，与 `rotated_mesh` 同约定 |
| **尺度 OakInk** | 盘上 HP **未** metric → prepare 时 `point_cloud` × `scale.json` |
| **尺度 ycb_dex_*** | 盘上 HP **已是** metric → **不再**乘 scale（mesh 仍按 `scale.json` 乘） |
| **映射** | 对 mesh 表面 4096 采样点，KNN 取最近 HP 点的 `human_prior` 标量 |

对齐检查：`summary.csv` 含 `hp_nn_median_cm`（HP 点到 mesh 表面中位距离）；>2.5cm 会打 QC warning。

可视化核对：

```bash
python3 tools/vis_rotated_mesh_hp.py --obj A01001 --dataset oakink
python3 tools/vis_rotated_mesh_hp.py --obj ycb_dex_10 --dataset dexycb
```

## 接触点方法（C → B → A）

1. **C**：沿指长扫描；相邻表面交点对；选最接近 `finger_width_actual` 的站位  
2. **B**：C 与指宽差 > max(2cm, 35%) → raycast  
3. **A**：解析 fallback  
4. 物体级池化 → KDTree 5mm → `labels`

## 输出

`output/affordance_no_rot_executed/`

- `affordance_train.h5` / `affordance_val.h5`（`human_priors` 默认非零）
- `objects_trainable.txt`, `objects_train_val_split.json`, `dataset_info.json`
- `qc/summary.csv`, `qc/vis/*.png`（`--qc-vis`）

## 用法

```bash
conda activate bundlesdf
cd ~/Project/V2AP

python3 tools/prepare_affordance_executed.py
python3 tools/prepare_affordance_executed.py --workers 8   # 按物体多进程（默认 --workers 1）
python3 tools/prepare_affordance_executed.py --obj A01001 --qc-vis
python3 tools/prepare_affordance_executed.py --no-hp   # 关闭 HP
bash scripts/run_prepare_affordance_executed.sh
```

## 输出 HDF5（默认）

| 文件 | 说明 |
|------|------|
| `affordance_all.h5` | 全量二值（默认必写） |
| `affordance_all_soft.h5` | 全量 soft（默认必写，除非 `--no-soft`） |
| `objects_train_val_split.json` | train/val 物体列表（用于训练划分） |

加 **`--write-split`** 时再写：`affordance_train.h5`, `affordance_val.h5`, `affordance_train_soft.h5`, `affordance_val_soft.h5`（共 6 个 h5）。

仅从已有 binary 重算 soft（不跑 merged）：

```bash
python3 tools/prepare_affordance_executed.py \
  --export-soft-only \
  --dataset-dir output/affordance_no_rot_executed \
  --heatmap-sigma-ratio 0.03 \
  --overwrite
```

已有 train/val、要补全量且**不覆盖原文件**：

```bash
python3 tools/merge_affordance_h5_splits.py --dataset-dir output/affordance_no_rot_executed
```

## 训练

见 **[`docs/train_affordance.md`](train_affordance.md)**（`python -m model.train_v6`，仍用 train/val soft h5）

`human_priors` 在 HDF5 中有字段，v6 默认 **未** 用作监督或输入。
