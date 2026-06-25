# OakInk 抓取采集流水线使用说明

从 **USD 转换** → **候选生成** → **Isaac Sim 验证** → **可视化 / 全量 batch**。

**两条 batch 路线：**

| 路线 | 脚本 | 说明 |
|------|------|------|
| **Pool（当前推荐）** | `batch_gen_candidates_pool.py` + `batch_sim_candidates_pool.py` | 先建 candidate 池，再 sim（默认 weighted 抽样；可选 equal / 成功上限）；长驻 Isaac worker、chunk 续跑 |
| **Legacy** | `batch_grasp_collect.py` | 每轮对所有物体重新 sampler + 每物体一个 `run_grasp_sim` |

Pool 路线详见 **第 5 节**；Legacy 见 **第 4 节**。  
**Pool sim 部署、HF 下载/上传、续跑：** [`pool_grasp_sim_pipeline.md`](pool_grasp_sim_pipeline.md)（唯一入口文档）。  
**已有 candidate 的模块化 eval（批量 / z-yaw / 录像）：** [`evaluation.md`](evaluation.md)（`evaluation/eval_single.py`、`eval_batch.py`）。

## 输出目录约定

| 目录 | 用途 |
|------|------|
| `output/grasp_collect_no_rot/` | **当前** 修改版 legacy batch 默认（rotated SAM3D + `train_fp_rotated` sampler） |
| `output/grasp_collect_legacy/` | 旧一轮实验（原 `grasp_collect` 改名保留） |
| `output/grasp_collect/` | 旧实验目录（placement 流水线已移除） |

下文示例路径以 **`grasp_collect_no_rot`** 为准；续跑时 `--outdir` 必须与已有 `state.json` 一致。

默认策略：**不旋转 mesh**（`--no-rotation` USD；采样 mesh 用 `rotated_mesh` + `train_fp_rotated` HP）。

---

## 0. 环境准备（每次新开终端）

**路径因机器而异**，先把下面两个变量改成你自己的：

| 变量 | 含义 | 示例 |
|------|------|------|
| `PROJ` | 本仓库根目录 | `~/Project/V2AP` |
| `ISAAC_SIM_PATH` | Isaac Sim 安装根目录（含 `python.sh`） | 见下方说明 |

**如何找到 Isaac Sim 路径：** 安装目录下应有 `python.sh`，例如：

- `~/isaacsim`
- `~/.local/share/ov/pkg/isaac-sim-4.x.x`
- NVIDIA 默认安装路径（按你本机文档为准）

```bash
# 任选一种方式确认
ls "$ISAAC_SIM_PATH/python.sh"
# 或: find ~ -name 'python.sh' -path '*/isaac*' 2>/dev/null | head
```

```bash
conda activate bundlesdf   # 或你用于 sampler/convert 的环境名

export PROJ=/path/to/V2AP
cd "$PROJ"

export ISAAC_SIM_PATH=/path/to/your/isaac-sim   # ← 改成你的 Isaac 根目录
```

| 阶段 | Python | 说明 |
|------|--------|------|
| USD 转换 / sampler / batch 主进程 | `bundlesdf` 里的 `python3` | 需要 `trimesh`, `h5py`, `scipy`, `rtree`；USD 需要 `usd-core`（`pxr`） |
| Isaac Sim | `$ISAAC_SIM_PATH/python.sh` | batch 会自动调用；也可传 `--isaac-python /path/to/python.sh` |

安装缺失依赖（若报错）：

```bash
pip install rtree usd-core
```

---

## 1. USD 转换（全量或单个）

输出目录：`output/obj_usd/{dataset}/{OBJ}.usd` 与 `{OBJ}_meta.json`。

**默认 mesh 源**（与 legacy sampler 一致）:

- `data_hub/meshes/SAM3DMesh/rotated_mesh/{oakink|ycb}/{obj}/mesh.ply`（已含 +90°X）
- `scale.json` 仍从 `data_hub/ProcessedData/obj_meshes/` 读取

请始终加 **`--no-rotation`**（脚本在 rotated_mesh 模式下会自动启用）。**不要**再叠 `rotation.json`。

### 1.1 OakInk + YCB 全量重转（换 rotated mesh 后必做）

```bash
conda activate bundlesdf
cd "$PROJ"

python3 tools/convert_obj_usd.py --dataset oakink --no-rotation --force
python3 tools/convert_obj_usd.py --dataset ycb --no-rotation --force
```

### 1.2 单个物体 smoke test

```bash
python3 tools/convert_obj_usd.py --obj A01001 --dataset oakink --no-rotation --force
python3 tools/convert_obj_usd.py --obj ycb_dex_01 --dataset ycb --no-rotation --force
```

### 1.3 回退 obj_meshes（旧资产）

```bash
python3 tools/convert_obj_usd.py --obj A01001 --legacy-assets --no-rotation --force
```

### 1.4 检查是否生成成功

```bash
ls -lh output/obj_usd/oakink/A01001.usd output/obj_usd/oakink/A01001_meta.json
```

---

## 2. 单物体：候选生成 + Sim + 可视化

### 2.1 生成 20 个候选（sampler）

```bash
conda activate bundlesdf
cd "$PROJ"

python3 tools/random_grasp_sampler.py \
  --obj A01001 \
  --dataset oakink \
  --no-rotation \
  --force \
  --target 20 \
  --output-dir output/grasp_collect_no_rot/candidates/round_0000
```

输出：`output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5`

### 2.2 Isaac Sim 验证（单物体）

```bash
# 确保已 export ISAAC_SIM_PATH（见第 0 节）

$ISAAC_SIM_PATH/python.sh sim/run_grasp_sim.py \
  --hdf5 output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5 \
  --result-dir output/grasp_collect_no_rot/robot_gt/round_0000 \
  --save-result \
  --headless \
  --max-candidates 20
```

输出：`output/grasp_collect_no_rot/robot_gt/round_0000/A01001_robot_gt.hdf5`

### 2.2.1 `robot_gt` HDF5 里有什么（schema v2）

整物体一个 `{OBJ}_robot_gt.hdf5`（不是每个 try 一个文件）。

| 位置 | 内容 |
|------|------|
| **根 attrs** | `obj_id`, `n_successful`, `robot_gt_schema_version=2`, `executed_pose_frame=object_mesh`, `executed_ee_frame=panda_hand` |
| **`candidate_results/candidate_i`** | **每个 try 都有**：候选 `grasp_point`/`rotation`、`success`、（跑完闭合+提起后）`executed_panda_hand_at_close` / `executed_panda_hand_post_lift` |
| **`successful_grasps/grasp_j`** | 仅 **Sim 成功** 的 try：同上 + `approach_dir`/`finger_dir`（来自候选旋转） |
| **`winning_candidate`** | 第一个成功者（候选 + executed，无 `contact_points_local`） |

**语义（必读）：**

- **`grasp_point` / `rotation`**：Stage A **规划候选**（grasp 帧），不是 Sim 执行真值。
- **`executed_panda_hand_at_close`**：闭合结束、**lift 前**，真实 **`panda_hand` 手腕** 在物体系下的位姿（`position`, `rotation`, `approach_dir`, `finger_dir`）。
- **`executed_panda_hand_post_lift`**：提起稳定后、与 **Δz>3cm 成功判定** 同刻的 `panda_hand` 位姿。
- **`gripper_tips_loc`** `(2,3)`：闭合后、lift 前（与 `executed_panda_hand_at_close` 同时刻）左右指尖在 **物体 mesh 局部系**；`finger_width_actual` 为两点距离。凡执行到 close 的候选都会写入 `candidate_results`（不限 success）。

**录屏：** batch **不**录屏。调试时用 `sim/run_grasp_sim_rec.py`（每 try 一个 mp4，与 headless sim 结果可能不一致）。

### 2.3 可视化（matplotlib PNG）

全部候选 + 成功标绿：

```bash
conda activate bundlesdf
cd "$PROJ"

python3 tools/vis_grasp_candidates.py \
  --hdf5 output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5 \
  --robot-gt output/grasp_collect_no_rot/robot_gt/round_0000/A01001_robot_gt.hdf5
```

只画成功的：

```bash
python3 tools/vis_grasp_candidates.py \
  --hdf5 output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5 \
  --robot-gt output/grasp_collect_no_rot/robot_gt/round_0000/A01001_robot_gt.hdf5 \
  --only-success
```

PNG 目录：`output/grasp_vis/A01001/`（成功-only 在 `success/` 子目录）。

### 2.4 查看成功数量与名称

```bash
python3 -c "
import h5py
p='output/grasp_collect_no_rot/robot_gt/round_0000/A01001_robot_gt.hdf5'
with h5py.File(p) as f:
    print('n_successful:', f.attrs.get('n_successful'))
    for k in sorted(f['successful_grasps'].keys()):
        print(' ', f['successful_grasps/'+k].attrs.get('name'))
"
```

---

## 3. 显存压测：每张 GPU 最多几个并行 sim

在正式 batch 前，用 A01001 测 `--sim-per-gpu`。压测会**真实启动多个 Isaac sim**（默认每个只验 1 个候选），因此需要一份**候选 HDF5** 作为输入。

### 3.0 前置：候选 HDF5 从哪来？

默认路径：

```text
output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5
```

**检查是否已有：**

```bash
ls -lh output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5
```

**若没有**（压测脚本报 `HDF5 不存在`），先跑 sampler 生成（只需几分钟，与第 2.1 节相同）：

```bash
conda activate bundlesdf
cd "$PROJ"

mkdir -p output/grasp_collect_no_rot/candidates/round_0000

python3 tools/random_grasp_sampler.py \
  --obj A01001 \
  --dataset oakink \
  --no-rotation \
  --force \
  --target 20 \
  --output-dir output/grasp_collect_no_rot/candidates/round_0000
```

生成后再确认：

```bash
ls -lh output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5
```

也可用其它物体 / 路径，压测时显式指定：

```bash
python3 scripts/benchmark_sim_parallel.py \
  --obj A01010 \
  --hdf5 output/grasp_collect_no_rot/candidates/round_0000/A01010_grasp.hdf5 \
  ...
```

> 压测**不需要** `robot_gt`（sim 成功结果），只要 **`*_grasp.hdf5`（候选）**。  
> 若还没有 USD，可先跑第 1 节，或确保 `output/obj_usd/oakink/A01001.usd` 存在，否则 sim 子进程会失败、显存读数不准。

### 3.1 运行压测

```bash
conda activate bundlesdf
cd "$PROJ"
# 确保已 export ISAAC_SIM_PATH（见第 0 节）

python3 scripts/benchmark_sim_parallel.py \
  --gpu 0 \
  --start-n 6 \
  --min-free-gb 3
```

### 3.2 双卡各测一次

```bash
python3 scripts/benchmark_sim_parallel.py --gpu 0 --start-n 6 --min-free-gb 3
python3 scripts/benchmark_sim_parallel.py --gpu 1 --start-n 6 --min-free-gb 3
```

### 3.3 看结果

```bash
cat output/benchmark_sim_parallel/gpu0_A01001/result_min_free_3.0gb.json | grep max_sim_per_gpu
```

记下推荐值，例如 `max_sim_per_gpu: 1` → batch 用 `--sim-per-gpu 1`。

---

## 4. Legacy 全量 batch（`batch_grasp_collect.py`）

脚本：`scripts/batch_grasp_collect.py`

**每轮两阶段：**

1. 并行生成**全部**物体的 candidate（`--sampler-workers`，**仅 raycast**，不传 `--structured-contacts`）
2. 并行 sim（`--sim-per-gpu` × `--sim-gpu-ids`，**`sim/run_grasp_sim.py --headless`**，不录屏）

**物体列表：** 与 `random_grasp_sampler.list_dataset_objs` 相同（**不是** `obj_meshes` 目录扫描）：

| `--dataset` | 条件 |
|-------------|------|
| `oakink` | `rotated_mesh/oakink/{id}/mesh.ply` + `train_fp_rotated/oakink/{id}.hdf5` + `scale.json` |
| `ycb` | `rotated_mesh/ycb/ycb_dex_*/mesh.ply` + `train_fp_rotated/dexycb/ycb_dex_*.hdf5` + scale |

**每轮可同时跑多个 dataset**（默认 `oakink+ycb` 共 120 物）：`--dataset oakink,ycb` 或 `--dataset all`。Sampler 按物传各自 `--dataset`；HDF5 文件名仅含 `obj_id`（OakInk / YCB id 不冲突）。全量前对两个 dataset 都做第 1 节 USD 转换。

输出根目录：`output/grasp_collect_no_rot/`（`batch_grasp_collect.py` 默认 `--outdir`）

| 路径 | 内容 |
|------|------|
| `candidates/round_XXXX/{OBJ}_grasp.hdf5` | 候选抓取（raycast） |
| `robot_gt/round_XXXX/{OBJ}_robot_gt.hdf5` | Sim 结果（schema v2，含 `executed_*`；可选 `sim_z_yaw_*` attrs） |
| `merged/{OBJ}_robot_gt_merged.hdf5` | 多轮 **successful_grasps** 合并（**默认不去重**；含 `executed_*`） |
| `summary.csv` | 每物体每轮状态（追加，不删历史） |
| `state.json` | 下次 `--resume` 从哪一轮开始 |
| `sim_logs/round_XXXX/` | convert / gen / sim 日志 |

**轮次与覆盖：** 不同 `round_XXXX/` **互不覆盖**。危险操作：不带 `--resume` 又从 `round_0000` 跑——sampler `--force` 会**重写**该轮已有 HDF5。

### 4.1 首次全量（默认 10 轮，已转 USD，双卡示例）

```bash
conda activate bundlesdf
cd "$PROJ"
# 确保已 export ISAAC_SIM_PATH

# OakInk + YCB 同一 batch（默认即两者；每轮 round_XXXX 下 100+20 个物体）
python3 scripts/batch_grasp_collect.py \
  --dataset all \
  --sampler-workers 8 \
  --sim-gpu-ids 0,1 \
  --sim-per-gpu 1 \
  --target 20 \
  --headless \
  --no-convert

# 仅 OakInk: --dataset oakink
# 仅 YCB:     --dataset ycb
```

- 默认 `--max-rounds 10` → `round_0000` … `round_0009`（可不写该参数）。
- 不加 `--rotation` = **no-rotation**。
- `--no-convert`：跳过 USD（需已完成第 1 节）。
- **成功判据**由 sim 固定为物体抬高 Δz>3cm，batch 不修改。

### 4.2 跑完预设轮数后继续加轮（不覆盖旧 round）

**轮次怎么算：** `state.json` 里的 `"round": N` 表示下一轮要写的目录是 `round_{N:04d}`。  
`--resume` + `--max-rounds K` 会跑 `round_N` … `round_{N+K-1}`（共 **K** 轮）。

| 已完成 | `state.json` | 命令 | 将写入的目录 |
|--------|--------------|------|----------------|
| `round_0000`…`0009` | `"round": 10` | `--resume --max-rounds 10` | `0010`…`0019` |
| `round_0000`…`0019` | `"round": 20` | `--resume --max-rounds 5` | `0020`…`0024` |

续跑前确认（`--outdir` 必须与历史一致）：

```bash
cat output/grasp_collect_no_rot/state.json
# 期望例如: "round": 10  → 下一批从 round_0010 开始
```

**示例 A** — 已跑满默认 10 轮，再加 **5** 轮（双卡、每卡 1 sim）：

```bash
python3 scripts/batch_grasp_collect.py \
  --dataset oakink \
  --max-rounds 5 \
  --resume \
  --sampler-workers 8 \
  --sim-gpu-ids 0,1 \
  --sim-per-gpu 1 \
  --headless \
  --no-convert
```

**示例 B** — 从 **round 10** 起再跑 **10** 轮（`round_0010`…`0019`），16 路 CPU 采样、双卡各 5 路 Isaac（最多 **10** 个 sim 并行）：

```bash
conda activate bundlesdf
export PROJ=/path/to/V2AP
export ISAAC_SIM_PATH=/path/to/your/isaac-sim
cd "$PROJ"

python3 scripts/batch_grasp_collect.py \
  --dataset oakink,ycb \
  --resume \
  --max-rounds 10 \
  --sampler-workers 16 \
  --sim-gpu-ids 0,1 \
  --sim-per-gpu 5 \
  --target 20 \
  --headless \
  --no-convert
```

并行参数说明：

| 参数 | 含义 |
|------|------|
| `--sampler-workers 16` | Phase 1：最多 16 个进程并行跑 `random_grasp_sampler`（CPU） |
| `--sim-gpu-ids 0,1` | Phase 2：使用物理 GPU 0 与 1 |
| `--sim-per-gpu 5` | 每张 GPU 上同时最多 5 个 `run_grasp_sim` |
| 合计 | 最多 **2 × 5 = 10** 个 Isaac 进程；显存紧张时不要加大 `--sim-per-gpu` |

`--resume` 作用：

1. 从 `state.json` 的 `round` 继续编号（不回到 0000）。
2. 某一 round 目录里已有 `*_grasp.hdf5` / `*_robot_gt.hdf5` 的物体会跳过 gen/sim。

**不要**在已有 10 轮数据后去掉 `--resume` 再跑默认 10 轮——会从 `round_0000` 重来并覆盖该轮文件。

另开一套输出（与旧实验完全隔离）：

```bash
python3 scripts/batch_grasp_collect.py \
  --dataset oakink \
  --outdir output/grasp_collect_exp2 \
  ...
```

### 4.3 断点续跑（同一轮未完成）

batch 中途断了，同一轮里补跑未完成物体：

```bash
python3 scripts/batch_grasp_collect.py \
  --dataset oakink \
  --max-rounds 1 \
  --resume \
  --sampler-workers 8 \
  --sim-gpu-ids 0,1 \
  --sim-per-gpu 1 \
  --headless \
  --no-convert
```

若 `state.json` 已进位到下一轮，但你想**只补** `round_0003`，需手动把 `state.json` 里 `"round"` 改回 `3`，或只删该轮缺失物体的 HDF5 后 `--resume`。

### 4.4 单卡

```bash
python3 scripts/batch_grasp_collect.py \
  --dataset oakink \
  --sampler-workers 8 \
  --sim-gpu-ids 0 \
  --sim-per-gpu 1 \
  --headless \
  --no-convert
```

### 4.5 Sim 内随机绕竖直轴（可选，`--random-z-yaw`）

**默认关闭**（不加 flag = 与改动前完全一致）：物体按原 `OBJECT_ORIENTATION` / override / meta 放置，不额外绕竖直轴转。

**开启后**（仅影响 Phase 2 Isaac，不改 Phase 1）：

| 项目 | 行为 |
|------|------|
| Candidate HDF5 | **不改**；仍为 sampler 在 canonical mesh 系下的 `position` / `rotation` |
| Sim 放置 | 在桌上对物体绕**竖直轴**加 yaw `θ`（世界 Z） |
| 执行 | 候选仍在 mesh 系；`execute_grasp` 用含 θ 的 `T_world_obj` 映到世界系 |
| 多 candidate | 每个 `(round, object)` **一次** `run_grasp_sim` 抽一个 θ，该物体本轮 ~20 个 candidate **共用** |
| 落盘 GT | `executed_panda_hand_*`、`gripper_tips` 经 `world_pose_to_object_mesh` 写回 **object_mesh** 系（与未开 augment 时语义相同） |
| 可复现 | batch 传 `--round-idx` + 可选 `--z-yaw-seed`；`θ = hash(seed, obj_id, round)` |

`robot_gt` 根 attrs（便于排查）：

- `sim_z_yaw_enabled`（bool）
- `sim_z_yaw_deg`（float；关闭时为 `0`）
- `sim_z_yaw_round_idx`、`sim_z_yaw_seed`（若提供）

与 `--resume`：已有 `*_robot_gt.hdf5` 的物体会 **skip sim**，不会用新 θ 重跑；要重采需删该轮 gt 或换新 `--outdir`。

```bash
# 在 4.2 的续跑命令上追加即可，例如:
python3 scripts/batch_grasp_collect.py \
  --dataset oakink,ycb \
  --resume --max-rounds 10 \
  --random-z-yaw --z-yaw-seed 42 \
  --sampler-workers 16 --sim-gpu-ids 0,1 --sim-per-gpu 5 \
  --headless --no-convert
```

单物体调试：

```bash
$ISAAC_SIM_PATH/python.sh sim/run_grasp_sim.py \
  --hdf5 output/grasp_collect_no_rot/candidates/round_0010/A01001_grasp.hdf5 \
  --result-dir output/grasp_collect_no_rot/robot_gt/round_0010 \
  --random-z-yaw --round-idx 10 --z-yaw-seed 42 --headless
```

### 4.6 监控进度

```bash
tail -f output/grasp_collect_no_rot/summary.csv
```

```bash
# 统计已有 robot_gt 数量
ls output/grasp_collect_no_rot/robot_gt/round_0000/*_robot_gt.hdf5 2>/dev/null | wc -l
```

建议用 `tmux` / `screen` 长时间挂后台。

### 4.6 单独重跑 merge（不跑 batch）

batch 每个物体 sim 结束后会自动 merge；也可手动合并已有各轮 `robot_gt`：

```bash
conda activate bundlesdf
cd "$PROJ"

python3 tools/merge_robot_gt.py --obj A01001 \
  --inputs \
    output/grasp_collect_no_rot/robot_gt/round_0000/A01001_robot_gt.hdf5 \
    output/grasp_collect_no_rot/robot_gt/round_0001/A01001_robot_gt.hdf5 \
  --output output/grasp_collect_no_rot/merged/A01001_robot_gt_merged.hdf5
```

默认 **保留全部** 成功条目（不去重），并拷贝每条成功的 **`executed_panda_hand_at_close` / `post_lift`**。需要 dedup 时加 `--deduplicate`（按候选 `grasp_point` 判近，不是 executed）。

**指尖 / 接触点来源（merged schema v3，必读）：**

| 字段 | 含义 | 训练可用？ |
|------|------|------------|
| `gripper_tips_loc` | 新 Sim：`at_close` 真指尖（物体系） | ✅ `gripper_tips_trusted=True` |
| `contact_points_local` | 旧轮：lift 后伪接触点 | ❌ `contact_points_trusted=False` |

每条 `successful_grasps/grasp_*` 有 **`gripper_tips_source`**：`at_close` / `legacy_post_lift` / `none`。**不会**再把旧 `contact_points_local` 改名成 `gripper_tips_loc`。训练读 merged 时请只看 `gripper_tips_trusted`（`build_dataset.py` 已按此过滤）。若 merged 里不要任何 legacy 指尖，merge 时加 **`--exclude-legacy-contact`**（只丢接触点，抓取 pose 仍保留）。

整库重 merge 示例（bash）：

```bash
for f in output/grasp_collect_no_rot/robot_gt/round_0000/*_robot_gt.hdf5; do
  obj=$(basename "$f" _robot_gt.hdf5)
  inputs=$(ls output/grasp_collect_no_rot/robot_gt/round_*/${obj}_robot_gt.hdf5 2>/dev/null)
  [ -n "$inputs" ] || continue
  python3 tools/merge_robot_gt.py --obj "$obj" --output \
    "output/grasp_collect_no_rot/merged/${obj}_robot_gt_merged.hdf5" \
    --inputs $inputs
done
```

---

## 5. Pool-based 流水线（候选池生成 + 候选池 sim batch，当前推荐）

将 **候选生成** 与 **Isaac sim** 拆成两个脚本，共用 `output/grasp_collect_no_rot/` 目录结构（`candidates/round_R`、`robot_gt/round_R`、`merged/`、`state.json`、`summary.csv`），与 Legacy batch 兼容。

### 5.1 流程概览

```text
batch_gen_candidates_pool.py（候选池生成）
    → candidates/pool/{obj}_grasp.hdf5   （每物体 candidate 池，CPU sampler）

batch_sim_candidates_pool.py（候选池 sim batch）
    → 读 pool + merged + sim_pool_registry.json
    → 每轮规划 slots（默认 weighted；可选 equal / max-success cap）
    → 展开为 candidate × 4 yaw task；early-stop 可减少实际 sim 次数
    → 长驻 Isaac worker（sim/run_grasp_sim_pool.py），chunk 增量写 results
    → sync_queue_and_registry_from_chunks：chunk 为唯一真相
    → candidates/round_R/、robot_gt/round_R/、自动 merge
    → pool 耗尽时自动再调 batch_gen_candidates_pool.py 补货（auto-refill）
```

| 脚本 | 作用 | Sim |
|------|------|-----|
| **`batch_gen_candidates_pool.py`** | 为 merged 成功数 `< threshold` 的物体批量跑 `random_grasp_sampler` | 无 |
| **`batch_sim_candidates_pool.py`** | 从 pool 抽 candidate 并 sim（默认 weighted；4 固定 Z-yaw） | `run_grasp_sim_pool.py` |

**与 Legacy 的主要区别：**

- 不是每轮给**全部**物体重新 gen 20 个 candidate，而是从 **pool** 里按策略抽 slot（默认 merged 成功 **加权** `1/(success+1)`；`--equal-object-prob` 等概率；`--max-success-per-object` 封顶）。
- Registry **resolved**：任一 yaw **成功**（early-stop），或 4 yaw 都 attempted 且全失败（标 `simulated`，不再重抽）。
- **Early-stop（默认）：** 同一 candidate 任一 yaw 成功后，其余 yaw 写 synthetic `skipped` chunk 行，不进 Isaac。禁用：`--no-early-stop-yaw-on-success`。
- Sim 侧 **长驻 worker**，chunk 内顺序多物体；**换物体不重建 World/Franka**；chunk 异常退出 **最多自动重试 2 次**。
- 同一张 GPU 上多个 worker **错开启动**（默认 45s，`--same-gpu-stagger-s`；每 worker 独立 Kit cache 目录，减轻 kvdb/CUDA 竞态）。
- **续跑：** `sim_logs/round_R/chunks/chunk_*_results.json` 为真相；`state.round` 仅 **sync_ok** 后 +1。

### 5.2 输出路径（在 Legacy 基础上新增）

| 路径 | 内容 |
|------|------|
| `candidates/pool/{OBJ}_grasp.hdf5` | `batch_gen_candidates_pool.py` 生成的 candidate 池 |
| `candidates/pool/gen_pool_manifest.json` | 候选池生成进度 manifest |
| `sim_pool_registry.json` | `(obj, candidate_key)` 的 yaw 进度 / `success_yaws` / resolved |
| `round_{R}_task_queue.json` | task 队列、`completed_task_ids`、`object_sampling` |
| `sim_logs/round_{R}/chunks/chunk_*` | worker chunk、`chunk_*_results.json`、`chunk_*_progress.json` |
| `sim_logs/round_{R}/chunks/synthesized_skipped_results.json` | early-stop 合成的 skipped 行（sync 时合并） |
| `sim_logs/round_{R}/chunks/accumulated_results.json` | resume 前归档的 round 级 results（chunk 重分片不丢进度） |
| `sim_logs/round_{R}/kit_cache/` | 每 worker 独立 Kit/Omni 缓存目录 |
| `sim_logs/round_{R}/chunk_*_gpu*.log` | 各 worker Isaac 日志 |

`candidates/round_R/`、`robot_gt/round_R/`、`merged/`、`state.json`、`summary.csv` 与 Legacy **同一约定**。

### 5.3 候选池生成（`batch_gen_candidates_pool.py`）

只对 **merged 目录里已有** `{obj}_robot_gt_merged.hdf5` 且 successful 条数 **低于** `--success-threshold` 的物体生成（无 merged 文件的物体跳过）。

```bash
conda activate bundlesdf
export PROJ=/path/to/V2AP
cd "$PROJ"

python3 scripts/batch_gen_candidates_pool.py \
  --merged-dir output/grasp_collect_no_rot/merged \
  --output-dir output/grasp_collect_no_rot/candidates/pool \
  --success-threshold 20 \
  --target 50 \
  --sampler-workers 16 \
  --force
```

- `--target 50`：每个待补物体在 pool 里目标 candidate 数（传给 `random_grasp_sampler --target`）。
- `batch_sim_candidates_pool.py` 在 `pool_exhausted` 时会 **自动** 以 median(merged success) 为 threshold 再调 `batch_gen_candidates_pool.py`（可用 `--no-auto-refill` 关闭）。

### 5.4 候选池 sim batch 正式生产（双卡 × 每卡 4 worker）

**从 Legacy 切到 Pool 时：** 若旧 `batch_grasp_collect` 在 round 14 中途停下，请先把 `state.json` 设为 **15**，避免 `batch_sim_candidates_pool.py` 写入未完成的 `round_0014`：

```bash
# 一次性（按需）
echo '{"round": 15, "objects": {}}' > output/grasp_collect_no_rot/state.json
```

**生产命令（bundlesdf 内跑 batch 即可；Isaac 由 `python.sh` 子进程启动）：**

```bash
conda activate bundlesdf
export ISAAC_SIM_PATH=/path/to/isaac-sim   # 例: /home/vision/isaacsim
cd "$PROJ"

python3 scripts/batch_sim_candidates_pool.py \
  --outdir output/grasp_collect_no_rot \
  --resume \
  --max-rounds 10 \
  --sim-gpu-ids 0,1 \
  --sim-per-gpu 4 \
  --headless
```

**物体抽样（与 hard 文档一致）：**

```bash
# 默认：低成功物体更常被抽到
python3 scripts/batch_sim_candidates_pool.py --resume --max-rounds 10 ...

# 物体等概率
python3 scripts/batch_sim_candidates_pool.py --equal-object-prob --max-rounds 1

# merged 累计成功 ≥80 的物体本轮不再参与规划
python3 scripts/batch_sim_candidates_pool.py --max-success-per-object 80 --max-rounds 1
```

| 参数 | 默认 | 含义 |
|------|------|------|
| `--outdir` | `output/grasp_collect_no_rot` | 实验根目录 |
| `--pool-dir` | `{outdir}/candidates/pool` | candidate 池 |
| `--merged-dir` | `{outdir}/merged` | 加权 / cap 用 merged |
| `--slots-per-round` | 500 | 每轮 candidate **槽位**数（×4 yaw = 规划 task 上限） |
| `--max-rounds` | 1 | 本次最多跑几轮 |
| `--pool-target` | 50 | auto-refill 时 `batch_gen_candidates_pool.py` 的 `--target` |
| `--score-threshold` | 70 | auto-refill sampler 分数门槛 |
| `--sim-gpu-ids` | `0` | 物理 GPU 列表 |
| `--sim-per-gpu` | 1 | 每张 GPU 并行 Isaac worker 数 |
| `--sim-timeout` | 7200 | 单 worker chunk 超时（秒） |
| `--resume` | 关 | 读 `state.json`；从 chunk 重建 queue 并补 pending |
| `--no-auto-refill` | 关 | 关闭 pool 空时自动候选池生成 |
| `--equal-object-prob` | 关 | slot 规划时物体等概率 |
| `--max-success-per-object` | 无 | merged 成功 ≥ N 的物体 prob=0 |
| `--no-early-stop-yaw-on-success` | 关 | 禁用 early-stop |
| `--plan-seed` | 无 | slot 规划可复现 |
| `--same-gpu-stagger-s` | `45` | 同 GPU 上多 worker 启动间隔（秒） |
| `--incremental-merge` | 关 | merge：已有 `merged/` + 仅本轮 `robot_gt/round_R` |
| `--full-merge` | 关 | 强制全量扫描 `robot_gt/round_*` |

迁服务器续跑示例（不必拷贝历史 `robot_gt/round_*`）：

```bash
python3 scripts/batch_sim_candidates_pool.py \
  --outdir output/grasp_collect_no_rot \
  --incremental-merge --resume --max-rounds 5 ...
```

上例：**2 × 4 = 8** 个 Isaac 进程；每轮规划 **500×4 = 2000** task（early-stop 下实际 sim 可能更少）。

**轮次：** `state.json` 里 `"round": N` → 下一批写 `round_{N:04d}`；`--resume --max-rounds 10` 跑 `round_N` … `round_{N+9}`。`state.round` 仅在轮末 **`sync_queue_and_registry_from_chunks` 返回 sync_ok** 后 +1；不要求 2000/2000 全跑满。

**Crash / 续跑 / registry：**

| 情况 | 行为 |
|------|------|
| worker **0 results**（未写出 chunk results） | task 不在 `completed_task_ids`；`--resume` 补跑 |
| worker chunk 崩溃 | batch **自动重试 ≤2 次**（间隔 15s），重试前 sync chunk |
| mid-round 中断 / resume 重分 chunk | 重分 chunk 前写入 `accumulated_results.json`；~5s / worker 结束 / Ctrl+C / 轮末从 chunk+archive 扫盘 sync |
| 有 chunk 行的 task | **不会**重复 sim |
| 4 yaw 全失败 | 标 `simulated`，下轮不再抽 |
| 任一 yaw 成功 | early-stop + `success_yaws`；candidate **resolved** |

Worker 每 task 增量写 `chunk_*_results.json` 与 `chunk_*_progress.json`。

### 5.5 Smoke test（隔离 outdir）

**单卡 + 同进程换物体（2 物体 × 4 yaw）：**

```bash
python3 scripts/batch_sim_candidates_pool.py \
  --outdir output/grasp_collect_smoke \
  --pool-dir output/grasp_collect_no_rot/candidates/pool \
  --merged-dir output/grasp_collect_no_rot/merged \
  --max-rounds 1 --slots-per-round 2 \
  --sim-gpu-ids 0 --sim-per-gpu 1 \
  --headless --no-auto-refill
```

**双卡 4 进程 + 每 chunk 2 物体（测 swap + stagger）：**

```bash
python3 scripts/batch_sim_candidates_pool.py \
  --outdir output/grasp_collect_smoke_4p_swap \
  --pool-dir output/grasp_collect_no_rot/candidates/pool \
  --merged-dir output/grasp_collect_no_rot/merged \
  --max-rounds 1 --slots-per-round 8 \
  --sim-gpu-ids 0,1 --sim-per-gpu 2 \
  --headless --no-auto-refill
```

> `slots-per-round 4` + `sim-per-gpu 2` 时每 chunk 仅 1 物体，**测不到**同进程换物体；要 swap 需每 chunk ≥2 物体（见上 `slots=8` 或 `slots=4, sim-per-gpu=1`）。

### 5.6 监控

```bash
tail -f output/grasp_collect_no_rot/summary.csv
cat output/grasp_collect_no_rot/state.json
cat output/grasp_collect_no_rot/round_0015_task_queue.json | python3 -m json.tool | head
```

---

## 6. 参数速查（Legacy `batch_grasp_collect.py`）

| 参数 | 默认 | 含义 |
|------|------|------|
| `--sampler-workers` | 4 | Phase 1 CPU 并行数 |
| `--sim-gpu-ids` | `0` | 物理 GPU 列表，如 `0,1` |
| `--sim-per-gpu` | 1 | **每张 GPU** 同时跑的 Isaac 数 |
| `--target` | 20 | 每物体每轮候选数 |
| `--max-rounds` | 10 | 轮数（默认 `round_0000`…`0009`） |
| `--no-convert` | 关 | 跳过 USD 转换 |
| `--resume` | 关 | 跳过已有 HDF5 |
| `--merge-deduplicate` | 关 | 写 `merged/` 时去掉相近 pose |
| `--incremental-merge` | 关 | 读已有 `merged/` + 仅本轮 `robot_gt/round_R`（迁服务器不必拷历史 `round_*`） |
| `--full-merge` | 关 | 强制全量扫描 `robot_gt/round_*`（覆盖 incremental） |
| `--rotation` | 关 | 加上则使用 `rotation.json`（一般不要） |

### HDF5 中的 `mesh_prerotation/`（pose 级）

**每个 pose**（`candidates/candidate_i`、`successful_grasps/grasp_i` 等）下各有自己的 `mesh_prerotation/`，记录该 pose 生成/测试时 mesh 实际用的预旋转。文件根级不再写。

| 情况 | `euler_xyz_deg` / `matrix` |
|------|----------------------------|
| 默认 `--no-rotation` | `[0,0,0]` + 单位阵，`method=identity` |
| `--rotation` | 与 `rotation.json` 一致（经 canonical 阈值） |

| 字段 | 含义 |
|--------|------|
| `euler_xyz_deg` | 该 pose 对应的欧拉角 (度, xyz) |
| `matrix` | 对应 3×3 旋转矩阵 |
| `@method` | `identity` 或 `rotation.json` 的 method |

**总 sim 并行数** = `len(sim-gpu-ids) × sim-per-gpu`  
例：`0,1` + `sim-per-gpu 1` → 最多 2 个 Isaac 同时跑。

---

## 7. 常见问题

### `No module named 'pxr'`

在 bundlesdf 中：`pip install usd-core`，或用已装好 pxr 的环境跑 `convert_obj_usd.py`。

### Sim 找不到 USD

确认存在：`output/obj_usd/oakink/{OBJ}.usd`，或去掉 `--no-convert` 让 batch 自动转换。

### 可视化 mesh 躺着、抓取不对齐

候选需 **no-rotation**；`vis_grasp_candidates` 会从 HDF5 `metadata/no_rotation` 自动对齐。旧图请重新生成。

### 可视化全是红色 FAILED

需加 `--robot-gt` 或 `--success raycast_1 ...`；仅候选 HDF5 不会标成功。

### 多轮候选是否相同？

**不同**。无固定 seed，每轮 `--force` 重新随机采样。

---

## 8. 最小命令清单（复制即用）

```bash
# 0. 环境（路径改成你的）
conda activate bundlesdf
export PROJ=/path/to/V2AP
export ISAAC_SIM_PATH=/path/to/your/isaac-sim
cd "$PROJ"

# 1. USD（全量，只需一次）
python3 tools/convert_obj_usd.py --dataset oakink --no-rotation --force

# 2. 显存压测（可选；若无 A01001_grasp.hdf5 先做第 3.0 节 sampler）
python3 scripts/benchmark_sim_parallel.py --gpu 0 --start-n 6 --min-free-gb 3

# 3. Pool batch（当前推荐；需先有 pool，见第 5.3 节）
python3 scripts/batch_sim_candidates_pool.py \
  --outdir output/grasp_collect_no_rot \
  --resume --max-rounds 10 \
  --sim-gpu-ids 0,1 --sim-per-gpu 4 \
  --headless

# 3-legacy. Legacy batch（双卡，默认 10 轮 round_0000..0009）
python3 scripts/batch_grasp_collect.py \
  --dataset oakink \
  --sampler-workers 8 \
  --sim-gpu-ids 0,1 \
  --sim-per-gpu 1 \
  --headless \
  --no-convert

# 3b. 跑满 10 轮后再加 5 轮（不覆盖旧 round）:
# python3 scripts/batch_grasp_collect.py --dataset oakink --max-rounds 5 --resume ...

# 4. 抽查可视化
python3 tools/vis_grasp_candidates.py \
  --hdf5 output/grasp_collect_no_rot/candidates/round_0000/A01001_grasp.hdf5 \
  --robot-gt output/grasp_collect_no_rot/robot_gt/round_0000/A01001_robot_gt.hdf5 \
  --only-success
```

---

## 9. 相关文件

| 脚本 | 作用 |
|------|------|
| `tools/convert_obj_usd.py` | PLY → USD |
| `tools/random_grasp_sampler.py` | 候选抓取 |
| `sim/run_grasp_sim.py` | Isaac 验证（单物体 / Legacy batch） |
| `sim/run_grasp_sim_pool.py` | Isaac 验证（Pool batch 长驻 worker） |
| `scripts/batch_gen_candidates_pool.py` | **候选池生成**：批量生成 candidate pool |
| `scripts/batch_sim_candidates_pool.py` | **候选池 sim batch**：加权 sim + merge |
| `tools/grasp_pool_common.py` | Pool 规划、registry、task queue |
| `scripts/batch_grasp_collect.py` | Legacy 全量两阶段 batch |
| `scripts/benchmark_sim_parallel.py` | 显存 / 并行数压测 |
| `tools/vis_grasp_candidates.py` | 候选 + 成功 PNG |
| `tools/merge_robot_gt.py` | 多轮 GT 合并（batch 内自动调用） |
