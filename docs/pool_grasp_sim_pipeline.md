# 候选池抓取 Sim 流水线（Pool → Isaac → merged）

从 **预生成的 candidate pool**（`candidates/pool/{obj}_grasp.hdf5`）出发，用 Isaac Sim + cuRobo 批量验证抓取，写入 `robot_gt/round_R/`，合并进 `merged/`，并可用 `state.json` 多轮续跑。

**本文档包含：** 新服务器部署、HF 数据集下载、pool sim 运行、HF 上传维护（原独立 upload 指南已合并到 §12）。

HDF5 schema、Legacy 对比见 [`grasp_collect_pipeline.md`](grasp_collect_pipeline.md) 第 5 节。

---

## 0. 新服务器 Setup Guide

按顺序完成；完成后用 **§11 部署核对清单** 打勾。

### 0.1 架构（两层 Python）

```text
┌─────────────────────────────────────────────────────────┐
│  Conda（如 bundlesdf）                                   │
│  batch_sim_candidates_pool.py — 规划、merge、续跑        │
│  batch_gen_candidates_pool.py — auto-refill / 手动补货   │
│  requirements.txt + rtree + huggingface_hub              │
└──────────────────────────┬──────────────────────────────┘
                           │ 子进程
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Isaac Sim: $ISAAC_SIM_PATH/python.sh                    │
│  sim/run_grasp_sim_pool.py + isaacsim + cuRobo           │
│  每 GPU × --sim-per-gpu 个长驻 worker                    │
└─────────────────────────────────────────────────────────┘
```

### 0.2 Phase 0 — 机器前提

| 项 | 说明 |
|----|------|
| GPU | NVIDIA；`--sim-per-gpu` 按显存压测（OOM 则降低） |
| CUDA / 驱动 | 与 Isaac Sim 版本匹配 |
| 磁盘 | HF 包约 4GB+；`robot_gt/`、`sim_logs/` 持续增长 |
| 网络 | `git clone`、`hf download`、（装 Isaac 时）NVIDIA 源 |

### 0.3 Phase 1 — 代码

```bash
git clone <repo-url> V2AP
cd V2AP
git checkout <branch>
```

| 脚本 | 作用 |
|------|------|
| `scripts/batch_sim_candidates_pool.py` | **主入口**：pool sim batch |
| `scripts/batch_gen_candidates_pool.py` | CPU 候选池生成 |
| `scripts/merge_candidate_pools.py` | 离线合并多个 pool 目录 |
| `sim/run_grasp_sim_pool.py` | Isaac 长驻 worker |

### 0.4 Phase 2 — HuggingFace 数据

见 **§3** 下载。Dataset（当前 HF 仓库名，见 §3.0）：

```bash
export PROJ=/path/to/V2AP
export HF_DATASET=UCBProject/hard_obj_grasp_collect_pipeline   # 见 §3.0
cd "$PROJ"

pip install -U huggingface_hub
hf auth login

hf download "$HF_DATASET" --repo-type dataset --local-dir "$PROJ"
```

新部署最小子集（`--no-auto-refill`）：

```bash
hf download "$HF_DATASET" sim --repo-type dataset --local-dir "$PROJ"
hf download "$HF_DATASET" output/obj_usd --repo-type dataset --local-dir "$PROJ"
hf download "$HF_DATASET" output/grasp_collect_no_rot --repo-type dataset --local-dir "$PROJ"
```

下载后检查：`candidates/pool/`、`merged/`、`state.json`；**勿**从其它机器拷贝 `sim_pool_registry.json`。

### 0.5 Phase 3 — Conda

```bash
conda create -n bundlesdf python=3.10 -y
conda activate bundlesdf
cd "$PROJ"
pip install -r requirements.txt
pip install rtree huggingface-hub
```

Sim 不在 conda 里装 `isaacsim`；USD 转换可选 `pip install usd-core`。

### 0.6 Phase 4 — Isaac Sim

1. 按 NVIDIA 文档安装（版本与参考机对齐）。
2. `export ISAAC_SIM_PATH=/path/to/isaac-sim`（目录含 `python.sh`）。

### 0.7 Phase 5 — cuRobo

1. 克隆 cuRobo（如 `~/Project/curobo`），在 Isaac Python 中安装。
2. 必要时改 `sim/run_grasp_sim_pool.py` 顶部 `sys.path`。
3. `$ISAAC_SIM_PATH/python.sh -c "import curobo; print('OK')"`

### 0.8 Phase 6 — Preflight

```bash
conda activate bundlesdf
export PROJ=/path/to/V2AP
export ISAAC_SIM_PATH=/path/to/isaac-sim
cd "$PROJ"

test -f sim/assets_franka/franka.usd && echo OK franka
test -f sim/assets_scene/Collected_default_environment/default_environment.usd && echo OK scene
ls output/obj_usd/oakink/*.usd 2>/dev/null | head -1
ls output/grasp_collect_no_rot/candidates/pool/*_grasp.hdf5 2>/dev/null | wc -l
ls output/grasp_collect_no_rot/merged/*_merged.hdf5 2>/dev/null | wc -l
cat output/grasp_collect_no_rot/state.json
test -f "$ISAAC_SIM_PATH/python.sh" && echo OK isaac
```

### 0.9 Phase 7 — Smoke → 生产

Smoke：§7。生产示例：

```bash
python3 scripts/batch_sim_candidates_pool.py \
  --outdir output/grasp_collect_no_rot \
  --resume --max-rounds 1 \
  --equal-object-prob --max-success-per-object 200 \
  --incremental-merge --no-auto-refill \
  --sim-gpu-ids 0,1 --sim-per-gpu 4 \
  --same-gpu-stagger-s 45 --headless
```

建议 `tmux` / `screen`。

---

## 1. 流水线概览

```text
candidates/pool/{obj}_grasp.hdf5
        │
        ▼  batch_sim_candidates_pool.py
        │    · merged/ → slot 规划（weighted / equal / cap）
        │    · sim_pool_registry.json → 跳过已 resolved
        │    · 4 固定 Z-yaw × N slots → Isaac workers
        ▼
robot_gt/round_R/{obj}_robot_gt.hdf5
        │
        ▼  merge（推荐 --incremental-merge）
merged/{obj}_robot_gt_merged.hdf5
        │
        ▼  state.round += 1（chunk↔queue sync_ok）
```

| 组件 | 脚本 |
|------|------|
| Pool sim | `batch_sim_candidates_pool.py` |
| Pool 生成 | `batch_gen_candidates_pool.py` |
| Pool 离线合并 | `merge_candidate_pools.py` |
| Isaac worker | `run_grasp_sim_pool.py` |

典型根目录：`output/grasp_collect_no_rot/`。

**规划与 cap 只读 `merged/`**，不扫历史 `robot_gt/round_*`。合并推荐 **`--incremental-merge`**；全量重建用 **`--full-merge`**。

---

## 2. 获取代码

GitHub 含脚本与 `sim/env_config/`；二进制与 HDF5 在 HF §3。

---

## 3. HuggingFace 数据集（下载）

### 3.0 仓库名

当前线上 dataset：

**[UCBProject/hard_obj_grasp_collect_pipeline](https://huggingface.co/datasets/UCBProject/hard_obj_grasp_collect_pipeline)**

（HF 上仍为历史 slug；内容与通用 pool sim 一致。若 org 新建 dataset，建议命名为 `pool_grasp_collect_pipeline` 并更新下文 `HF_DATASET`。）

顶层路径 **mirror 仓库**（`sim/`、`output/`、`data_hub/`）：

```bash
export HF_DATASET=UCBProject/hard_obj_grasp_collect_pipeline
hf download "$HF_DATASET" --repo-type dataset --local-dir "$PROJ"
```

### 3.1 文件清单

| Dataset 路径 | 必须 | 本地 |
|--------------|------|------|
| `sim/assets_franka/` | ✅ | `sim/assets_franka/` |
| `sim/assets_scene/` | ✅ | `sim/assets_scene/` |
| `output/obj_usd/` | ✅ | `output/obj_usd/` |
| `.../candidates/pool/` | ✅ | 标准 pool |
| `.../merged/` | ✅ | 规划 / merge 起点 |
| `.../state.json` | ✅ | 续跑 round |
| `.../sim_pool_registry.json` | 宜空 | 新部署勿拷它机 |
| `.../robot_gt/` | 可选 | 本地 sim 产生；规划不依赖 |
| `data_hub/...` | auto-refill | §8 |

**不在 HF：** Isaac Sim、cuRobo。

### 3.2 下载后目录树

```text
$PROJ/
├── scripts/ ...
├── sim/                    ← HF + GitHub
├── output/
│   ├── obj_usd/            ← HF
│   └── grasp_collect_no_rot/
│       ├── candidates/pool/
│       ├── merged/
│       ├── state.json
│       ├── sim_pool_registry.json
│       └── robot_gt/round_*   ← 本地
└── data_hub/               ← HF（auto-refill）
```

### 3.3 Pool 路径与 registry

- **唯一标准路径：** `{outdir}/candidates/pool/`（dataset 内也不要另建 `pool_500_*` 给下游）。
- 离线合并后 `rsync` 进标准路径再上传（§8.2、§12）。
- **整包换 pool** 后清 registry：

```bash
rm -f output/grasp_collect_no_rot/sim_pool_registry.json
```

### 3.4 本地生成 USD（备选）

```bash
pip install usd-core
python3 tools/convert_obj_usd.py --dataset oakink --no-rotation --force
python3 tools/convert_obj_usd.py --dataset ycb --no-rotation --force
```

---

## 4. 运行前数据检查

| 检查项 | 命令 |
|--------|------|
| Franka | `test -f sim/assets_franka/franka.usd` |
| 场景 | `test -f sim/assets_scene/Collected_default_environment/default_environment.usd` |
| 物体 USD | `ls output/obj_usd/oakink/*.usd \| head -1` |
| Pool | `ls .../candidates/pool/*_grasp.hdf5 \| wc -l` |
| Merged | `ls .../merged/*_merged.hdf5 \| wc -l` |
| Round | `cat .../state.json` |
| auto-refill | `ls data_hub/meshes/.../mesh.ply \| head -1` |

无 `data_hub` → `--no-auto-refill`。

---

## 5. 软件环境

| 组件 | 说明 |
|------|------|
| Conda `bundlesdf` | batch、sampler、merge |
| Isaac Sim | `ISAAC_SIM_PATH/python.sh` |
| cuRobo | Isaac Python 内 |

```bash
pip install -r requirements.txt && pip install rtree
```

---

## 6. 运行 Pool Sim Batch

### 6.1 并行度

进程数 = `len(--sim-gpu-ids) × --sim-per-gpu`。同 GPU 默认 **`--same-gpu-stagger-s 45`**。

### 6.2 生产命令

```bash
python3 scripts/batch_sim_candidates_pool.py \
  --outdir output/grasp_collect_no_rot \
  --resume --max-rounds 10 \
  --incremental-merge --no-auto-refill \
  --sim-gpu-ids 0,1,2,3 --sim-per-gpu 2 \
  --headless
```

默认每轮 **500 slots × 4 yaw**（early-stop 会减少实际 sim 次数）。

### 6.3 物体抽样

| 模式 | 行为 |
|------|------|
| weighted（默认） | `1/(n_success+1)`，n 来自 merged |
| `--equal-object-prob` | 等概率 |
| `--max-success-per-object N` | 成功 ≥ N 的物体本轮不抽 |

### 6.4 Early-stop 与 registry

- 默认：任一 yaw 成功 → 其余 yaw 跳过。
- Resolved：成功，或 4 yaw 均 attempted 且全失败。
- `--no-early-stop-yaw-on-success` 关闭 early-stop。

### 6.5 Merge

| 标志 | 行为 |
|------|------|
| `--incremental-merge` | 已有 merged + 本轮 robot_gt |
| `--full-merge` | 扫描全部 `robot_gt/round_*` |

### 6.6 主要 CLI

| 参数 | 默认 | 含义 |
|------|------|------|
| `--outdir` | `output/grasp_collect_no_rot` | 实验根 |
| `--pool-dir` | `{outdir}/candidates/pool` | pool HDF5 |
| `--merged-dir` | `{outdir}/merged` | 规划数据源 |
| `--slots-per-round` | 500 | 每轮槽位 |
| `--resume` | 关 | 续跑 |
| `--no-auto-refill` | 关 | 不自动补 pool |
| `--incremental-merge` | 关 | 增量 merge |
| `--same-gpu-stagger-s` | 45 | 同 GPU 启动间隔 |

### 6.7 监控

```bash
tail -f output/grasp_collect_no_rot/summary.csv
cat output/grasp_collect_no_rot/state.json
tail -f output/grasp_collect_no_rot/sim_logs/round_*/chunk_*_gpu*.log
```

---

## 7. Smoke test

```bash
python3 scripts/batch_sim_candidates_pool.py \
  --outdir output/grasp_collect_smoke \
  --pool-dir output/grasp_collect_no_rot/candidates/pool \
  --merged-dir output/grasp_collect_no_rot/merged \
  --max-rounds 1 --slots-per-round 4 \
  --sim-gpu-ids 0 --sim-per-gpu 1 \
  --headless --no-auto-refill --incremental-merge
```

期望：`grasp_collect_smoke/robot_gt/round_0000/*.hdf5` 有输出。

---

## 8. 候选池生成与合并

### 8.1 生成

```bash
python3 scripts/batch_gen_candidates_pool.py \
  --merged-dir output/grasp_collect_no_rot/merged \
  --output-dir output/grasp_collect_no_rot/candidates/pool \
  --success-threshold 20 --target 500 \
  --sampler-workers 16 --force
```

`--force` 覆盖 pool 后需清对应物体的 registry（auto-refill 时 batch 会自动清）。

### 8.2 离线合并

```bash
python3 scripts/merge_candidate_pools.py \
  --dir-a output/pool_a --dir-b output/pool_b --out-dir output/pool_merged

rsync -a output/pool_merged/ output/grasp_collect_no_rot/candidates/pool/
```

再按 §12 上传 HF。

---

## 9. 续跑说明

| 场景 | 做法 |
|------|------|
| 正常续跑 | 保留 state / merged / pool / robot_gt / registry；`--resume` |
| crash / Ctrl+C | `--resume` 从 chunk 补无记录 task |
| 新服务器 | HF 拉 pool+merged+state；`--incremental-merge`；空 registry |
| 换 pool | 删 `sim_pool_registry.json` 或按物体清条目 |
| 重建 merged | `--full-merge` |

---

## 10. 常见问题

| 现象 | 处理 |
|------|------|
| `USD not found` | `output/obj_usd/` 或 `convert_obj_usd.py` |
| `cuRobo` import 失败 | 改 `run_grasp_sim_pool.py` sys.path |
| pool 空 | `data_hub` 或 `--no-auto-refill` |
| GPU OOM | 降 `--sim-per-gpu`、增 stagger |
| 换 pool 后错乱 | 清 registry |
| sampler OOM | 降 `--sampler-workers` |

---

## 11. 部署核对清单

- [ ] git clone + 正确 branch
- [ ] HF：`sim/`、`obj_usd/`、`pool/`、`merged/`、`state.json`
- [ ] 空 registry（新部署）
- [ ] conda + Isaac + cuRobo
- [ ] §4 preflight、§7 smoke、§6 生产命令

---

## 12. HuggingFace 上传（维护者）

把 pool sim 所需 **最小资产** 打进 HF Dataset。下载说明见 §3。

### 12.0 前置

```bash
pip install -U huggingface_hub
hf auth login

export PROJ=/path/to/V2AP
export HF_DATASET=UCBProject/hard_obj_grasp_collect_pipeline
export STAGING=/tmp/hf_pool_grasp_staging
```

需对 [UCBProject](https://huggingface.co/UCBProject) org 有 **write** 权限。

### 12.1 创建空 Dataset（仅首次）

网页：UCBProject → New dataset → 名称（建议 `pool_grasp_collect_pipeline`）→ Private。

```bash
hf repos create "$HF_DATASET" --repo-type dataset --private
```

### 12.2 准备 staging

```bash
rm -rf "$STAGING" && mkdir -p "$STAGING" && cd "$PROJ"
```

**Sim 资产：**

```bash
mkdir -p "$STAGING/sim"
rsync -a sim/assets_franka/ "$STAGING/sim/assets_franka/"
rsync -a sim/assets_scene/  "$STAGING/sim/assets_scene/"
```

无 `assets_scene` 时可从 [`UCBProject/assets_scene`](https://huggingface.co/datasets/UCBProject/assets_scene) 下载后放入 `sim/assets_scene/`。

**物体 USD：**

```bash
mkdir -p "$STAGING/output"
rsync -a output/obj_usd/ "$STAGING/output/obj_usd/"
```

**`grasp_collect_no_rot`：**

Candidate pool **必须**在标准路径 `candidates/pool/`。合并多个 staging pool 后再 rsync：

```bash
# 示例：合并后进标准 pool
# rsync -a output/pool_merged/ output/grasp_collect_no_rot/candidates/pool/

mkdir -p "$STAGING/output/grasp_collect_no_rot/candidates/pool"
rsync -a output/grasp_collect_no_rot/candidates/pool/ \
  "$STAGING/output/grasp_collect_no_rot/candidates/pool/"
rsync -a output/grasp_collect_no_rot/merged/ \
  "$STAGING/output/grasp_collect_no_rot/merged/"
cp output/grasp_collect_no_rot/state.json \
  "$STAGING/output/grasp_collect_no_rot/state.json"
```

**不要上传：**

- `robot_gt/`（规划读 merged；新机器本地生成 round）
- `sim_logs/`、`summary.csv`、`*_task_queue.json`
- 整包换 pool 时通常 **不要** 带旧机器的 `sim_pool_registry.json`

可选：空 registry 供新部署：

```bash
echo '{}' > "$STAGING/output/grasp_collect_no_rot/sim_pool_registry.json"
```

**auto-refill — `data_hub`：**

```bash
mkdir -p "$STAGING/data_hub/meshes/SAM3DMesh" "$STAGING/data_hub/ProcessedData"
rsync -a data_hub/meshes/SAM3DMesh/rotated_mesh/ \
  "$STAGING/data_hub/meshes/SAM3DMesh/rotated_mesh/"
rsync -a data_hub/ProcessedData/train_fp_rotated/ \
  "$STAGING/data_hub/ProcessedData/train_fp_rotated/"
```

仅 `scale.json`（oakink / ycb，不要 mesh.ply）：

```bash
BASE=data_hub/ProcessedData/obj_meshes
DEST="$STAGING/data_hub/ProcessedData/obj_meshes"
for ds in oakink ycb; do
  find "$BASE/$ds" -name scale.json | while read -r f; do
    rel="${f#$BASE/}"
    mkdir -p "$DEST/$(dirname "$rel")"
    cp "$f" "$DEST/$rel"
  done
done
find "$DEST" -name scale.json | wc -l
find "$DEST" -name '*.ply' | wc -l   # 期望 0
```

**Dataset README（可选）：**

```bash
cat > "$STAGING/README.md" <<'EOF'
# Pool grasp collect pipeline assets

Download into V2AP repo root:
```bash
hf download UCBProject/hard_obj_grasp_collect_pipeline \
  --repo-type dataset --local-dir /path/to/V2AP
```

See `docs/pool_grasp_sim_pipeline.md`.
EOF
```

```bash
du -sh "$STAGING"/*
```

### 12.3 上传

整包：

```bash
hf upload "$HF_DATASET" "$STAGING/." . \
  --repo-type dataset \
  --commit-message "Update pool grasp collect assets"
```

大目录可分步上传：

```bash
hf upload "$HF_DATASET" "$STAGING/sim" sim --repo-type dataset
hf upload "$HF_DATASET" "$STAGING/output/obj_usd" output/obj_usd --repo-type dataset
hf upload "$HF_DATASET" "$STAGING/output/grasp_collect_no_rot" output/grasp_collect_no_rot --repo-type dataset
hf upload "$HF_DATASET" "$STAGING/data_hub" data_hub --repo-type dataset
```

中断续传：`hf upload-large-folder "$HF_DATASET" "$STAGING" --repo-type dataset`。

### 12.4 上传后验证

```bash
TMP=$(mktemp -d)
hf download "$HF_DATASET" sim/assets_franka/franka.usd \
  --repo-type dataset --local-dir "$TMP"
ls -la "$TMP/sim/assets_franka/franka.usd"
rm -rf "$TMP" "$STAGING"
```

### 12.5 Staging 目录树（参考）

```text
staging/
├── README.md
├── sim/{assets_franka,assets_scene}/
├── output/
│   ├── obj_usd/{oakink,ycb}/
│   └── grasp_collect_no_rot/
│       ├── candidates/pool/
│       ├── merged/
│       ├── state.json
│       └── sim_pool_registry.json  (可选空 {})
└── data_hub/
    ├── meshes/SAM3DMesh/rotated_mesh/
    └── ProcessedData/{train_fp_rotated,obj_meshes/**/scale.json}
```

---

## 13. 相关文档与脚本

| 资源 | 内容 |
|------|------|
| [`grasp_collect_pipeline.md`](grasp_collect_pipeline.md) | HDF5 schema、Legacy、USD/采样 |
| `scripts/batch_sim_candidates_pool.py` | Sim batch |
| `scripts/batch_gen_candidates_pool.py` | Pool 生成 |
| `scripts/merge_candidate_pools.py` | Pool 合并 |
