# Evaluation

本文档描述 evaluation pipeline：把评估主流程从 Isaac Sim 细节里拆出来。`evaluation/` 负责 episode 调度、policy adapter、结果格式；`sim/evaluation/` 负责 Isaac Sim 场景、Franka、RigidObject 和 cuRobo 执行。

## 当前能力

| 能力 | 入口 | 说明 |
|------|------|------|
| 单物体单次 eval | `evaluation/eval_single.py` | 一个 episode，一次 Isaac 启动 |
| 批量 eval | `evaluation/eval_batch.py` | **wrapper**：每个 episode 子进程调 `eval_single`（每次重启 Sim） |
| Pool eval | `evaluation/eval_pool.py` | 集中生成 solution/task queue，长驻 Isaac worker 内 reset/swap 物体 |
| Policy | `a2g_pdm` | 从 candidate HDF5 选一个 open-loop grasp |
| 可选生成 candidate | `--generate-candidate` | 调 `tools/glb_to_pdm_grasp.py`（real-machine GLB / rotated SAM3D mesh 等） |
| Sim z-yaw | `--z-yaw-deg` / batch 网格·随机 | 物体绕世界 Z 放置；与 yaw-conditioned PDM 对齐 |
| 随机抽解 | `--selection sample`，`--seed 42`（默认） | 同场景多次 trial 用可复现的 candidate 抽样 |
| 可选录像 | `--record-video` | viewport → `{episode_id}.mp4` |
| 批量选择性录像 | `--record-video --record-count-per-object K` | 每物体 N 次 trial 中随机录 K 段（K≥N 则全录） |

尚未实现（接口已预留）：GraspNet / DP3 closed-loop adapter。

相关文档：抓取采集 [`grasp_collect_pipeline.md`](grasp_collect_pipeline.md)、PDM [`pdm.md`](pdm.md)、real-machine mesh [`train_affordance.md`](train_affordance.md#inference-on-arbitrary-meshes-real-machine-glb)。

## 目录结构

```text
evaluation/
  eval_single.py              # 单物体单次 evaluation CLI
  eval_batch.py               # 批量 wrapper（子进程 eval_single）
  eval_pool.py                # pool orchestrator（solution + task chunk + worker launch）
  episode.py                  # 物体发现、episode_id、录像 trial 选择
  solution_gen.py             # IsaacSim 前集中生成/选择 policy solution
  task_queue.py               # task queue / chunk schema
  randomness.py               # 非固定 policy 采样 seed
  yaw.py                      # z-yaw 解析
  specs.py                    # 可序列化的 SceneSpec / PolicyOutput / ExecutionResult
  results.py                  # JSON / JSONL 结果写出
  policies/
    base.py                   # policy adapter 抽象接口
    a2g_pdm.py                # 当前 A2G/PDM candidate HDF5 adapter

sim/evaluation/
  scene_builder.py            # Isaac Sim scene setup, object placement, cuRobo mesh extraction
  context.py                  # SimEvaluationContext runtime handles
  curobo_executor.py          # cuRobo open-loop grasp executor
  run_eval_worker.py          # 长驻 IsaacSim worker，按 chunk 连续执行 episodes
  video_recorder.py           # viewport PNG → MP4
```

设计原则：

- `evaluation/` 尽量不依赖 Isaac Sim，适合写主流程、接口和结果格式。
- `sim/evaluation/` 可以 import Isaac Sim、cuRobo、Franka 和 RigidObject。
- policy 不直接 step simulation。policy 读取 context 后返回动作意图，executor 负责实际仿真执行和成功判定。

## 环境

Isaac Sim 需要用安装目录里的 `python.sh` 启动。不要依赖本地 alias，例如 `sim45`，因为不同机器不一定配置了它。

```bash
export PROJ=/home/vision/Project/V2AP
export ISAAC_SIM_PATH=/home/vision/isaacsim
cd "$PROJ"
```

如果在 conda 环境中启动 Isaac Sim 遇到 numpy/scipy 路径混用问题，可以用更干净的环境变量启动：

```bash
env -u PYTHONPATH -u PYTHONHOME $ISAAC_SIM_PATH/python.sh evaluation/eval_single.py --help
```

## HF Evaluation Assets

评估代码之外，还需要从 HF dataset 下载 assets。HF dataset 内的目录结构与 repo 根目录下的相对路径一致；下载后同步到 repo 根目录即可。

### 下载

```bash
cd /path/to/V2AP
huggingface-cli download UCBProject/<dataset-name> \
  --repo-type dataset \
  --local-dir /tmp/a2g_eval_assets
```

### 放置位置

将下载内容同步到 repo 根目录，保持相对路径不变：

```bash
cd /path/to/V2AP
rsync -a /tmp/a2g_eval_assets/ ./
```

同步后，以下路径应存在于 repo 中：

```text
data_hub/meshes/SAM3DMesh/rotated_mesh/                 # batch candidate generation 输入 mesh
data_hub/meshes/SAM3DMesh/rotated_mesh/unseen/{obj_id}/mesh.ply
                                                        # unseen 物体（obj_id 如 unseen_000）
data_hub/ProcessedData/obj_meshes/{oakink,ycb,unseen}/{obj_id}/scale.json
                                                        # 仅 metric scale；eval 不读 obj_meshes/mesh.ply 或 rotation.json
output/obj_usd/                                         # IsaacSim object USD + *_meta.json
output/obj_usd/unseen/                                  # unseen USD + *_meta.json
output/affordance_no_rot_executed/min20/checkpoints_v6/best_v6_model.pth
output/pdm/checkpoints_yaw/best_model.pth
sim/assets_scene/                                       # 可视背景/scene assets
sim/object_rotation_overrides.json                      # 物体放置 z_offset / rotation override
evaluation/configs/eval_objects_merged_success_ge30.csv # 推荐 evaluation object list (round>=3 成功数>=30)
```

`obj_meshes` 说明：

- eval pipeline 的 mesh 来自 `rotated_mesh/.../mesh.ply`，不是 `obj_meshes/.../mesh.ply`
- **不需要** `SAM3DMesh/meshes/`（原始未旋转 mesh，仅用于离线生成 rotated_mesh）
- candidate 生成和 USD 转换只会读取 `obj_meshes/{dataset}/{obj_id}/scale.json` 做米制缩放
- rotated mesh 已是 canonical frame，**不需要** `rotation.json`
- HF dataset 因此只打包 `scale.json`（oakink / ycb），不打包 `mesh.ply`、`splat.ply`、`rotation.json`

HF dataset 不包含以下内容，需要单独安装或在 eval 运行时生成：

```text
cuRobo source/install
Isaac Sim install
output/grasp_collect_no_rot/candidates/pool/            # pre-generated candidates
output/grasp_collect_no_rot/merged/                     # merged GT
output/evaluation/                                      # eval 结果、视频、日志
```

## 最小用法：已有 Candidate HDF5

如果已经有 A2G/PDM candidate HDF5：

```bash
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id A16013 \
  --candidate-hdf5 output/grasp_collect_no_rot/candidates/pool/A16013_grasp.hdf5 \
  --headless \
  --result-dir output/evaluation/single \
  --save-hdf5
```

这条命令会：

1. 启动 Isaac Sim。
2. 读取 candidate HDF5。
3. 默认选择 score 最高的 candidate。可用 `--selection index --candidate-index N` 或 `--selection sample` 改变选择方式。
4. 查找 `A16013.usd`：
   - `output/obj_usd/{oakink,ycb,arctic,dexycb,egocentric,ho3d_v3}/A16013.usd`
   - `sim/assets/A16013.usd`
5. 按 `sim/run_grasp_sim.py` 的默认场景放置 Franka、桌子和物体。可选 `--random-obj-xy` 在基准 XY `(0, 0.55)` 附近均匀抖动（默认 ±5cm），Z 仍用 `z_offset`。
6. 从 Isaac stage 提取物体 mesh，加入 cuRobo collision world。
7. 将 candidate 的 object-mesh grasp pose 转成 world pose，并适配到 Franka `panda_hand` frame。
8. cuRobo 规划并执行：
   - pre-grasp：带 object mesh 避障
   - final approach：清掉 object mesh，只保留桌面/地面，允许夹爪接触物体
   - lift：闭合夹爪后上提
9. 用 `object_final_z - object_initial_z > 0.03m` 判定 success。

## 可选：先生成 Candidate

对于 real-machine mesh，可以让 runner 先调用现有 A2G/PDM 生成 candidate，再启动 Isaac Sim：

```bash
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id IMG_4477 \
  --mesh data_hub/real_machine/sam3d_glb/IMG_4477.glb \
  --generate-candidate \
  --candidate-python python \
  --headless
```

注意：

- `--candidate-python python` 通常指当前 shell 里的 conda Python，用来跑 `tools/glb_to_pdm_grasp.py`。
- `--mesh` 支持 triangle mesh，不限 GLB；`tools/glb_to_pdm_grasp.py` 可处理 GLB/OBJ/PLY。
- 仿真阶段仍然需要 Isaac Sim 可加载的 USD。当前会按 obj id 搜索 `output/obj_usd/.../{obj_id}.usd` 或 `sim/assets/{obj_id}.usd`。
- 如果 real-machine 物体还没有 USD，需要先转换 USD，或后续给 runner 增加 `--usd-path` 接口。

对于 OakInk/YCB 这类已有 rotated SAM3D mesh 的物体，可以不传 `--mesh`，让 `eval_single.py` 按 `obj-id` 自动找：

```bash
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id A16013 \
  --generate-candidate \
  --candidate-python python \
  --headless
```

默认查找：

```text
data_hub/meshes/SAM3DMesh/rotated_mesh/{dataset}/{obj_id}/mesh.ply
```

也可以显式传入 rotated mesh：

```bash
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id A16013 \
  --mesh data_hub/meshes/SAM3DMesh/rotated_mesh/oakink/A16013/mesh.ply \
  --sam3d-rotated-mesh \
  --generate-candidate \
  --candidate-python python \
  --headless
```

`--sam3d-rotated-mesh` 会告诉生成脚本：该 mesh 已经是 rotated SAM3D frame，不要再做 real-machine GLB 的 `+90°X` 预旋转，并尽量按 rotated mesh / scale.json 的 metric convention 生成 candidate。

## 随机选 candidate（无固定 seed）

默认 `--selection top` 每次选最高分。批量或需要「同场景、不同解」时用：

```bash
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id C22001 \
  --candidate-hdf5 output/grasp_collect_no_rot/candidates/pool/C22001_grasp.hdf5 \
  --selection sample \
  --headless
```

不传 `--policy-seed` 时，policy 抽样使用 `--seed`（默认 42），并按 trial 偏移：`seed + trial * 10007`。显式 `--policy-seed N` 可只覆盖 policy 抽样。

`eval_single` 也可用 `--trial 2` 区分 episode_id 后缀；batch 会自动传入。

## Sim z-yaw（场景 + PDM 条件）

物体在桌上的绕 **世界竖直轴** 旋转，与 pool sim 的 `sim_z_yaw_deg` 一致。`SceneSpec.sim_z_yaw_deg` 写入 episode JSON；`scene_builder` 在放置 USD 时加到物体欧拉角 Z 上。

**单物体：**

```bash
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id C22001 \
  --candidate-hdf5 output/grasp_collect_no_rot/candidates/pool/C22001_yaw090_grasp.hdf5 \
  --z-yaw-deg 90 \
  --headless
```

使用 yaw-conditioned PDM（`output/pdm/checkpoints_yaw`）时，`--generate-candidate` 会把同一 `--z-yaw-deg` 传给 `glb_to_pdm_grasp`。sim 与生成 **必须用同一 yaw**，否则 grasp 与物体朝向不一致。

**批量** 见下文 `--z-yaw-deg` / `--z-yaw-grid` / `--z-yaw-random`。

## 可选录像

需要 viewport（headed 或 Xvfb）。`--record-video` 会写出 `{result_dir}/{episode_id}.mp4`，并在 episode JSON 里写 `video_path`。

如果当前环境没有 `DISPLAY`，`eval_single.py` 会自动用 `xvfb-run` 重新启动自己；不需要手动先启动一次 Xvfb。可用 `--no-auto-xvfb` 关闭这个行为。

```bash
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id C22001 \
  --candidate-hdf5 output/grasp_collect_no_rot/candidates/pool/C22001_grasp.hdf5 \
  --record-video \
  --result-dir output/evaluation/single
```

| Flag | 默认 | 说明 |
|------|------|------|
| `--record-video` | off | 写入 `{episode_id}.mp4` |
| `--record-every` | 3 | 每 N 次 `world.step` 抓一帧 |
| `--record-fps` | 30 | ffmpeg 合成帧率 |
| `--record-keep-frames` | off | 保留临时 PNG |
| `--xvfb-screen` | `-screen 0 1280x720x24` | 自动 Xvfb 的 screen 参数 |
| `--no-auto-xvfb` | off | 不自动重启到 Xvfb |

单物体时 `--record-video` 会强制 **headed**（关闭 headless），因需要 viewport。服务器上若没有真实显示器，默认自动使用 `xvfb-run -a`。依赖系统 `ffmpeg`；失败时 JSON 的 `execution.metadata.video_encode_error` 可能有说明，临时 PNG 留在 `.video_frames_tmp`（若未 `--record-keep-frames` 会尽量删除）。

## 输出模式

Isaac Sim 会产生大量 extension startup / deprecation / PhysX warning。evaluation 默认压制完整 stdout/stderr，只保留关键摘要。

| 模式 | flag | 终端输出 | 完整日志 |
|------|------|----------|----------|
| 默认 | 无 | 只打印关键摘要 | 不保存 |
| log-only | `--log-only` | 不打印 | 保存到 `{result_dir}/logs/{episode_id}.log` |
| loud | `--loud` | 完整 stdout/stderr + 关键摘要 | 保存到 `{result_dir}/logs/{episode_id}.log` |

默认关键摘要包括：

- episode / object / yaw
- candidate HDF5
- simulation app started
- scene ready
- video path（如果录像）
- plan summary：pregrasp / direct / final / lift 成功与否
- JSON / HDF5 / video 输出路径
- 最终 success/failure、`z_delta_m`、`failure_stage`

单物体示例：

```bash
# 不刷屏，只保存完整日志
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id C22001 \
  --candidate-hdf5 output/grasp_collect_no_rot/candidates/pool/C22001_grasp.hdf5 \
  --result-dir output/evaluation/single \
  --log-only

# 终端刷完整 Isaac 输出，同时保存日志
$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py \
  --obj-id C22001 \
  --candidate-hdf5 output/grasp_collect_no_rot/candidates/pool/C22001_grasp.hdf5 \
  --result-dir output/evaluation/single \
  --loud
```

## 批量 eval（wrapper）

`evaluation/eval_batch.py` **不**长驻 Isaac；每个 episode 子进程调用一次 `eval_single.py`（每次重启 Sim）。如果要避免每个 episode 冷启动 Isaac，优先用下一节的 `eval_pool.py`。

```bash
export ISAAC_SIM_PATH=/home/vision/isaacsim

# 从 pool 目录发现物体，每物体 5 次 trial，随机抽 candidate，不录像
python evaluation/eval_batch.py \
  --candidate-dir output/grasp_collect_no_rot/candidates/pool \
  --trials-per-object 5 \
  --selection sample \
  --headless \
  --result-dir output/evaluation/batch_pool

# 从 USD 目录发现物体（仍需有对应 candidate 或 --generate-candidate-each-trial）
python evaluation/eval_batch.py \
  --usd-root output/obj_usd/oakink \
  --candidate-dir output/grasp_collect_no_rot/candidates/pool \
  --trials-per-object 3 \
  --obj-limit 10

# z-yaw：固定 / 网格 / 随机池
python evaluation/eval_batch.py \
  --candidate-dir output/grasp_collect_no_rot/candidates/pool \
  --obj C22001 \
  --trials-per-object 4 \
  --z-yaw-grid 0,90,180,270

python evaluation/eval_batch.py \
  --candidate-dir output/grasp_collect_no_rot/candidates/pool \
  --trials-per-object 5 \
  --z-yaw-random \
  --z-yaw-pool 0,90,180,270

# 录像：默认录满所有 trial；或每物体只随机录 K 次
python evaluation/eval_batch.py \
  --candidate-dir output/grasp_collect_no_rot/candidates/pool \
  --obj C22001 \
  --trials-per-object 5 \
  --record-video \
  --record-count-per-object 2
```

批量输出：

- 每个 episode 仍写 `{episode_id}.json` / `episodes.jsonl`
- 汇总 `batch_summary.json`（returncode、success、video_path）

**每 trial 重新 PDM 采样**（需 `--mesh-root` 且物体有对应 `.glb`）：

```bash
python evaluation/eval_batch.py \
  --obj IMG_4477 \
  --trials-per-object 3 \
  --generate-candidate-each-trial \
  --mesh-root data_hub/real_machine/sam3d_glb \
  --z-yaw-deg 0 \
  --selection sample
```

`glb_to_pdm_grasp` 在 eval 生成路径下默认带 `--random-seed`，避免重复候选。

### 物体从哪来

| 参数 | 作用 |
|------|------|
| `--candidate-dir DIR` | 扫描 `*_grasp.hdf5` 得到 obj_id；并作为默认 candidate 路径 |
| `--usd-root DIR` | 扫描 `**/*.usd`（或单个 `.usd` 文件）得到 obj_id |
| `--obj ID` | 可重复，显式指定 |
| `--obj-list path.csv/json/txt` | 推荐 CSV；支持 enabled 列，也兼容 JSON / 纯文本 |
| `--obj-limit N` | 发现列表后只取前 N 个 |
| `--shuffle-objects` | 随机打乱物体顺序（无固定 seed） |

缺少 candidate 且未开 `--generate-candidate-each-trial` 的物体会 **skip** 并打印提示。

### 批量 CLI 速查

| 参数 | 默认 | 说明 |
|------|------|------|
| `--trials-per-object` | 1 | 每物体跑几次 eval（场景相同，解可不同） |
| `--selection` | `sample` | 与 eval_single 相同 |
| `--policy-seed` | 无 | 仅覆盖 `--selection sample`；默认用 `--seed` |
| `--seed` | `42` | 统一 eval 随机种子 |
| `--z-yaw-deg` | 无 | 全部 episode 固定 yaw |
| `--z-yaw-grid` | 无 | 如 `0,90,180,270`，按 trial 下标循环 |
| `--z-yaw-random` | off | 每 trial 从 `--z-yaw-pool` 抽一个 |
| `--record-video` | off | 见下 |
| `--record-count-per-object` | 全录 | 仅 `--record-video` 时有效；随机选 K 个 trial 录像 |
| `--generate-candidate-each-trial` | off | 每 trial 调 glb_to_pdm（需 `--mesh-root`） |
| `--log-only` | off | 传给每个 `eval_single` 子进程；终端不打印 episode 输出 |
| `--loud` | off | 传给每个 `eval_single` 子进程；完整 stdout/stderr + log |
| `--log-dir` | `{result-dir}/logs` | 子进程日志目录 |
| `--dry-run` | off | 只打印子进程命令 |

调度器用 **普通 `python`** 调 `eval_batch.py`；每个子进程用 `$ISAAC_SIM_PATH/python.sh evaluation/eval_single.py`。

## Pool eval（长驻 Isaac worker）

`evaluation/eval_pool.py` 把 batch 拆成两个阶段：

1. 普通 Python 阶段：发现物体，解析 trial/yaw，生成或复用 candidate HDF5，并把选中的 `PolicyOutput` 写成 `solutions/*.json`。
2. Isaac worker 阶段：把任务切成 `chunks/*.json`，每个 worker 启动一次 IsaacSim，第一题 `setup_scene`，后续同物体用 `reset_scene_pose`，换物体用 `swap_scene_object`。

```bash
export ISAAC_SIM_PATH=/home/vision/isaacsim

python evaluation/eval_pool.py \
  --obj A16013 \
  --obj S10010 \
  --trials-per-obj-yaw 2 \
  --generate-candidate-each-trial \
  --mesh-root data_hub/meshes/SAM3DMesh/rotated_mesh \
  --z-yaw-grid 0,90 \
  --selection sample \
  --headless \
  --sim-gpu-ids 0 \
  --sim-per-gpu 1 \
  --result-dir output/evaluation/pool_smoke \
  --save-hdf5
```

输出结构：

```text
output/evaluation/pool_smoke/
  solutions/{episode_id}.json       # 仿真前已经选好的 policy output
  solutions/manifest.json
  task_queue.json
  chunks/chunk_000.json
  chunks/chunk_000_progress.json
  chunks/chunk_000_results.json
  episodes/{episode_id}.json
  episodes/{episode_id}_robot_gt.hdf5
  episodes.jsonl
  batch_summary.json
  eval_summary.json                  # compact aggregate summary
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--trials-per-obj-yaw 2` | 每个物体每个 yaw 跑 2 次；两个 yaw 就是每物体 4 次 |
| `--candidate-gpu-ids 0,1` | 在线 batch candidate generation 使用的 GPU；默认跟随 `--sim-gpu-ids` |
| `--hp-affordance` | 默认 off；开启后用 `output/affordance_hp_v6/.../best_v6_model.pth` 作 affordance（ablation）；否则 `affordance_no_rot_executed/.../best_v6_model.pth` |
| `--affordance-checkpoint PATH` | 显式覆盖 affordance 权重（优先于 `--hp-affordance`） |
| `--candidate-batch-multiplier 2` | 每个 obj×yaw 每轮采样 `2 * trials-per-obj-yaw` 个 PDM pose |
| `--candidate-max-batches 10` | 每个 obj×yaw 最多采样 10 轮，硬门通过不够时用 forced-fill 补齐 |
| `--sim-gpu-ids 0,1` | worker 分配到这些 `CUDA_VISIBLE_DEVICES` |
| `--sim-per-gpu 1` | 每张 GPU 启动几个 worker chunk |
| `--sim-startup-stagger-s 15` | 同一 GPU 上第 2 个及之后 worker 延迟启动，降低 Isaac 冷启动互锁概率 |
| `--resume` | 已存在的 solution JSON 会复用，不重新选 candidate |
| `--dry-run` | 只生成 task/chunk 并打印 worker 命令，不启动 Isaac |
| `--log-only` / `--loud` | worker stdout/stderr 写到 `{result-dir}/logs/worker_*.log` |

物体列表建议用 CSV，便于 spreadsheet 维护和临时开关物体：

```csv
obj_id,enabled,notes
A16013,1,smoke
S10010,1,smoke
C22001,0,disabled example
```

然后：

```bash
python evaluation/eval_pool.py --obj-list evaluation/configs/eval_objects_merged_success_ge30.csv ...
```

`--obj-list` 也兼容 JSON：

```json
{"objects": [{"obj_id": "A16013", "enabled": true}, {"obj_id": "S10010"}]}
```

Pool eval 的 yaw 是外层循环：`--z-yaw-grid 0,90 --trials-per-obj-yaw 2` 会生成并执行 `yaw000 t000,t001`，然后 `yaw090 t000,t001`。`--trials-per-object` 仍可用作兼容 alias，但新脚本建议使用 `--trials-per-obj-yaw`。

在线生成 candidate 时，pool eval 使用独立的 `tools/batch_pdm_candidates.py`，不会改变 `tools/glb_to_pdm_grasp.py`。生成阶段先按 `obj×yaw` 在 GPU 上批量采样 PDM pose，再经过硬门筛选：左右指尖连线与物体 mesh 有接触、目标关键点不低于桌面、hand-up axis 不倒置（可自动翻转修正）。每个 batch 会打 log（PDM 采样开始 / 硬门结束）。每个 `obj×yaw` 写一个 candidate pool HDF5，随后每个 trial 使用其中不同的 candidate index。

录像仍用 `--record-video` / `--record-count-per-object`。含录像 task 的 chunk 会自动用 headed worker；如果服务器没有 `DISPLAY`，`eval_pool.py` 会像 `eval_single.py` 一样自动用 `xvfb-run` 重启自己（可用 `--no-auto-xvfb` 关闭）。

多 worker 并行时，默认只允许一个 worker 录像，避免多个 Isaac/Kit 进程同时抢同一个 viewport/Xvfb 导致视频串帧或异常。需要强制所有 worker 都录像时可显式加 `--record-all-workers`。

多 worker 时，`eval_pool.py` 会为每个 Isaac worker 设置独立的 Kit/Omniverse cache、`HOME`、`TMPDIR`、`OMNI_USER` 和 GPU routing，并默认保存 worker 日志到 `{result-dir}/logs/worker_*.log`。这不会改变 task 内容，只隔离 worker 进程的运行时状态。

## cuRobo 执行阶段（eval 与 collect 相同）

单次 grasp 在 `sim/evaluation/curobo_executor.py` 中分多段 **规划**（不是 pre-grasp 后直线插值）：

| 阶段 | 标签 | 目标 | 碰撞模型 |
|------|------|------|----------|
| 1 | `pre-grasp` | 抓取点沿 approach 后退 **15cm** | 含物体 mesh |
| 1b | `direct`（fallback） | 若 pre-grasp 失败，直接规划到抓取点 | 含物体 mesh |
| 2 | `final` | 精确 TCP 抓取位姿 | **仅桌面+地面**（去掉物体 mesh，允许夹爪进入物体） |
| 3 | `lift` | 抓取点上移 15cm | 仅桌面+地面 |

因此可能出现 `pre-grasp plan OK` 但 `final plan failed`：两段规划起点、目标、碰撞世界不同。详见 executor 源码。

## 输出

默认输出目录是 `output/evaluation/single/`，可用 `--result-dir` 修改。

每次 episode 会写：

```text
output/evaluation/single/          # 或 --result-dir 指定目录
  {episode_id}.json
  episodes.jsonl
  {episode_id}.mp4                 # --record-video
  {episode_id}_robot_gt.hdf5       # --save-hdf5

output/evaluation/batch/           # 批量默认目录
  {episode_id}.json
  episodes.jsonl
  batch_summary.json               # 汇总 returncode / success / video_path
  candidates/{obj_id}/trial_*.hdf5   # 仅 --generate-candidate-each-trial
```

默认 `episode_id`：

```text
{obj_id}_{policy}_yaw{yaw:03d}_t{trial:03d}
```

例如：

```text
C22001_a2g_pdm_yaw000_t002
```

JSON 包含：

- `scene`：USD 路径、object pose、z-yaw、table/robot pose、object scale
- `policy_output`：选中的 candidate、score、gripper width、frame、mesh prerotation
- `execution`：规划阶段状态、executed panda hand pose、gripper tips、初始/最终物体位置
- `success`
- `failure_stage`
- `z_delta_m`
- `video_path`（若 `--record-video`）

常见 `failure_stage`：

- `target_transform`：policy 输出无法转换成当前 executor 支持的 target
- `curobo_init`：cuRobo 初始化失败
- `pregrasp_plan`：pre-grasp 和 direct plan 都失败
- `final_plan`：最后接近阶段规划失败
- `lift_plan`：lift 规划失败
- `lift_result`：执行完成但物体没有被 lift 超过 3cm

## 当前 Policy 接口

当前支持的 policy output 是 `OpenLoopGraspCommand`：

```python
OpenLoopGraspCommand:
  position              # grasp position, 当前 frame=object_mesh
  rotation              # 3x3 rotation, 当前 frame=object_mesh
  gripper_width
  frame                 # 当前支持 "object_mesh"
  ee_frame_convention   # 当前为 "a2g_grasp_frame"
  name
  score
  mesh_prerotation_euler
  metadata
```

当前 executor 只支持：

```text
PolicyOutput.kind == "open_loop_grasp"
command.frame == "object_mesh"
```

## `eval_single` CLI 速查

| 参数 | 默认 | 说明 |
|------|------|------|
| `--obj-id` | （必填） | 物体 ID，用于找 USD |
| `--candidate-hdf5` | 无 | 已有 candidate；与 `--generate-candidate` 二选一 |
| `--generate-candidate` | off | 需 `--mesh` |
| `--candidate-dir` | `output/evaluation/candidates` | 生成 HDF5 的目录 |
| `--candidate-output` | 自动 | 显式指定生成路径 |
| `--candidate-python` | `/home/vision/miniconda3/envs/bundlesdf/bin/python`（存在时） | 跑 glb_to_pdm 的 Python |
| `--selection` | `top` | `top` / `index` / `sample` |
| `--candidate-index` | 0 | `--selection index` 时用 |
| `--policy-seed` | 无 | `sample` 时可复现；省略则每次随机 |
| `--trial` | 0 | episode_id 的 `t{trial:03d}` |
| `--z-yaw-deg` | 0 | Sim 放置 + 生成 candidate 时的 yaw |
| `--object-scale` | 1.0 | |
| `--seed` | `42` | 统一 eval 随机种子 |
| `--random-obj-xy` | off | 在默认物体 XY `(0, 0.55)` 上均匀随机平移 |
| `--obj-xy-jitter-m` | `0.05` | 随机半宽 (m)：`dx,dy ~ U[-jitter,+jitter]` |
| `--headless` | off | 与 `--record-video` 互斥（录像会强制 headed） |
| `--result-dir` | `output/evaluation/single` | |
| `--episode-id` | 自动 | 覆盖默认 id |
| `--save-hdf5` | off | 写 robot_gt 兼容 HDF5 |
| `--record-video` | off | |
| `--record-fps` / `--record-every` | 30 / 3 | |

## 后续扩展接口

### GraspNet / 其他 grasp pose 方法

建议新增：

```text
evaluation/policies/graspnet.py
```

adapter 负责把方法输出统一成 `OpenLoopGraspCommand`。如果方法输出在 camera/world/object frame，需要在 adapter 中显式转换到 `object_mesh` 或在 executor 中新增 frame 支持。

### Diffusion Policy / DP3

DP3 这类 closed-loop policy 不适合只返回一个 grasp pose。建议扩展：

```python
PolicyOutput.kind = "closed_loop_actions"
```

然后新增 executor：

```text
sim/evaluation/closed_loop_executor.py
```

它应负责：

- 每个 sim step 读取 observation
- 构造 policy 输入，例如 point cloud、EE pose、gripper state
- 调 policy server 或本地 policy
- 执行动作序列
- 统一 success 判定和结果写出

现有 `sim/eval_dp3_policy.py` 可作为 DP3 closed-loop adapter 的参考。

### 多物体 / 多 rollout / 并行

**已实现：** `eval_batch.py` 作为 restart-per-episode wrapper；`eval_pool.py` 作为 solution/task queue + 长驻 Isaac worker 路径，worker 内支持同物体 reset 和换物体 swap。

**待做：** GraspNet / DP3 等非 A2G policy adapter，以及更细的 worker 健康检查和失败任务自动重试。

## 已知日志噪音

第一次 setup scene 时可能看到一些类似：

```text
/World/Table does not exist
/World/Rigid/rigid_0 does not exist
```

这是清理旧 prim 时删除了不存在的 prim，通常不影响结果。后续可以用 `safe_delete_prim()` 降低日志噪音。

Isaac Sim 还可能输出 GPU dynamics/CCD、Franka mimic joint、mesh normal、TGS velocity iterations 等 warning。只要 episode JSON 中 `success=true` 且 `z_delta_m > 0.03`，当前 evaluation 结果可视为有效。

