# Sim-to-Real 实施计划 / 操作指南 —— V2AP 主方法部署到 Dexmate Vega + SharpaWave

> 状态：**讨论稿 v0**（2026-05-12）。先把方案、取舍、需要新写的脚本、分期都列清楚，定下来再动手写代码。

---

## 0. 前提事实（先对齐认知）

### 0.1 我们训练出来的网络（主方法）吃什么、吐什么

- **网络**：PointNet++ 分割网络（`model/pointnet2.py` 的 `PointNet2Seg`；推理入口 `inference/predictor.py`，`predictor.predict_from_points(points, normals)` 已经存在，可以直接喂点云数组、不必给 mesh 路径）。
- **输入**：一个**物体表面的点云** —— `xyz (N,3)` + 每点法向 `normals (N,3)`（= 6 通道；"M5" 变体多塞一个事先算好的 `human_prior (N,1)` 当第 7 通道）。N = 1024 或 4096。训练时这些点是从**完整 mesh 表面采样**的（`trimesh.sample.sample_surface(SAM3D_mesh, N)`，应用 scale.json 缩放到真实尺寸，**未做随机旋转/平移增广，只有 ±2mm jitter + 点丢弃**）。
  - 名词："点云" = 一堆 3D 点的集合，描述物体表面几何；每点配一个法向（该点处表面朝向）。
- **输出**：每点一个**接触概率** `(N,)`（sigmoid，[0,1]，"这个点是不是抓取时手会碰到的地方"）+ 可选一个 `force_center (3,)`（高接触区质心）。
- **下游**（`inference/grasp_pose.py` + `tools/random_grasp_sampler.py`）：接触概率 → 在高概率区附近**采抓取候选**（每个候选 = 抓取点 `grasp_point` + 接近方向 `approach` + 夹爪开合宽度 `width` + 旋转矩阵 `rotation`），用对踵性/局部平整度/离质心/工作空间打分 → 选 top-N。原 pipeline 还把候选丢进 Isaac Sim + cuRobo（Franka）做物理校验再执行；**部署时跳过 Isaac Sim 校验，直接拿最高分且可达的候选 → 规划 → 执行**。
- ⚠️ **重要**：当前部署链路 `run.py --mesh object.obj` 假定你**已经有这个物体的一份完整 mesh**。它不吃视频、不吃 RGB 图、不吃原始深度图。sim-to-real 的核心工作就是补上"传感器 → 网络要的那种点云"这一段。

### 0.2 机器人：Dexmate Vega（[humanoid.guide](https://humanoid.guide/product/vega/) / [dexmate.ai](https://www.dexmate.ai/product/vega)）

- 全向移动底盘（4 km/h）+ 可折叠躯干（最高伸到 2.2m）+ **2 × 7-DOF 手臂**（每臂负载 15kg），原配是五指手。
- **感知**：内置 **RGB-D 相机**（头部；官方没公开型号/分辨率，按惯例多半是 RealSense/Orbbec 一类）、LiDAR、IMU、超声、力/力矩传感。
  - 名词："RGB-D 相机" 给你每帧两张对齐的图：① 彩色图，② 深度图（每像素一个数 = 该像素对应表面离相机多远，米）。点云不是相机直接给的，是用深度图 + 相机内参（K：焦距 fx/fy、主点 cx/cy）"反投影"算出来的。
- **软件**：Linux + Dexmate ROS 兼容 SDK + Python API + 提供 URDF/USD（给主流仿真器用）；Intel x86 + Jetson Orin 算力；Bluetooth/Ethernet/USB/WiFi/DisplayPort + 5V/12V 自定义负载供电口。

### 0.3 末端执行器：SharpaWave（[sharpa.com/pages/wave](https://www.sharpa.com/pages/wave) / [humanoidsdaily](https://www.humanoidsdaily.com/news/sharpa-robotics-begins-shipping-sharpawave-dexterous-hand)）

- **22 主动 DOF**，五指、1:1 人手尺寸（约 200×90×50mm，~1.2kg），单指尖力 >20N，开合速度 >4Hz，全关节可反驱。
- **触觉**：每指尖一个微型相机 + >1000 触觉像素，0.005N 精度，180FPS，<1mm 空间分辨率，6 维力检测 —— 这是它最大的卖点，sim-to-real 里可以拿来做**抓取成功/打滑检测 + 力闭合控制**。
- 开发支持：SharpaPilot app + 跨平台开发套件 + 高保真仿真模型（具体 ROS/API/控制模式/挂载法兰规格官网没写全，要找 Sharpa 要文档：sales@sharpa.com）。
- **我们的用法**：把 SharpaWave **当二指夹爪用** —— 只指挥"拇指" + "食指（或食+中并拢当一个'虚拟指')" 做对捏，忽略其余灵巧度。理由：主方法的接触预测 + 抓取采样本来就是按平行二指夹爪设计的（Baseline2、Mano2Gripper sim 场景也都是 Franka 二指 8cm）。代价：浪费了 SharpaWave 的多指能力 —— 留作以后的改进项（多指 grasp 合成）。

---

## 1. 总览：sim-to-real 要搭三座桥 + 一堆配套

```
Vega 头部 RGB-D
   │  ┌─────────────────────────── 桥 A：感知 ───────────────────────────┐
   ├──┤ 检测+分割物体 → (可选)多视角融合 → 物体点云(4096,3)+法向，机器人 base 系 │
   │  └────────────────────────────────────────────────────────────────┘
   ▼
PointNet++（主方法网络）  →  每点接触概率
   │  ┌──────────────── 桥 B：网络输出 → 可执行抓取 ────────────────┐
   ├──┤ 接触概率 → 采抓取候选(grasp_point/approach/width) →          │
   │  │ retarget 成 SharpaWave-当二指夹爪 的关节配置 + 手臂腕部 6D 位姿 │
   │  └────────────────────────────────────────────────────────────┘
   ▼
   │  ┌──────────────── 桥 C：Vega 上规划+执行 ────────────────┐
   └──┤ (定位底盘/躯干) → cuRobo-用-Vega-URDF 或 Vega SDK 规划 → │
      │ home→pre-grasp→grasp→闭手→抬起，触觉/力反馈判成败→重试    │
      └──────────────────────────────────────────────────────┘
```

配套（贯穿三座桥）：相机↔base 外参标定、SharpaWave↔臂法兰变换、物体检测/分割模型、坐标系一致性、6通道 vs 7通道网络的选择、抓取成功检测与重试。

---

## 2. 桥 A —— 怎么拿到"物体点云"（你最关心的那个问题）

**核心矛盾**：网络是在**完整、干净、从 mesh 采样**的点云上训练的；真实 RGB-D 给你的是**部分（只看得到朝向相机那一面）、带噪、相机系**的点云。要么把传感器数据"补"成训练那种，要么把网络"改"成能吃传感器那种。下面是几条路，按推荐度排：

### 方案 A1（推荐做主路）：单视角 → 部分点云 → **在部分点云上 finetune 网络**

- **思路**：部署时其实你**只需要可见那一面的接触点**（你也只能从可见那面去抓），完整点云是个**训练时的便利，不是部署时的必需**。所以：用主方法的训练 mesh 从随机视角渲出"可见面部分点云"+ 加传感器噪声 + 加 SE(3) 随机旋转/平移增广 → finetune PointNet++ → 得到一个"部署版" checkpoint。部署时：单帧 RGB-D → 分割出物体 → 反投影出部分点云 → 减质心居中 → 喂网络 → 可见面上的接触概率 → 采抓取。
- **优点**：闭掉了"完整 vs 部分""mesh 干净 vs 传感器噪""固定朝向 vs 任意朝向"三个域差，是**最干净的长期方案**；部署链路最短（一帧就够）。
- **缺点**：要做一轮 finetune（不贵，几小时；用已有的 SAM3D mesh 当渲染源）；下游抓取采样器里有些打分项（离物体质心、对踵性）在只有部分点云时会偏向可见面 —— 影响可接受，必要时配合方案 A2 的部分融合。
- **新脚本**：`s2r/finetune_affordance_on_partial.py`（渲染部分点云 + 增广 + 训练）；改一下 `model/train.py` 的数据增广（加 SE(3) + 渲染噪声）。

### 方案 A2（推荐做"完整点云"那一档，或当 A1 的补充）：机器人**多视角融合**（自主，不需要人手）

- **思路**：Vega 碰到一个物体时：① 头部 RGB-D 里检测+分割出它；② 机器人自己**移动头/底盘**采 3~6 帧不同角度的 RGB-D（约 30~60° 一档）；③ 用机器人运动学（臂 FK）+ 底盘里程计知道每帧相机位姿 → 把每帧 mask 内的物体点都变换到一个共同坐标系拼起来 → 体素降采样、去噪、降到 4096 → 得到**接近完整**（顶+侧；桌面挡住的底面拿不到，通常无所谓）的点云。④ 减质心居中 → 喂网络。
- **优点**：自主（不需要人），对新物体也能用；完整点云让抓取采样器的 COM/对踵性打分更准。
- **缺点**：物体要能从多个角度看到；依赖准确的相机位姿（Vega 的臂 FK + 底盘里程计够用）；每物体多花 ~10-30s。
- **新脚本**：`s2r/perceive_object.py` 里实现"采多帧 + 用位姿融合"分支。

### 方案 A3：物体有 CAD 模型就直接查表（最干净，但只对已知物体）

- 如果是已知的 YCB 类物体（有 CAD 模型）：估它的 6D 位姿（"物体位姿" = 把物体模型坐标里的点变到相机坐标的那个旋转+平移；用 FoundationPose 配该物体的 CAD mesh，或别的位姿估计器）→ 直接从 CAD mesh 在它的 canonical 坐标系采 4096 点。零域差、最准。**建议作为"物体库"里已知物体的快路径**，未知物体走 A1/A2。

### 方案 A4（你提到的）：人手转动物体 + 录视频 → 离线重建 → 用

- 一个人拿着物体在相机前转一圈、录一段（RGB-D 的话直接 TSDF 融合；纯 RGB 的话 SfM / 单图转 3D 模型如 InstantMesh）→ 得到一份 mesh/点云 → 注册后当 CAD 用（走 A3 的路）。
- **优点**：能拿到包括底面的全表面；一次性，每物体做一次就行。
- **缺点**：要人在环（不自主）；部署时该物体当下的位姿仍要单独估计。
- **定位**：适合"一次性建一个物体库"或"机器人怎么都拿不到好视角的物体"的补充手段，不建议当主路（主路应该是自主的 A1/A2）。

### 方案 A5（不推荐）：单视角部分点云直接喂现有网络（不 finetune）

- 能跑，但和训练分布对不上（完整 vs 部分、干净 vs 噪、固定朝向 vs 任意朝向）→ 效果打折，不稳定。只适合最快速地验证整条链路通不通，不当正式方案。

### 我的推荐组合

**主路 = A1（在部分点云上 finetune 网络）+ 单帧 RGB-D 部署**；**已知物体走 A3（CAD + 位姿估计）快路径**；**A2（机器人多视角融合）作为"想要完整点云、不想 finetune" 的备选或 A1 的增强**。先用 **A5（不 finetune，直接喂）跑通整条链路**验证管道，确认下游 OK 后立刻上 A1。

### 关于"坐标系/朝向"的关键点（必须处理）

网络训练时点云没做随机旋转增广，所以它对朝向有依赖。部署时物体朝向是相机随便看到的。两个办法：
1. **首选**：在 A1 的 finetune 里加 **SE(3) 随机增广** → 网络变成朝向无关 → 部署时减质心居中、朝向随便都行。
2. 不 finetune 的话：至少做**减质心居中**（PointNet++ 的局部特征对朝向有一定容忍，靠局部几何"认出这是个把手"，不太依赖全局朝向 —— 但不保证）。

---

## 3. 桥 B —— 网络输出 → SharpaWave（当二指夹爪）能执行的抓取

### 3.1 在 SharpaWave 上定义一个"虚拟二指夹爪"

- **finger 1 = 拇指尖**；**finger 2 = 食指尖**（或食+中并拢当一个"虚拟指"，接触面更大、更稳）。
- "夹爪开合宽度 `width`" = 这两个有效接触点之间的距离（SharpaWave 是人手尺寸，拇-食对捏的有效行程大概几 cm 到 ~8-10cm，需要实测；这正好对上 Baseline2 / Mano2Gripper 那套 8cm 的设定）。
- "接近轴 `approach`" = 手掌正对的方向；"闭合轴" = 拇-食连线（垂直于 approach）。
- **EE/腕部 6D 位姿**（给手臂规划用）= 由抓取候选的 (grasp_point, approach, 闭合轴) + 一个固定的"手掌相对臂法兰"偏移 推出来 —— 这个偏移需要 SharpaWave 的挂载帧相对 Vega 臂法兰的变换（向 Sharpa / Dexmate 要，或标定）。

### 3.2 抓取候选 → SharpaWave 关节配置 的 retarget 脚本

- 输入：一个抓取候选（grasp_point 在物体表面、approach 方向、width）。
- 输出：① 手臂腕部 6D 目标位姿（让"虚拟二指"的中点落在 grasp_point、闭合轴垂直于 approach、手掌沿 approach 方向）；② SharpaWave 的关节目标（先张到比 width 略大，到位后闭到比 width 略小施加预紧力 —— 用指尖触觉的力读数停在目标力上，比纯位置控制稳得多）。
- 这本质就是我给 Baseline1 写的 "MANO→Franka 二指" retarget 的镜像版，只是目标换成 "SharpaWave 当二指"。
- **新脚本**：`s2r/grasp_from_contacts.py`（接触概率 → 采候选[复用 `inference/grasp_pose.py` + `tools/random_grasp_sampler.py`]→ retarget 成 SharpaWave 配置 + 腕部位姿）。

### 3.3 一个建议：先在仿真里验证桥 B

Dexmate 提供 Vega 的 URDF/USD，Sharpa 提供 SharpaWave 的高保真仿真模型 —— 先在 Isaac Sim 里把 Vega + SharpaWave 拼起来，回放几个 retarget 出的抓取看手/臂位姿对不对、会不会自碰，再上真机。

---

## 4. 桥 C —— Vega 上的运动规划与执行（替掉 cuRobo-for-Franka）

- **手臂规划**：把原 pipeline 里的 "cuRobo-for-Franka" 换成 (a) cuRobo **重新配置成 Vega 的 7-DOF 臂**（cuRobo 是机器人无关的，加载 Vega 的臂 URDF + 碰撞世界即可 —— 这条最省事，因为 pipeline 已经会说 cuRobo），或 (b) 直接用 Vega SDK 自带的规划器（如果有）。规划序列：home → pre-grasp（沿 approach 往后退 ~15cm）→ grasp → 闭手 → 抬起 ~15cm。
- **底盘 + 躯干**：桌面抓取要先把全向底盘开到合适站位、把可折叠躯干摆到合适高度，再做臂动作 —— 这部分用 Vega SDK。
- **跳过 Isaac Sim 物理校验**：原 pipeline 用 Isaac Sim 在执行前筛候选 —— 真机上不需要，直接试最高分且可达的候选；失败了（触觉/F-T 告诉你抓空或打滑、或物体没被抬起）就退而试下一个候选。
- **新脚本**：`s2r/plan_and_execute.py`（给定选中的抓取[腕部位姿 + 手配置]：摆底盘/躯干 → 规划臂动作 → 执行 home→pre-grasp→grasp→闭手→抬起 → 触觉/F-T 判成败 → 重试）。

---

## 5. 配套（贯穿三座桥的杂项，都得有人处理）

| 项 | 说明 | 谁提供 / 怎么搞 |
|---|---|---|
| **相机↔base 外参** | 头部 RGB-D 相机相对机器人 base 的位姿（手眼标定）—— 把物体点云放进机器人规划坐标系必需 | Vega 内置相机，Dexmate SDK 应给；不给就手眼标定（棋盘格 / AprilTag） |
| **相机内参 K** | 反投影深度图必需 | Vega RGB-D 驱动里读 |
| **SharpaWave↔臂法兰变换** | 桥 B 算腕部位姿必需 | 向 Sharpa / Dexmate 要挂载帧规格，或标定 |
| **物体检测/分割** | 这个项目目前**没有** | 用开放词表检测器 + SAM2 出 mask；demo 阶段可以人点框给 SAM2 |
| **坐标系一致性** | 网络的点云系 ↔ 抓取位姿系 ↔ 臂规划系 —— 规划那一刻必须都在机器人 base 系 | 在 `perceive_object.py` 里把点云一步到位变到 base 系 |
| **6通道 vs 7通道网络** | "M5"/7通道变体要每物体一份事先算好的 `human_prior` → 对新物体是循环依赖 | **部署用 6通道（只 xyz+normals）变体** |
| **抓取成功检测 + 重试** | SharpaWave 指尖触觉(1000px) + Vega F/T → 判抓住/打滑；物体被抬起 = 成功 | `plan_and_execute.py` 里做闭环 |

---

## 6. 需要新写的衔接脚本清单（放 `s2r/` 下）

1. **`s2r/perceive_object.py`** —— 头部 RGB-D（+ 可选多视角）→ 检测+分割物体 → 反投影/融合 → 体素降采样到 4096 + 估法向 → 减质心居中 → 输出物体点云（在机器人 base 系）+ 质心/PCA 帧。
2. **`s2r/predict_affordance.py`** —— 包一层 `inference/predictor.AffordancePredictor`，直接吃点云数组（`predict_from_points` 已存在），返回接触概率 + force_center；处理居中/通道选择（默认 6 通道）。
3. **`s2r/grasp_from_contacts.py`** —— 接触概率 + 物体点云 → 采抓取候选（复用 `inference/grasp_pose.py` + `tools/random_grasp_sampler.py`）→ 每个候选 retarget 成 SharpaWave-当二指 的关节配置 + 手臂腕部 6D 位姿 → 按分数 + 可达性排序。
4. **`s2r/plan_and_execute.py`** —— 给定选中抓取：摆底盘/躯干 → cuRobo-用-Vega-URDF（或 Vega SDK）规划臂动作 → 执行 home→pre-grasp→grasp→闭手→抬起 → 触觉/F-T 判成败 → 失败重试下一候选。
5. **`s2r/calib/`** —— 相机↔base 外参、SharpaWave↔臂法兰变换、（如需）相机内参标定脚本（一次性）。
6. **`s2r/finetune_affordance_on_partial.py`**（强烈建议）—— 用 SAM3D mesh 从随机视角渲部分点云 + SE(3) 增广 + 传感器噪声 → finetune PointNet++ → 部署版 checkpoint。配套改 `model/train.py` 的增广。
7. **`s2r/run_grasp.py`** —— 编排 1→2→3→4 的总入口（真机版的 `run.py`）。
8. **`s2r/BRINGUP.md`** —— Vega/SharpaWave 上线文档：装 SDK、拿 URDF、cuRobo 配 Vega 臂、验证 RGB-D 流、SharpaPilot/SDK 控手、各项标定。

---

## 7. 实施分期 + 每期验收标准

| Phase | 内容 | 验收标准 |
|---|---|---|
| **0 · 上线** | 装 Dexmate SDK + Sharpa 开发套件；拿 Vega/SharpaWave 的 URDF/USD；cuRobo 配 Vega 臂跑通空规划；RGB-D 流能读；SharpaPilot/SDK 能控手开合；做相机↔base、SharpaWave↔法兰的标定 | 能在 Python 里：读到一帧 RGB-D + K、命令臂走到一个目标位姿、命令 SharpaWave 张开/闭合到指定开度 |
| **1 · 感知独立跑通** | 实现 `perceive_object.py`（先单视角，再加多视角融合）；可视化输出的物体点云 | 把一个桌面物体（杯子）的点云可视化出来，形状/尺寸/位置（在 base 系）肉眼看着对 |
| **2 · 真实点云上的接触预测** | `predict_affordance.py`；把真实物体点云喂网络，可视化预测的接触概率热图；若热图明显不对 → 跑 `finetune_affordance_on_partial.py` 出部署版 checkpoint，再验 | 杯子/盒子/罐子上的接触热图落在"人会去抓的地方"（杯把/侧壁/边缘） |
| **3 · 抓取生成 + 仿真验证** | `grasp_from_contacts.py`；在 Isaac Sim 里加载 Vega + SharpaWave，回放生成的抓取（腕部位姿 + 手配置） | 仿真里手/臂位姿对得上接触点、不自碰；二指开度合理 |
| **4 · 真机闭环（已知物体）** | `plan_and_execute.py` + `run_grasp.py`；在 3-5 个已知物体（杯、盒、罐、香蕉、钻……）上跑全链路，带触觉/F-T 成败检测 + 重试 | 已知物体抓取成功率 ≥ 某阈值（先定个 50-70% 的及格线）；失败能自动重试 |
| **5 · 泛化测试** | 换没见过的新物体跑全链路 | 新物体上仍有可观成功率（这才是主方法 vs baseline 的卖点要兑现的地方） |

---

## 8. 已知风险与取舍

- **域差（最大风险）**：mesh 完整点云 → 真实部分/带噪点云。缓解 = Phase 2 的 finetune-on-partial。**强烈建议不要跳过**。
- **朝向依赖**：网络没做旋转增广。缓解 = finetune 时加 SE(3) 增广；不 finetune 就只能靠局部特征的容忍 + 减质心居中，不稳。
- **SharpaWave 当二指 = 浪费灵巧度**：先这么做（管道最短、对得上现有 grasp 采样器）；以后可以上"多指 grasp 合成"用满它的 22 DOF + 触觉。
- **抓取采样器的打分项偏向可见面**（只有部分点云时 COM/对踵性会偏）：用方案 A2 多视角融合补，或接受偏差（demo 够用）。
- **未知物体的 6D 位姿估计**：A3 快路径只对有 CAD 的物体；新物体走 A1（不需要位姿，网络直接吃部分点云）。
- **跨机协调**：Vega SDK、SharpaWave SDK 的具体接口/控制模式官网没写全 —— Phase 0 第一件事就是把这俩 SDK 的文档要齐、跑通最小例子，否则后面全是空中楼阁。
- **跟 sim/cuRobo 的现有代码**：现有 `sim/`、`tools/random_grasp_sampler.py`、`inference/grasp_pose.py` 全是 Franka 二指 8cm 配的 —— 复用它们的"采候选 + 打分"逻辑（这部分和夹爪种类无关），换掉"Franka 的 cuRobo 配置"和"Franka 的 gripper 几何"。

---

## 9. 下一步（讨论项）

1. 你认可"主路 = A1（finetune-on-partial）+ 单帧部署，已知物体走 A3，A2 多视角融合作备选"这个组合吗？还是更想先用人手转动录视频（A4）建物体库？
2. SharpaWave / Vega 的 SDK 文档你那边能拿到吗？（Phase 0 的前置）
3. Phase 顺序、验收阈值有要调的吗？
4. 定下来后，我先写哪个脚本？（建议从 `s2r/perceive_object.py` + `s2r/predict_affordance.py` 起 —— 这两个能最快让你在真机上看到"物体点云 + 接触热图"，是整条链路的地基）
