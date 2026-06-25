# Razor 端脚本 — 部署说明

这些文件需要拷贝到 Razor 笔记本的 `V2AP-demo/demo/phase2/` 目录下。

## 拷贝方法

### 方法 A: USB
```bash
# 在 5090 上
cp -r ~/Project/V2AP/s2r/razor_scripts/phase2/* /media/lyh/USB/razor_phase2/

# 在 Razor 上
cp -r /media/usb/razor_phase2/* ~/V2AP-demo/demo/phase2/
```

### 方法 B: rsync
```bash
# 在 5090 上
rsync -avz ~/Project/V2AP/s2r/razor_scripts/phase2/ \
    razor@<RAZOR_IP>:~/V2AP-demo/demo/phase2/
```

### 方法 C: GitHub (推荐长期)
```bash
# 提交到 V2AP-demo 仓库
cd ~/V2AP-demo
cp ~/从5090拷来的/phase2/* demo/phase2/
git add demo/phase2/run_auto_grasp.py demo/phase2/retarget.py demo/phase2/calib/
git commit -m "Add Phase 2 auto grasp from candidates.json"
git push
```

## 文件清单

| 文件 | 目标位置 | 用途 |
|------|----------|------|
| `run_auto_grasp.py` | `demo/phase2/run_auto_grasp.py` | 自动抓取入口 |
| `retarget.py` | `demo/phase2/retarget.py` | 坐标变换 (mesh→base→ee) |
| `calib/ee_retarget.yaml` | `demo/phase2/calib/ee_retarget.yaml` | T_ee_pinch 标定 (**需在真机上标定!**) |

## 依赖 (V2AP-demo 已有)

- `demo/phase1/executor.py` — GraspExecutor (OMPL 规划 + 执行)
- `demo/phase1/grasp_geometry.py` — pre-grasp / lift 几何
- `demo/phase1/config_io.py` — YAML 配置读写
- `demo/hardware.py` — 硬件连接
- `demo/hand_close.py` — 手部闭合控制

## 使用

```bash
# 在 Razor 上
cd ~/V2AP-demo
source setup_local.sh

# 干跑 (不连硬件, 检查坐标转换)
python demo/phase2/run_auto_grasp.py --session 20260603_143022_chips --dry-run

# 真机执行
python demo/phase2/run_auto_grasp.py --session 20260603_143022_chips
```
