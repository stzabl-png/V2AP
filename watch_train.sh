#!/bin/bash
# 实时监控训练进度
# 用法: bash watch_train.sh

CKPT=/home/lyh/Project/Affordance2Grasp/output/checkpoints_new
LOG=/home/lyh/Project/Affordance2Grasp/output/train_new.log

while true; do
    printf "\033c"
    echo "══════════════════════════════════════════════════════════"
    echo "  🧠  Affordance 训练监控  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "══════════════════════════════════════════════════════════"

    # 进程状态
    PID=$(pgrep -f "model.train" | head -1)
    GEN_PID=$(pgrep -f "gen_m5_training" | head -1)
    if [ -n "$GEN_PID" ]; then
        echo "  ⏳ 数据生成中  PID=$GEN_PID"
    elif [ -n "$PID" ]; then
        echo "  ✅ 训练运行中  PID=$PID"
    else
        echo "  ❌ 无训练进程"
    fi

    # GPU
    echo ""
    echo "── GPU ────────────────────────────────────────────────────"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader | awk -F', ' '{
        printf "  GPU%s | 利用率: %s | 显存: %s/%s | 温度: %s°C\n", $1, $2, $3, $4, $5}'

    # 训练进度（从 training_history.json）
    echo ""
    echo "── 训练进度 ────────────────────────────────────────────────"
    if [ -f "$CKPT/training_history.json" ]; then
        python3 - "$CKPT/training_history.json" << 'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
epochs = data if isinstance(data, list) else data.get('epochs', [data])
if not epochs: print("  (暂无数据)"); exit()
last = epochs[-1]
total = last.get('epoch', len(epochs))
best_f1 = max((e.get('val_f1', 0) for e in epochs), default=0)
best_ep = max(epochs, key=lambda e: e.get('val_f1', 0)).get('epoch', '?')
print(f"  当前 Epoch:   {total}")
print(f"  Train Loss:   {last.get('train_loss', '?'):.4f}  (seg={last.get('train_seg','?'):.4f})")
print(f"  Val Loss:     {last.get('val_loss', '?'):.4f}")
print(f"  Val F1:       {last.get('val_f1', '?'):.4f}   IoU={last.get('val_iou','?'):.4f}")
print(f"  Val FC (mm):  {last.get('val_fc_mm', '?'):.1f} mm")
print(f"  最佳 F1:      {best_f1:.4f}  @ Epoch {best_ep}")
print(f"  学习率:       {last.get('lr', '?')}")
PYEOF
    else
        echo "  (training_history.json 尚未生成)"
    fi

    # 检查点
    echo ""
    echo "── 已保存检查点 ────────────────────────────────────────────"
    if [ -d "$CKPT" ]; then
        ls "$CKPT"/*.pth 2>/dev/null | wc -l | xargs printf "  共 %s 个 .pth\n"
        ls -t "$CKPT"/*.pth 2>/dev/null | head -3 | while read f; do
            printf "  %s  (%s)\n" "$(basename $f)" "$(du -sh $f | cut -f1)"
        done
    fi

    # 最新日志
    echo ""
    echo "── 最新日志（后15行）──────────────────────────────────────"
    if [ -f "$LOG" ]; then
        tail -15 "$LOG" | sed 's/^/  /'
    else
        echo "  (日志未找到: $LOG)"
    fi

    echo ""
    echo "  [每10秒刷新 | Ctrl+C 退出]"
    sleep 10
done
