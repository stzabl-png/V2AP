#!/bin/bash
# HaWoR 实时监控 — 进度条 + 当前序列 + GPU + 最新结果
LOG=/tmp/hawor_parallel.log
MANO=/home/lyh/Project/Affordance2Grasp/data_hub/ProcessedData/mano/Egocentric/Egodex/egodex
TOTAL=3051

bar() {
    local done=$1 total=$2 width=40
    local pct=$(( done * 100 / total ))
    local filled=$(( done * width / total ))
    local empty=$(( width - filled ))
    printf "["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %3d%%  %d / %d" "$pct" "$done" "$total"
}

watch -n 3 '
LOG=/tmp/hawor_parallel.log
MANO=/home/lyh/Project/Affordance2Grasp/data_hub/ProcessedData/mano/Egocentric/Egodex/egodex
TOTAL=3051

# ── counts ──────────────────────────────────────────────────────
DONE=$(find -L "$MANO" -name "*.npz" 2>/dev/null | wc -l)
DONE_LOG=$(grep -c "^  ✅" "$LOG" 2>/dev/null || echo 0)
FAIL_LOG=$(grep -c "^  ❌" "$LOG" 2>/dev/null || echo 0)
SKIP_LOG=$(grep -c "^  ⏭" "$LOG" 2>/dev/null || echo 0)

# ── progress bar ─────────────────────────────────────────────────
PCT=$(( DONE * 100 / TOTAL ))
FILLED=$(( DONE * 50 / TOTAL ))
EMPTY=$(( 50 - FILLED ))
BAR=$(printf "%${FILLED}s" | tr " " "█")$(printf "%${EMPTY}s" | tr " " "░")

echo "╔══════════════════════════════════════════════════════════╗"
printf "║  HaWoR  %s  %s  ║\n" "$(date +%H:%M:%S)" "workers=4"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  [%s]  %3d%%  %d / %d   ║\n" "$BAR" "$PCT" "$DONE" "$TOTAL"
printf "║  ✅ 本次新增: %-4s  ❌ 失败: %-4s  ⏭ 跳过: %-8s  ║\n" "$DONE_LOG" "$FAIL_LOG" "$SKIP_LOG"
echo "╠══════════════════════════════════════════════════════════╣"

# ── 当前正在处理 ──────────────────────────────────────────────────
echo "║  ▶ 正在处理:                                             ║"
PROCS=$(ps aux | grep "[r]un_hawor_seq" | grep -oP "egodex/test/\K\S+(?=\.mp4)" | sort -u)
if [ -z "$PROCS" ]; then
    echo "║    （无活跃序列，可能在 skip 或启动中...）                 ║"
else
    while IFS= read -r line; do
        printf "║    %-54s  ║\n" "$line"
    done <<< "$PROCS"
fi
echo "╠══════════════════════════════════════════════════════════╣"

# ── GPU ──────────────────────────────────────────────────────────
echo "║  ▶ GPU (RTX 5090):                                       ║"
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null)
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -v "^$")
printf "║    整体: %-48s  ║\n" "$GPU_UTIL"
N_GPU=$(echo "$GPU_PROCS" | grep -c "[0-9]" || echo 0)
printf "║    compute 进程数: %-38s  ║\n" "$N_GPU 个"
echo "╠══════════════════════════════════════════════════════════╣"

# ── 最近完成/失败（最新8条）─────────────────────────────────────
echo "║  ▶ 最新结果:                                             ║"
RECENT=$(grep -E "^  ✅|^  ❌" "$LOG" 2>/dev/null | tail -8)
if [ -z "$RECENT" ]; then
    echo "║    （暂无结果）                                           ║"
else
    while IFS= read -r line; do
        # 截断超长行
        short="${line:0:56}"
        printf "║  %-56s  ║\n" "$short"
    done <<< "$RECENT"
fi
echo "╚══════════════════════════════════════════════════════════╝"
'
