#!/bin/bash
# 实时监控 batch_grasp_collect 运行状态
# 用法: bash watch_sim.sh

LOG=$(ls /home/lyh/Project/Affordance2Grasp/output/grasp_collect/batch_run_*.log 2>/dev/null | tail -1)
SUMMARY=/home/lyh/Project/Affordance2Grasp/output/grasp_collect/summary.csv
OUT=/home/lyh/Project/Affordance2Grasp/output/grasp_collect

clear_screen() { printf "\033c"; }

while true; do
    clear_screen

    echo "═══════════════════════════════════════════════════════════"
    echo "  🤖  OakInk Grasp Sim 监控  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════"

    # 进程状态
    PID=$(pgrep -f "batch_grasp_collect" | head -1)
    if [ -n "$PID" ]; then
        echo "  ✅ 进程运行中  PID=$PID"
    else
        echo "  ❌ 进程已停止!"
    fi

    # GPU 状态
    echo ""
    echo "── GPU ────────────────────────────────────────────────────"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader | awk -F', ' '{
        printf "  GPU%s | 利用率: %s | 显存: %s/%s | 温度: %s°C\n", $1, $3, $4, $5, $6}'

    # summary 统计
    echo ""
    echo "── 进度统计 ────────────────────────────────────────────────"
    if [ -f "$SUMMARY" ]; then
        TOTAL=$(tail -n +2 "$SUMMARY" | wc -l)
        OK=$(tail -n +2 "$SUMMARY" | awk -F',' '$NF=="ok"' | wc -l)
        GEN_OK=$(tail -n +2 "$SUMMARY" | awk -F',' '{print $(NF-1)}' | grep -c "gen_ok\|gen_skip")
        TIMEOUT=$(tail -n +2 "$SUMMARY" | grep -c "sim_timeout")
        FAILED=$(tail -n +2 "$SUMMARY" | grep -c "sim_failed\|all_failed")
        SKIP=$(tail -n +2 "$SUMMARY" | grep -c "no_candidates")
        printf "  ✅ 抓取成功:   %d 条记录\n" "$OK"
        printf "  🔄 候选已生成: %d 条记录\n" "$GEN_OK"
        printf "  ⏱️  Sim超时:   %d\n" "$TIMEOUT"
        printf "  ❌ Sim失败:   %d\n" "$FAILED"
        printf "  ⬛ 无候选:    %d\n" "$SKIP"
        printf "  📋 总记录数:  %d\n" "$TOTAL"
    else
        echo "  (summary.csv 尚未生成)"
    fi

    # 已完成物体数
    echo ""
    echo "── 输出文件 ────────────────────────────────────────────────"
    MERGED=$(ls $OUT/merged/*.hdf5 2>/dev/null | wc -l)
    CANDS=$(ls $OUT/candidates/round_*/*.hdf5 2>/dev/null | wc -l)
    GT=$(ls $OUT/robot_gt/round_*/*.hdf5 2>/dev/null | wc -l)
    printf "  📦 候选 HDF5:    %d 个\n" "$CANDS"
    printf "  🎯 GT HDF5:      %d 个\n" "$GT"
    printf "  🗂️  已Merge物体: %d / 100\n" "$MERGED"

    # 最新日志
    echo ""
    echo "── 最新日志（后20行）──────────────────────────────────────"
    if [ -f "$LOG" ]; then
        tail -20 "$LOG" | sed 's/^/  /'
    else
        echo "  (日志文件未找到: $LOG)"
    fi

    echo ""
    echo "  [每5秒刷新 | Ctrl+C 退出]"
    sleep 5
done
