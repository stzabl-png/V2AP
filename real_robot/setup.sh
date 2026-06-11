#!/bin/bash
# V2AP Real Robot Setup
# Copy this file to setup_local.sh (gitignored) and fill in your robot details.
#
# conda activate <your-env-name>

# Required: set these to match your robot
export ROBOT_NAME=dm/<your-robot-id>    # e.g. dm/vgd1262ab823-1p
export ROBOT_IP=<your-robot-ip>         # e.g. 192.168.50.20

# Optional: camera IP (for Phase 2 ZED stream)
# export CAMERA_IP=<your-zed-host-ip>

# Kill any stale processes on tactile ports
for port in 50001 50002; do
    pids=$(sudo lsof -ti :"$port" 2>/dev/null | sort -u)
    if [ -n "$pids" ]; then
        echo "Killing PID(s) on port $port: $pids"
        sudo kill -9 $pids
    fi
done
unset port pids
