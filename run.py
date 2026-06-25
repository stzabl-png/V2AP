#!/usr/bin/env python3
"""
V2AP — Unified Entry Point
========================================
Single command for: object mesh → affordance inference → grasp pose → Isaac Sim execution

Usage:
    # Inference + simulation (most common)
    python run.py --mesh /path/to/object.obj

    # Inference only (no simulation)
    python run.py --mesh /path/to/object.obj --no-sim

    # Prepare training data from scratch
    python run.py --prepare

    # Train the affordance model
    python run.py --train --epochs 150

    # Execute a pre-generated grasp HDF5 in simulation
    python run.py --execute output/grasps/A16013_grasp.hdf5
"""

import os
import sys
import argparse
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
import config


def cmd_prepare(args):
    """Stage 1-2: Extract contacts + build training dataset."""
    print("=" * 60)
    print("Stage 1: Extract contact points")
    print("=" * 60)
    cmd = [sys.executable, "-m", "data.extract_contacts"]
    if args.threshold:
        cmd += ["--threshold", str(args.threshold)]
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)

    print("\n" + "=" * 60)
    print("Stage 2: Build training dataset")
    print("=" * 60)
    cmd = [sys.executable, "-m", "data.build_dataset"]
    if args.intent:
        cmd += ["--intent", args.intent]
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def cmd_train(args):
    """Stage 3: Train affordance model."""
    print("=" * 60)
    print("Stage 3: Train Affordance Model")
    print("=" * 60)
    cmd = [sys.executable, "-m", "model.train",
           "--epochs", str(args.epochs),
           "--batch_size", str(args.batch_size)]
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def cmd_infer(args):
    """Stage 4: Affordance inference + grasp pose generation."""
    print("=" * 60)
    print("Stage 4: Grasp Pose Generation")
    print("=" * 60)
    cmd = [sys.executable, "-m", "inference.grasp_pose",
           "--mesh", args.mesh,
           "--threshold", str(args.affordance_threshold)]
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    return result.returncode == 0


def cmd_sim(hdf5_path, args):
    """Stage 5: Isaac Sim grasp execution."""
    print("\n" + "=" * 60)
    print("Stage 5: Isaac Sim Grasp Execution")
    print("=" * 60)

    # Isaac Sim requires its own Python runtime
    isaac_python = os.path.join(config.ISAAC_SIM_PATH, "python.sh")
    if not os.path.exists(isaac_python):
        print(f"Error: Isaac Sim python not found: {isaac_python}")
        print(f"  Set ISAAC_SIM_PATH in your environment or .env file.")
        print(f"  Example: export ISAAC_SIM_PATH=/path/to/isaac-sim")
        return False

    sim_script = os.path.join(PROJECT_DIR, "sim", "run_grasp.py")
    cmd = [isaac_python, sim_script,
           "--hdf5", hdf5_path,
           "--object_scale", str(args.object_scale)]
    if args.headless:
        cmd.append("--headless")

    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="V2AP — Full pipeline from object mesh to simulated grasping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mesh path/to/object.obj            # inference + simulation
  python run.py --mesh path/to/object.obj --no-sim   # inference only
  python run.py --prepare                             # build training data
  python run.py --train --epochs 200                  # train model
  python run.py --execute output/grasps/xxx.hdf5      # run pre-generated grasp
        """,
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mesh", type=str, help="Object mesh path -> inference + simulation")
    mode.add_argument("--prepare", action="store_true", help="Stage 1-2: Extract contacts + build dataset")
    mode.add_argument("--train", action="store_true", help="Stage 3: Train model")
    mode.add_argument("--execute", type=str, help="Stage 5: Execute a pre-generated grasp HDF5")

    # General options
    parser.add_argument("--no-sim", action="store_true", help="Skip Isaac Sim (inference only)")
    parser.add_argument("--headless", action="store_true", help="Isaac Sim headless mode")
    parser.add_argument("--object_scale", type=float, default=config.OBJECT_SCALE)
    parser.add_argument("--affordance_threshold", type=float, default=config.AFFORDANCE_THRESHOLD)

    # Training options
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.TRAIN_BATCH_SIZE)

    # Data options
    parser.add_argument("--threshold", type=float, default=None, help="Contact distance threshold")
    parser.add_argument("--intent", type=str, default=None, help="OakInk intent filter (hold/use)")

    args = parser.parse_args()
    config.ensure_dirs()

    # ---- Route to command ----
    if args.prepare:
        cmd_prepare(args)
    elif args.train:
        cmd_train(args)
    elif args.execute:
        if not os.path.exists(args.execute):
            print(f"Error: HDF5 not found: {args.execute}")
            return
        cmd_sim(args.execute, args)
    elif args.mesh:
        # Main flow: inference -> simulation
        if not os.path.exists(args.mesh):
            print(f"Error: Mesh not found: {args.mesh}")
            return

        # Stage 4: Inference
        obj_name = os.path.splitext(os.path.basename(args.mesh))[0]
        hdf5_path = os.path.join(config.GRASPS_DIR, f"{obj_name}_grasp.hdf5")

        if not cmd_infer(args):
            print("Error: Inference failed")
            return

        if args.no_sim:
            print(f"\nGrasp pose saved: {hdf5_path}")
            print(f"To run in simulation:")
            print(f"  export ISAAC_SIM_PATH=/path/to/isaac-sim")
            print(f"  $ISAAC_SIM_PATH/python.sh sim/run_grasp.py --hdf5 {hdf5_path}")
            return

        # Stage 5: Simulation
        cmd_sim(hdf5_path, args)


if __name__ == "__main__":
    main()
