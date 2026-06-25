#!/usr/bin/env python3
"""
smoke_test.py — V2AP 部署验证脚本
==============================================
在 Fresh 机器上运行，验证所有 pipeline 阶段能通过。
每个阶段只处理 1 条序列，总时间约 2-3 小时。

用法:
  python smoke_test.py            # 运行全部阶段
  python smoke_test.py --phase 1  # 只跑 Phase 1A
  python smoke_test.py --skip-gpu # 跳过需要 GPU 的阶段（仅检查环境）

输出: smoke_test_report.txt
"""

import os, sys, subprocess, time, argparse
from pathlib import Path

PROJECT = Path(__file__).parent
REPORT = PROJECT / "smoke_test_report.txt"

results = []


def run(label, cmd, env_name=None, timeout=1800, check_output=None):
    """Run a command and record pass/fail."""
    print(f"\n{'='*60}")
    print(f"  [{label}]")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    full_cmd = cmd
    if env_name:
        full_cmd = ["conda", "run", "-n", env_name, "--no-capture-output"] + cmd

    try:
        proc = subprocess.run(
            full_cmd, timeout=timeout,
            capture_output=False, text=True, cwd=str(PROJECT)
        )
        elapsed = time.time() - t0
        ok = proc.returncode == 0

        # Optionally verify output file exists
        if ok and check_output:
            ok = all(Path(f).exists() for f in check_output)
            if not ok:
                print(f"  ❌ Expected output(s) not found: {check_output}")

        status = "✅ PASS" if ok else "❌ FAIL"
        msg = f"{status}  {label}  ({elapsed:.0f}s)"
        print(msg)
        results.append((label, ok, elapsed, proc.returncode))
        return ok
    except subprocess.TimeoutExpired:
        msg = f"⏰ TIMEOUT  {label}"
        print(msg)
        results.append((label, False, timeout, -999))
        return False
    except Exception as e:
        msg = f"❌ ERROR  {label}: {e}"
        print(msg)
        results.append((label, False, 0, -1))
        return False


def check(label, condition, detail=""):
    status = "✅ PASS" if condition else "❌ FAIL"
    msg = f"{status}  {label}  {detail}"
    print(msg)
    results.append((label, condition, 0, 0))
    return condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0, help="Run only this phase (0=all)")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU-intensive steps")
    args = parser.parse_args()

    print("=" * 60)
    print("  V2AP Smoke Test")
    print(f"  Project: {PROJECT}")
    print("=" * 60)

    # ── Environment checks ────────────────────────────────────────
    if args.phase in (0, 0):
        print("\n[PHASE 0] Environment Checks")

        check("Python ≥ 3.9", sys.version_info >= (3, 9),
              f"got {sys.version}")

        check("config.py importable",
              run("import config", ["python3", "-c", "import config; print(config.DATA_HUB)"]))

        check("huggingface_hub installed",
              run("hf_hub", ["python3", "-c", "from huggingface_hub import HfApi; print('ok')"]))

        check("setup_weights.py exists", (PROJECT / "setup_weights.py").exists())

        check("EgoDataMasks present",
              (PROJECT / "data_hub" / "ProcessedData" / "obj_recon_input" / "egocentric").exists(),
              "run: python setup_weights.py --tool egomasks")

        fp_root_env = os.environ.get("FP_ROOT", str(PROJECT / "thirdparty" / "foundationpose"))
        check("FoundationPose weights",
              (Path(fp_root_env) / "weights" / "2023-10-28-18-33-37" / "model_best.pth").exists(),
              "run: python setup_weights.py --tool fp")

        check("HaWoR weights",
              (PROJECT / "thirdparty" / "hawor" / "weights" / "hawor" / "checkpoints" / "hawor.ckpt").exists(),
              "run: python setup_weights.py --tool hawor")

        check("MegaSAM weights",
              (PROJECT / "thirdparty" / "megasam" / "checkpoints" / "megasam_final.pth").exists(),
              "run: python setup_weights.py --tool megasam")

        check("DepthPro weights",
              (PROJECT / "thirdparty" / "depthpro" / "checkpoints" / "depth_pro.pt").exists(),
              "run: python setup_weights.py --tool depthpro")

        check("MANO_RIGHT.pkl (HaPTIC)",
              (PROJECT / "thirdparty" / "haptic" / "assets" / "mano" / "MANO_RIGHT.pkl").exists(),
              "manual download required")

        check("DATA_HUB exists",
              (PROJECT / "data_hub").exists())

        check("DexYCB raw data",
              (PROJECT / "data" / "third_person" / "dexycb").exists())

        # setuptools<70 required for detectron2 (pkg_resources removed in >=70)
        import subprocess as _sp
        _st = _sp.run(["python3", "-c", "import pkg_resources"], capture_output=True)
        check("setuptools<70 (detectron2 compat)",
              _st.returncode == 0,
              "fix: pip install 'setuptools<70'")

        # Build toolchain checks
        check("cmake available",
              run("cmake-check", ["cmake", "--version"]))
        check("nvcc available (for FP build)",
              run("nvcc-check", ["nvcc", "--version"]))
        check("FoundationPose cloned",
              (PROJECT / "thirdparty" / "foundationpose").exists() or
              bool(os.environ.get("FP_ROOT", "")),
              "export FP_ROOT=/path/to/foundationpose or clone to thirdparty/foundationpose")
        fp_root = os.environ.get("FP_ROOT",
                  str(PROJECT / "thirdparty" / "foundationpose"))
        check("FoundationPose built (mycpp)",
              (Path(fp_root) / "mycpp" / "build").exists(),
              "cd $FP_ROOT && bash build_all_conda.sh")

        # Pre-flight system check
        import shutil as _sh
        check("disk space > 100 GB free",
              _sh.disk_usage(str(PROJECT)).free > 100 * 1024**3,
              f"free: {_sh.disk_usage(str(PROJECT)).free / 1024**3:.0f} GB")

    # ── Phase 1A ──────────────────────────────────────────────────
    if args.phase in (0, 1) and not args.skip_gpu:
        print("\n[PHASE 1A] Third-Person Pipeline (1 DexYCB sequence)")

        run("1A-depth-pro",
            ["python", "pipeline/third_person/batch_depth_pro.py", "--dataset", "dexycb", "--limit", "1"],
            env_name="depth-pro", timeout=600)

        run("1A-hawor-ego-smoke",
            ["python", "pipeline/ego/batch_hawor.py", "--dataset", "egodex", "--end", "1"],
            env_name="hawor", timeout=900)

        run("1A-haptic",
            ["python", "pipeline/third_person/batch_haptic.py", "--dataset", "dexycb", "--limit", "1"],
            env_name="haptic", timeout=600)

        run("1A-fp-pose",
            ["python", "tools/batch/batch_obj_pose.py", "--dataset", "dexycb", "--limit", "1"],
            env_name="bundlesdf", timeout=1200)

    # ── Phase 1B ──────────────────────────────────────────────────
    if args.phase in (0, 2) and not args.skip_gpu:
        print("\n[PHASE 1B] Egocentric Pipeline (1 EgoDex sequence)")

        run("1B-megasam",
            ["python", "pipeline/ego/batch_megasam.py", "--dataset", "egodex",
             "--start", "0", "--end", "1"],
            env_name="mega_sam", timeout=900)

        run("1B-hawor-ego",
            ["python", "pipeline/ego/batch_hawor.py", "--dataset", "egodex", "--end", "1"],
            env_name="hawor", timeout=900)

        run("1B-fp-ego",
            ["python", "tools/batch/batch_obj_pose_ego.py",
             "--dataset", "egodex", "--limit", "1"],
            env_name="bundlesdf", timeout=1200)

    # ── Phase 2 ───────────────────────────────────────────────────
    if args.phase in (0, 3) and not args.skip_gpu:
        print("\n[PHASE 2] Aggregate HumanPrior (DexYCB subset)")

        run("2-align-mano-fp",
            ["python", "pipeline/ego/batch_align_mano_fp.py", "--dataset", "dexycb",
             "--obj", "003_cracker_box"],
            env_name="bundlesdf", timeout=600)

    # ── Phase 4 ───────────────────────────────────────────────────
    if args.phase in (0, 4):
        print("\n[PHASE 4] Build Dataset + Training (gtfree, 1 epoch)")

        hp_dir = PROJECT / "data_hub" / "human_prior"
        n_hp = len(list(hp_dir.glob("*.hdf5"))) if hp_dir.exists() else 0
        check("human_prior files present",
              n_hp > 0,
              f"found {n_hp} .hdf5 files — need at least 1 to train")

        if n_hp > 0 and not args.skip_gpu:
            run("4-build-dataset",
                ["python", "pipeline/aggregate/build_dataset.py", "--gtfree"],
                env_name="bundlesdf", timeout=600)

            run("4-train-1epoch",
                # model/train.py uses --batch_size (underscore), not --batch-size
                ["python", "-m", "model.train", "--epochs", "1", "--batch_size", "4"],
                env_name="bundlesdf", timeout=600)

    # ── Report ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SMOKE TEST REPORT")
    print("=" * 60)

    passed = sum(1 for _, ok, _, _ in results if ok)
    total  = len(results)

    for label, ok, elapsed, code in results:
        icon = "✅" if ok else "❌"
        t = f"({elapsed:.0f}s)" if elapsed > 0 else ""
        print(f"  {icon}  {label} {t}")

    print(f"\n  Result: {passed}/{total} passed")
    overall = passed == total

    # Write report
    with open(REPORT, "w") as f:
        f.write(f"V2AP Smoke Test\n")
        f.write(f"Project: {PROJECT}\n\n")
        for label, ok, elapsed, code in results:
            f.write(f"{'PASS' if ok else 'FAIL'}  {label}  ({elapsed:.0f}s)\n")
        f.write(f"\n{passed}/{total} passed\n")

    print(f"\n  Full report: {REPORT}")
    print("=" * 60)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
