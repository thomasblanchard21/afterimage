"""
Afterimage — Setup Verification

Run this after installation to verify everything is working:
    python scripts/verify_setup.py

Checks:
    1. Python version
    2. CUDA availability
    3. LeRobot importable
    4. PushT environment loadable
    5. PushT dataset downloadable
    6. W&B configured
"""

import sys


def check_python():
    v = sys.version_info
    print(f"[1/6] Python version: {v.major}.{v.minor}.{v.micro}", end=" ")
    if v.major == 3 and v.minor >= 12:
        print("✓")
        return True
    else:
        print("✗ (need >= 3.12)")
        return False


def check_cuda():
    print("[2/6] CUDA availability:", end=" ")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ ({torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda})")
            return True
        else:
            print("✗ (CUDA not available)")
            return False
    except ImportError:
        print("✗ (torch not installed)")
        return False


def check_lerobot():
    print("[3/6] LeRobot import:", end=" ")
    try:
        import lerobot
        print("✓")
        return True
    except ImportError as e:
        print(f"✗ ({e})")
        return False


def check_pusht_env():
    print("[4/6] PushT environment:", end=" ")
    try:
        import gym_pusht  # noqa: F401
        print("✓")
        return True
    except ImportError as e:
        print(f"✗ ({e})")
        return False


def check_pusht_dataset():
    print("[5/6] PushT dataset:", end=" ")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        metadata = LeRobotDatasetMetadata("lerobot/pusht")
        print(f"✓ ({metadata.total_episodes} episodes, {metadata.fps} fps)")
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False


def check_wandb():
    print("[6/6] W&B:", end=" ")
    try:
        import wandb
        if wandb.api.api_key:
            print("✓ (logged in)")
            return True
        else:
            print("✗ (not logged in — run: wandb login)")
            return False
    except ImportError:
        print("✗ (not installed — run: pip install wandb)")
        return False
    except Exception:
        print("✗ (not logged in — run: wandb login)")
        return False


def main():
    print("=" * 50)
    print("Afterimage — Setup Verification")
    print("=" * 50)
    print()

    results = [
        check_python(),
        check_cuda(),
        check_lerobot(),
        check_pusht_env(),
        check_pusht_dataset(),
        check_wandb(),
    ]

    print()
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"All {total}/{total} checks passed. You're ready to go!")
    else:
        print(f"{passed}/{total} checks passed. Fix the failures above before proceeding.")


if __name__ == "__main__":
    main()
