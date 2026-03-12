"""
Afterimage — Evaluation wrapper

Thin wrapper around LeRobot's eval script that also logs results to W&B.
Usage:
    python scripts/eval.py --checkpoint_dir outputs/train/phase2_diffusion/checkpoints/last/pretrained_model
"""

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy on PushT")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the pretrained_model checkpoint directory",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save eval outputs. Defaults to outputs/eval/<checkpoint_name>",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        # Derive output dir from checkpoint path
        checkpoint_name = args.checkpoint_dir.rstrip("/").split("/")[-3]  # e.g. phase2_diffusion
        args.output_dir = f"outputs/eval/{checkpoint_name}"

    cmd = [
        sys.executable, "lerobot/scripts/eval.py",
        f"--policy.path={args.checkpoint_dir}",
        f"--output_dir={args.output_dir}",
        "--env.type=pusht",
        f"--eval.n_episodes={args.n_episodes}",
        "--eval.batch_size=10",
        "--device=cuda",
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
