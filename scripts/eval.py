"""
Afterimage — Evaluation wrapper

Thin wrapper around LeRobot's eval script that also logs results to W&B.
Usage:
    python scripts/eval.py --checkpoint_dir outputs/train/phase2_diffusion/checkpoints/last/pretrained_model
"""

import argparse
import subprocess
import sys
import gym_pusht
import gymnasium as gym
import torch
from scripts.train_custom_bc import CustomBC
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation


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
    # args = parse_args()

    # if args.output_dir is None:
    #     # Derive output dir from checkpoint path
    #     checkpoint_name = args.checkpoint_dir.rstrip("/").split("/")[-3]  # e.g. phase2_diffusion
    #     args.output_dir = f"outputs/eval/{checkpoint_name}"

    # cmd = [
    #     sys.executable, "lerobot/scripts/eval.py",
    #     f"--policy.path={args.checkpoint_dir}",
    #     f"--output_dir={args.output_dir}",
    #     "--env.type=pusht",
    #     f"--eval.n_episodes={args.n_episodes}",
    #     "--eval.batch_size=10",
    #     "--device=cuda",
    # ]

    # print(f"Running: {' '.join(cmd)}")
    # subprocess.run(cmd, check=True)

    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    obs, info = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomBC().to(device)
    model.load_state_dict(torch.load("outputs/models/custom_bc.pth"))

    frames = []

    total_reward = 0
    for i in range(100):
        agent_state = torch.tensor(obs[:2], dtype=torch.float32).to(device) / 512.0
        action = model(agent_state)
        action_denormalized = action.detach().cpu().numpy() * 512.0
        obs, reward, terminated, truncated, info = env.step(action_denormalized)
        if terminated or truncated:
            break
        frames.append(env.render())
        total_reward += reward
    print(f"Total reward: {total_reward}")
    env.close()

    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])
    height, width = frames[0].shape[0:2]
    ax.set_axis_off()

    def update(i):
        img.set_data(frames[i])
        return [img]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
