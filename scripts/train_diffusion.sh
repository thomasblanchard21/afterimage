#!/bin/bash
# Phase 2: Diffusion Policy on PushT
#
# Trains a diffusion policy (DDPM for action generation) using LeRobot's
# built-in implementation based on Chi et al. 2023.
#
# INTERVIEW-CRITICAL: Understand denoising score matching, why diffusion
# handles multimodal action distributions, and action chunking.

set -e

echo "=== Afterimage Phase 2: Diffusion Policy on PushT ==="

python lerobot/scripts/train.py \
  --policy.type=diffusion \
  --dataset.repo_id=lerobot/pusht \
  --env.type=pusht \
  --output_dir=outputs/train/phase2_diffusion \
  --batch_size=64 \
  --steps=200000 \
  --eval_freq=25000 \
  --save_freq=25000 \
  --wandb.enable=true \
  --device=cuda
