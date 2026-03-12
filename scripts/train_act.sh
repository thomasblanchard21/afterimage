#!/bin/bash
# Phase 3: ACT (Action Chunking with Transformers) on PushT
#
# Trains an ACT policy using LeRobot's built-in implementation
# based on Zhao et al. 2023 (Learning Fine-Grained Bimanual Manipulation).
#
# INTERVIEW-CRITICAL: Understand CVAE architecture, why action chunking
# helps with temporal consistency, encoder-decoder transformer for policy.

set -e

echo "=== Afterimage Phase 3: ACT on PushT ==="

python lerobot/scripts/train.py \
  --policy.type=act \
  --dataset.repo_id=lerobot/pusht \
  --env.type=pusht \
  --output_dir=outputs/train/phase3_act \
  --wandb.enable=true \
  --device=cuda
