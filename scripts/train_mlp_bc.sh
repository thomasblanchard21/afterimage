#!/bin/bash
# Phase 1: MLP Behavioral Cloning on PushT
# 
# This trains a simple MLP policy using behavioral cloning.
# LeRobot doesn't have a standalone "mlp" policy type — we use the diffusion
# policy with minimal denoising steps as a baseline, OR we write a custom
# training script (see scripts/train_mlp_bc_custom.py).
#
# For now, this is a placeholder. We'll finalize the approach in Phase 1.
#
# INTERVIEW-CRITICAL: Understand why we start with the simplest possible
# baseline before moving to more complex methods.

set -e

echo "=== Afterimage Phase 1: MLP BC on PushT ==="
echo "TODO: Implement after environment setup is verified"

# Example of what the LeRobot training command looks like:
# python lerobot/scripts/train.py \
#   --policy.type=diffusion \
#   --dataset.repo_id=lerobot/pusht \
#   --env.type=pusht \
#   --output_dir=outputs/train/phase1_mlp_bc \
#   --wandb.enable=true \
#   --device=cuda
