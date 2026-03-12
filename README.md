# Afterimage

A robot learning project exploring behavioral cloning, diffusion policies, and action chunking transformers on the PushT task using [LeRobot](https://github.com/huggingface/lerobot).

## Project Phases

| Phase | Method | Status |
|-------|--------|--------|
| 1 | MLP Behavioral Cloning | ⬜ Not started |
| 2 | Diffusion Policy | ⬜ Not started |
| 3 | ACT (Action Chunking Transformer) | ⬜ Not started |

## Setup

```bash
# Clone and install LeRobot (editable mode)
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[pusht]"
cd ..

# Install project dependencies
pip install wandb matplotlib numpy
```

## Training

Each phase has a corresponding training script in `scripts/`:

```bash
# Phase 1: MLP BC
bash scripts/train_mlp_bc.sh

# Phase 2: Diffusion Policy
bash scripts/train_diffusion.sh

# Phase 3: ACT
bash scripts/train_act.sh
```

## Evaluation

```bash
# Evaluate any checkpoint
python scripts/eval.py --checkpoint_dir <path_to_checkpoint>

# Compare all trained policies
python analysis/compare_policies.py
```

## Hardware

- GPU: NVIDIA RTX 5060 (8GB VRAM)
- OS: Windows + WSL2
- CUDA: 12.0
- Python: 3.12

## Structure

```
afterimage/
├── README.md
├── journal.md              # daily learning journal
├── scripts/                # training & eval shell scripts + wrappers
├── analysis/               # visualization & comparison code
├── notebooks/              # exploration (optional)
└── outputs/                # gitignored: models, videos, wandb logs
```
