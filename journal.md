# Afterimage — Learning Journal

## How to use this journal
After each session, add an entry with:
- Date and session duration
- What you worked on
- Key concepts learned (mark interview-relevant ones)
- Blockers encountered
- Next session plan

---

## Session 1 — 12/03
**Duration:** ~3 hours
**Phase:** Setup + Dataset Exploration

### What I did
- Set up project structure (scripts/, analysis/, outputs/)
- Installed LeRobot + dependencies in WSL2 conda env
- Verified CUDA, W&B, PushT dataset with verify_setup.py
- First git commit
- Wrote dataset visualization script with frame playback (matplotlib TkAgg + FuncAnimation)
- Explored PushT dataset structure

### Key concepts
- **[INTERVIEW]** Action is goal position (position control), not velocity — the agent moves *toward* the predicted position
- Dataset keys: observation.state (2D agent pos), observation.image (96x96), action (2D goal pos), next.reward (overlap %), episode_index, frame_index
- 206 episodes, ~125 frames each at 10fps
- next.done = max timesteps reached (fixed horizon), next.success = 95% overlap threshold

### Blockers
- WSL2 display: plt.imshow() didn't work out of the box. Fixed with matplotlib.use('TkAgg') + sudo apt install python3-tk
- LeRobot API confusion: episodes= parameter exists but docs unclear. Used Subset + hf_dataset filtering instead

### Next session
- Phase 1: MLP BC baseline

---

## Session 2 — 13/03
**Duration:** ~3 hours
**Phase:** Phase 1 — MLP Behavioral Cloning

### What I did
- Built custom MLP BC model (2→256→256→2) with MSE loss, Adam optimizer
- Implemented episode-based train/val split (episodes 0-179 train, 180-205 val)
- Fixed normalization issue: raw [0,512] coordinates → [0,1] range, loss dropped from ~400 to ~0.0025
- Trained for 5 epochs, model converges by epoch 1
- Built evaluation script: loads trained model, runs in PushT gym env, renders episodes with FuncAnimation
- Observed result: agent learned to move toward center of image — the mean action

### Key concepts
- **[INTERVIEW]** Normalizing inputs/targets is critical — large ranges cause gradient issues and uninterpretable loss values
- **[INTERVIEW]** Multimodal action problem: MSE loss averages over multiple valid actions → model predicts the mean → useless policy. This is the core motivation for diffusion policies
- **[INTERVIEW]** Episode-based splitting (not random frame splitting) prevents data leakage from temporal correlation in sequential data
- **[INTERVIEW]** State-only BC fails when observations are insufficient — model can't distinguish situations requiring different actions (agent position alone doesn't tell you where the block is)
- Dataset only has observation.state (2D agent pos) and observation.image — no block state available, so image input is the only path to improvement

### Blockers
- LeRobotDataset returns dicts, not tuples — can't unpack in for loop
- Building index split by iterating dataset[i] is very slow — use dataset.hf_dataset["episode_index"] column access instead

### Next session
- Phase 2: Diffusion Policy on PushT using LeRobot's built-in implementation
- Will use image observations to give the model block information
- Compare against Phase 1 baseline