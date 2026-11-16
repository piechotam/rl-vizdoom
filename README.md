# RL ViZDoom — deathmatch

Reinforcement learning agents for the deathmatch scenario in
[ViZDoom](https://vizdoom.farama.org): a from-scratch **Double + Dueling DQN**
(PyTorch) compared against a random and a never-shooting baseline.

## Setup

- Input: last 4 RGB frames downscaled to 60×45 (12 channels), frame skip 4, plus
  14 normalized game variables (health, armor, weapon, ammo — the HUD is off).
- Auxiliary loss: a second head predicts "enemy in view" and "enemy under the
  crosshair" from the engine's labels buffer (as in Arnold, Lample & Chaplot 2017).
- Actions: 31 button combinations (movement, turning, shooting, weapon switching,
  sprint) — see `src/actions.py`.
- Training reward: shaped from game-variable deltas (`src/reward.py`); kills count
  only if the agent hit something in the previous 2 s, because the game also
  counts monsters killed by other monsters. Evaluation uses the original reward.

## Usage

```bash
uv sync

# train (10M game frames ≈ 2.5M agent steps at frame skip 4; MPS/CUDA if available)
uv run python -u scripts/train.py --frame-skip 4 --seed 0
# -> results/dqn/fs4/seed0/{eval.csv,train_episodes.csv,best.pt,config.json}

# live training curves
uv run tensorboard --logdir runs --host 127.0.0.1

# evaluation: 100 episodes per seed; for dqn seed N uses results/dqn/fs4/seedN/best.pt
uv run python scripts/evaluate.py --agent dqn --checkpoint-dir results/dqn/fs4
uv run python scripts/evaluate.py --agent random      # baselines: 5 seeds
uv run python scripts/evaluate.py --agent no_attack
uv run python scripts/evaluate.py --agent dqn --skill 5   # other difficulty

# figures for the presentation -> docs/presentations/img/
uv run python scripts/plot_results.py

# watch / record the trained agent
uv run python scripts/run_agent.py --agent dqn --visible --episodes 3
uv run python scripts/run_agent.py --agent dqn --episodes 1 --record media/dqn.mp4
```

## Evaluation protocol

Each seed drives both the policy and the game engine, so every `(seed, episode)`
is reproducible. The headline number is the mean of per-seed means ± std with a
95% bootstrap CI; the IQM is reported as a robust alternative. Main metrics:
kills, kills attributed to the agent, survival time, share of episodes reaching
the time limit, and the margin over the never-shooting baseline.
