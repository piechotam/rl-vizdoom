# RL ViZDoom — deathmatch

Reinforcement learning agents for the deathmatch scenario in
[ViZDoom](https://vizdoom.farama.org). A random agent serves as the baseline.

## Usage

```bash
uv sync

# watch the agent play in a game window
uv run python scripts/run_agent.py --visible --episodes 3

# evaluation: 5 seeds x 100 episodes, results in results/random/
uv run python scripts/evaluate.py --seeds 0 1 2 3 4 --episodes 100
```
