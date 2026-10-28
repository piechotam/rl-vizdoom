"""Run an agent in the ViZDoom deathmatch environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents import RandomAgent
from src.env import DoomEnv
from src.evaluation import run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an agent in ViZDoom deathmatch"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to run"
    )
    parser.add_argument(
        "--visible", action="store_true", help="Show the game window"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Policy/environment seed"
    )
    parser.add_argument(
        "--frame-skip", type=int, default=4, help="Number of tics per action"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = DoomEnv(visible=args.visible, frame_skip=args.frame_skip, seed=args.seed)
    agent = RandomAgent(num_actions=env.num_actions, seed=args.seed)

    print(
        f"Running {args.episodes} episodes (frame_skip={args.frame_skip}, seed={args.seed})"
    )
    print("=" * 70)

    for ep in range(1, args.episodes + 1):
        m = run_episode(env, agent, seed=args.seed, episode=ep)
        print(
            f"Episode {ep:3d} | "
            f"Kills: {m.kill_count:2d} | "
            f"Health: {m.final_health:5.1f} | "
            f"Survived: {m.survival_time:5d} tics | "
            f"Reward: {m.total_reward:6.1f}"
        )

    env.close()
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
