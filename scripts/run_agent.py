"""Run an agent in the ViZDoom deathmatch environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents import RandomAgent
from src.env import DoomEnv


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
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--frame-skip", type=int, default=4, help="Number of tics per action"
    )
    return parser.parse_args()


def run_episode(env: DoomEnv, agent: RandomAgent) -> dict:
    """Run a single episode and return stats."""
    obs = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    return {
        "kill_count": int(info["kill_count"]),
        "health": float(info["health"]),
        "survival_tics": steps * env._frame_skip,
        "total_reward": total_reward,
    }


def main() -> None:
    args = parse_args()

    env = DoomEnv(visible=args.visible, frame_skip=args.frame_skip)
    agent = RandomAgent(num_actions=env.num_actions, seed=args.seed)

    print(
        f"Running {args.episodes} episodes (frame_skip={args.frame_skip}, seed={args.seed})"
    )
    print("=" * 70)

    for ep in range(1, args.episodes + 1):
        stats = run_episode(env, agent)
        print(
            f"Episode {ep:3d} | "
            f"Kills: {stats['kill_count']:2d} | "
            f"Health: {stats['health']:5.1f} | "
            f"Survived: {stats['survival_tics']:5d} tics | "
            f"Reward: {stats['total_reward']:6.1f}"
        )

    env.close()
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
