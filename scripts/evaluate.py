"""Evaluate an agent over many episodes and produce statistical results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents import RandomAgent
from src.env import DoomEnv
from src.metrics import EpisodeMetrics, MetricsCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an agent in ViZDoom deathmatch"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--frame-skip", type=int, default=4, help="Number of tics per action"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/random_agent_metrics.csv",
        help="Path for CSV output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = DoomEnv(visible=False, frame_skip=args.frame_skip)
    agent = RandomAgent(num_actions=env.num_actions, seed=args.seed)
    collector = MetricsCollector()

    print(
        f"Evaluating RandomAgent for {args.episodes} episodes (seed={args.seed}) ..."
    )

    for ep in range(1, args.episodes + 1):
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

        metrics = EpisodeMetrics(
            episode=ep,
            kill_count=int(info["kill_count"]),
            death_count=1 if info["health"] <= 0 else 0,
            survival_time=steps * args.frame_skip,
            total_reward=total_reward,
            final_health=float(info["health"]),
            final_armor=float(info["armor"]),
        )
        collector.record(metrics)

        if ep % 10 == 0 or ep == args.episodes:
            print(f"  [{ep:>{len(str(args.episodes))}d}/{args.episodes}] done")

    env.close()

    collector.to_csv(args.output)
    print(f"\nResults saved to: {args.output}")
    collector.print_summary(title="Random Agent Baseline")


if __name__ == "__main__":
    main()
