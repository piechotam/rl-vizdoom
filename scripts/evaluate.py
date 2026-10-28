"""Evaluate an agent over many episodes and policy seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vizdoom as vzd

from src.agents import RandomAgent
from src.env import DoomEnv
from src.evaluation import AgentFactory, evaluate

AGENTS: dict[str, AgentFactory] = {
    "random": lambda env, seed: RandomAgent(env.num_actions, seed=seed),
    "no_attack": lambda env, seed: RandomAgent(
        env.num_actions,
        seed=seed,
        excluded_actions=[env.action_index(vzd.Button.ATTACK)],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an agent in ViZDoom deathmatch"
    )
    parser.add_argument(
        "--agent", choices=AGENTS, default="random", help="Agent to evaluate"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Policy/environment seeds (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Episodes per seed (default: 100)",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=4, help="Number of tics per action"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for episodes.csv and summary.json (default: results/<agent>)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or f"results/{args.agent}")

    env = DoomEnv(visible=False, frame_skip=args.frame_skip)

    print(
        f"Evaluating '{args.agent}' agent: {args.episodes} episodes x "
        f"{len(args.seeds)} seeds {args.seeds} ..."
    )
    collector = evaluate(env, AGENTS[args.agent], args.seeds, args.episodes)
    env.close()

    collector.to_csv(output_dir / "episodes.csv")
    collector.summary_to_json(
        output_dir / "summary.json",
        agent=args.agent,
        episodes_per_seed=args.episodes,
        frame_skip=args.frame_skip,
    )
    print(f"\nResults saved to: {output_dir}/")
    collector.print_summary(title=f"Agent '{args.agent}'")


if __name__ == "__main__":
    main()
