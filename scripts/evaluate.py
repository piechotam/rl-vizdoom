"""Evaluate an agent over many episodes and policy seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import vizdoom as vzd

from src.agents import Agent, RandomAgent
from src.agents.dqn import DQNAgent
from src.env import DoomEnv
from src.evaluation import AgentFactory, evaluate


class CountingAgent:
    """Wraps an agent and tallies the actions it takes."""

    def __init__(self, agent: Agent, counts: np.ndarray) -> None:
        self.agent = agent
        self.counts = counts

    def act(self, observation: np.ndarray, info: dict) -> int:
        action = self.agent.act(observation, info)
        self.counts[action] += 1
        return action

    def reset(self) -> None:
        self.agent.reset()


def make_factory(args: argparse.Namespace) -> AgentFactory:
    if args.agent == "random":
        return lambda env, seed: RandomAgent(env.num_actions, seed=seed)
    if args.agent == "no_attack":
        return lambda env, seed: RandomAgent(
            env.num_actions,
            seed=seed,
            excluded_actions=env.actions_with(vzd.Button.ATTACK),
        )

    def dqn_factory(env: DoomEnv, seed: int) -> Agent:
        path = Path(args.checkpoint_dir) / f"seed{seed}" / args.checkpoint
        return DQNAgent.load(path, device=args.device, seed=seed)

    return dqn_factory


def checkpoint_frame_skip(checkpoint_dir: Path, seed: int) -> int:
    with open(checkpoint_dir / f"seed{seed}" / "config.json") as f:
        return int(json.load(f)["frame_skip"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an agent in ViZDoom deathmatch"
    )
    parser.add_argument(
        "--agent",
        choices=["random", "no_attack", "dqn"],
        default="random",
        help="Agent to evaluate",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Policy/environment seeds (default: 0-4 for baselines, 0-2 for dqn); "
        "for dqn the seed selects the training run <checkpoint-dir>/seed<N>",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Episodes per seed (default: 100)"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Tics per action (default: 4, or the training value for dqn)",
    )
    parser.add_argument("--skill", type=int, default=None, help="doom_skill 1-5")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="results/dqn/fs4",
        help="dqn: directory with seed<N>/ training runs",
    )
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Default: results/<agent>[/skill<N>] or <checkpoint-dir>/eval[_skill<N>]",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds is None:
        args.seeds = [0, 1, 2] if args.agent == "dqn" else [0, 1, 2, 3, 4]
    if args.frame_skip is None:
        args.frame_skip = (
            checkpoint_frame_skip(Path(args.checkpoint_dir), args.seeds[0])
            if args.agent == "dqn"
            else 4
        )
    skill_suffix = f"skill{args.skill}" if args.skill is not None else ""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.agent == "dqn":
        name = f"eval_{skill_suffix}" if skill_suffix else "eval"
        output_dir = Path(args.checkpoint_dir) / name
    else:
        output_dir = Path("results") / args.agent / skill_suffix

    env = DoomEnv(visible=False, frame_skip=args.frame_skip, skill=args.skill)
    counts = np.zeros(env.num_actions, dtype=np.int64)
    base_factory = make_factory(args)

    def factory(env: DoomEnv, seed: int) -> Agent:
        return CountingAgent(base_factory(env, seed), counts)

    print(
        f"Evaluating '{args.agent}' agent: {args.episodes} episodes x "
        f"{len(args.seeds)} seeds {args.seeds}, frame_skip={args.frame_skip}, "
        f"skill={args.skill if args.skill is not None else 'cfg'} ..."
    )
    collector = evaluate(env, factory, args.seeds, args.episodes)

    collector.to_csv(output_dir / "episodes.csv")
    collector.summary_to_json(
        output_dir / "summary.json",
        agent=args.agent,
        episodes_per_seed=args.episodes,
        frame_skip=args.frame_skip,
        skill=args.skill,
        checkpoint_dir=args.checkpoint_dir if args.agent == "dqn" else None,
    )
    with open(output_dir / "action_counts.json", "w") as f:
        json.dump(
            dict(zip(env.action_names, counts.tolist(), strict=True)), f, indent=2
        )
    env.close()
    print(f"\nResults saved to: {output_dir}/")
    collector.print_summary(title=f"Agent '{args.agent}'")


if __name__ == "__main__":
    main()
