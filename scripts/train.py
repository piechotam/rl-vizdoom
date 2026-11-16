"""Train a DQN agent in ViZDoom deathmatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.dqn import DQNConfig
from src.training import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN in ViZDoom deathmatch")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--total-frames",
        type=int,
        default=10_000_000,
        help="Training budget in game tics (agent steps = frames / frame skip)",
    )
    parser.add_argument("--skill", type=int, default=None, help="doom_skill 1-5")
    parser.add_argument("--eval-every-frames", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default=None, help="mps, cuda or cpu")
    parser.add_argument(
        "--learning-starts", type=int, default=DQNConfig.learning_starts
    )
    parser.add_argument("--buffer-size", type=int, default=DQNConfig.buffer_size)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Default: fs{frame_skip}/seed{seed}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"fs{args.frame_skip}/seed{args.seed}"
    config = TrainConfig(
        output_dir=Path("results/dqn") / run_name,
        tensorboard_dir=Path("runs") / run_name,
        frame_skip=args.frame_skip,
        total_frames=args.total_frames,
        seed=args.seed,
        skill=args.skill,
        eval_every_frames=args.eval_every_frames,
        eval_episodes=args.eval_episodes,
        device=args.device,
        dqn=DQNConfig(
            learning_starts=args.learning_starts, buffer_size=args.buffer_size
        ),
    )
    train(config)


if __name__ == "__main__":
    main()
