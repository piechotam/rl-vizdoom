"""Run an agent in the ViZDoom deathmatch environment (optionally record video)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.agents import Agent, RandomAgent
from src.agents.dqn import DQNAgent
from src.env import DoomEnv
from src.evaluation import run_episode
from src.metrics import TICS_PER_SECOND


class RecordingAgent:
    """Passes observations to ``agent`` and stores them as video frames."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.frames: list[np.ndarray] = []

    def act(self, observation: np.ndarray, info: dict) -> int:
        self.frames.append(observation)
        return self.agent.act(observation, info)

    def reset(self) -> None:
        self.agent.reset()


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an agent in ViZDoom deathmatch")
    parser.add_argument("--agent", choices=["random", "dqn"], default="random")
    parser.add_argument(
        "--checkpoint", type=str, default="results/dqn/fs4/seed0/best.pt"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to run"
    )
    parser.add_argument("--visible", action="store_true", help="Show the game window")
    parser.add_argument("--seed", type=int, default=0, help="Policy/environment seed")
    parser.add_argument(
        "--frame-skip", type=int, default=4, help="Number of tics per action"
    )
    parser.add_argument("--skill", type=int, default=None, help="doom_skill 1-5")
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Save an mp4 of every episode (suffix _epN added)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = DoomEnv(
        visible=args.visible,
        frame_skip=args.frame_skip,
        seed=args.seed,
        skill=args.skill,
    )
    if args.agent == "dqn":
        agent: Agent = DQNAgent.load(args.checkpoint, seed=args.seed)
    else:
        agent = RandomAgent(num_actions=env.num_actions, seed=args.seed)

    print(
        f"Running {args.episodes} episodes of '{args.agent}' "
        f"(frame_skip={args.frame_skip}, seed={args.seed})"
    )
    print("=" * 70)

    for ep in range(1, args.episodes + 1):
        recorder = RecordingAgent(agent)
        m = run_episode(env, recorder, seed=args.seed, episode=ep)
        print(
            f"Episode {ep:3d} | "
            f"Kills: {m.kill_count:2d} (own {m.attributed_kills:2d}) | "
            f"Health: {m.final_health:5.1f} | "
            f"Survived: {m.survival_time:5d} tics | "
            f"Reward: {m.total_reward:6.1f}"
        )
        if args.record:
            base = Path(args.record)
            path = base.with_name(f"{base.stem}_ep{ep}{base.suffix or '.mp4'}")
            write_video(path, recorder.frames, TICS_PER_SECOND / args.frame_skip)
            print(f"  saved {path}")

    env.close()
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
