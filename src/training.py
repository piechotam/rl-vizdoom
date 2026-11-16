"""DQN training loop with periodic evaluation, CSV and TensorBoard logging."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from src.agents.dqn import DQNAgent, DQNConfig, ReplayBuffer
from src.env import DoomEnv
from src.evaluation import run_episode
from src.game_state import aux_targets
from src.metrics import TICS_PER_SECOND
from src.preprocessing import FrameStack, preprocess
from src.reward import RewardConfig, RewardShaper

# Periodic evaluation always replays the same engine episodes
EVAL_SEED = 10_000


@dataclass
class TrainConfig:
    output_dir: Path
    tensorboard_dir: Path
    frame_skip: int = 4
    total_frames: int = 10_000_000  # game tics; agent steps = total / frame_skip
    seed: int = 0
    skill: int | None = None
    eval_every_frames: int = 100_000
    eval_episodes: int = 10
    device: str | None = None
    dqn: DQNConfig = field(default_factory=DQNConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


class CsvLog:
    """CSV file written row by row; the header comes from the first row."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fields: list[str] | None = None

    def write(self, row: dict[str, Any]) -> None:
        new_file = self._fields is None
        if self._fields is None:
            self._fields = list(row)
        with open(self.path, "w" if new_file else "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fields)
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def periodic_eval(env: DoomEnv, agent: DQNAgent, episodes: int) -> dict[str, float]:
    """Mean metrics over ``episodes`` with the evaluation epsilon."""
    env.seed(EVAL_SEED)
    agent.epsilon = agent.config.eval_epsilon
    metrics = [run_episode(env, agent, episode=i) for i in range(episodes)]
    return {
        "kills": float(np.mean([m.kill_count for m in metrics])),
        "attributed_kills": float(np.mean([m.attributed_kills for m in metrics])),
        "survival_s": float(
            np.mean([m.survival_time for m in metrics]) / TICS_PER_SECOND
        ),
        "game_reward": float(np.mean([m.total_reward for m in metrics])),
        "hits": float(np.mean([m.hits for m in metrics])),
        "shots": float(np.mean([m.shots_fired for m in metrics])),
        "timeout_rate": float(np.mean([m.timed_out for m in metrics])),
        "action_entropy": float(np.mean([m.action_entropy for m in metrics])),
    }


def _config_json(cfg: TrainConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["output_dir"] = str(cfg.output_dir)
    data["tensorboard_dir"] = str(cfg.tensorboard_dir)
    return data


def train(cfg: TrainConfig) -> Path:
    """Train a DQN agent; returns the path of the best checkpoint."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / "config.json", "w") as f:
        json.dump(_config_json(cfg), f, indent=2)

    env = DoomEnv(
        frame_skip=cfg.frame_skip,
        seed=cfg.seed,
        skill=cfg.skill,
        labels=cfg.dqn.aux_dim > 0,
    )
    eval_env = DoomEnv(frame_skip=cfg.frame_skip, seed=EVAL_SEED, skill=cfg.skill)
    agent = DQNAgent(env.num_actions, cfg.dqn, device=cfg.device, seed=cfg.seed)
    c = agent.config
    buffer = ReplayBuffer(
        c.buffer_size,
        c.stack_size,
        seed=cfg.seed,
        n_step=c.n_step,
        gamma=c.gamma,
        feature_dim=c.feature_dim,
        aux_dim=c.aux_dim,
    )
    shaper = RewardShaper(cfg.reward)
    stack = FrameStack(c.stack_size)
    writer = SummaryWriter(str(cfg.tensorboard_dir))
    episode_log = CsvLog(cfg.output_dir / "train_episodes.csv")
    eval_log = CsvLog(cfg.output_dir / "eval.csv")
    best_path = cfg.output_dir / "best.pt"
    best_score = (-np.inf, -np.inf)

    total_steps = cfg.total_frames // cfg.frame_skip
    print(
        f"Training DQN: {total_steps} steps ({cfg.total_frames} frames), "
        f"frame_skip={cfg.frame_skip}, seed={cfg.seed}, device={agent.device}"
    )

    def evaluate_and_checkpoint(step: int) -> None:
        nonlocal best_score
        frames = step * cfg.frame_skip
        t0 = time.time()
        result = periodic_eval(eval_env, agent, cfg.eval_episodes)
        eval_log.write({"step": step, "frames": frames, **result})
        for key, value in result.items():
            writer.add_scalar(f"eval/{key}", value, frames)
        score = (result["attributed_kills"], result["survival_s"])
        if score > best_score:
            best_score = score
            agent.save(best_path, step=step, frames=frames, eval=result)
        print(
            f"[eval {frames:>9d} frames] kills {result['kills']:.2f} "
            f"(attributed {result['attributed_kills']:.2f}), "
            f"survival {result['survival_s']:.1f}s, "
            f"reward {result['game_reward']:.2f} ({time.time() - t0:.0f}s)"
        )

    def add_frame(frame: np.ndarray, info: dict, start: bool) -> np.ndarray | None:
        """Store a frame with its extras; return the game-state features."""
        features = agent.features_of(info)
        aux = aux_targets(info) if c.aux_dim else None
        buffer.add_frame(frame, episode_start=start, features=features, aux=aux)
        return features

    def start_episode() -> tuple[np.ndarray, np.ndarray | None]:
        frame = preprocess(env.reset())
        info = env.get_info()
        features = add_frame(frame, info, start=True)
        shaper.reset(info)
        return stack.reset(frame), features

    evaluate_and_checkpoint(0)
    state, features = start_episode()
    episode = 0
    ep_game_reward = ep_shaped_reward = 0.0
    ep_components: dict[str, float] = {}
    update_stats: list[dict[str, float]] = []
    t_start = t_last = time.time()
    step = 0

    try:
        for step in range(1, total_steps + 1):
            frames = step * cfg.frame_skip
            epsilon = agent.epsilon_at(step / total_steps)
            action = agent.select_action(state, features, epsilon)
            obs, game_reward, done, info = env.step(action)
            reward, components = shaper.step(info, game_reward, done)
            buffer.add_transition(action, reward, done)

            ep_game_reward += game_reward
            ep_shaped_reward += reward
            for key, value in components.items():
                ep_components[key] = ep_components.get(key, 0.0) + value

            if done or obs is None:
                episode += 1
                row = {
                    "episode": episode,
                    "step": step,
                    "frames": frames,
                    "game_reward": ep_game_reward,
                    "shaped_reward": ep_shaped_reward,
                    "kills": int(info["killcount"]),
                    "attributed_kills": shaper.attributed_kills,
                    "survival_s": info["episode_time"] / TICS_PER_SECOND,
                    "epsilon": epsilon,
                    **{f"r_{k}": v for k, v in ep_components.items()},
                }
                episode_log.write(row)
                for key, value in row.items():
                    if key not in ("episode", "step", "frames"):
                        writer.add_scalar(f"train/{key}", value, frames)
                ep_game_reward = ep_shaped_reward = 0.0
                ep_components = {}
                state, features = start_episode()
            else:
                frame = preprocess(obs)
                features = add_frame(frame, info, start=False)
                state = stack.push(frame)

            if step >= c.learning_starts and step % c.train_freq == 0:
                agent.set_lr(agent.lr_at(step / total_steps))
                update_stats.append(agent.update(buffer.sample(c.batch_size)))
            if step % c.target_update == 0:
                agent.sync_target()

            if step % 5_000 == 0:
                now = time.time()
                sps = 5_000 / (now - t_last)
                t_last = now
                writer.add_scalar("perf/steps_per_second", sps, frames)
                writer.add_scalar("train/epsilon_now", epsilon, frames)
                writer.add_scalar("update/lr", agent.lr_at(step / total_steps), frames)
                if update_stats:
                    for key in update_stats[0]:
                        mean = float(np.mean([s[key] for s in update_stats]))
                        writer.add_scalar(f"update/{key}", mean, frames)
                    update_stats = []
                eta_min = (total_steps - step) / sps / 60
                print(
                    f"step {step:>8d}/{total_steps} | episodes {episode} | "
                    f"eps {epsilon:.3f} | {sps:.0f} steps/s | ETA {eta_min:.0f} min"
                )

            if frames % cfg.eval_every_frames == 0:
                evaluate_and_checkpoint(step)
    finally:
        agent.save(cfg.output_dir / "last.pt", step=step, frames=step * cfg.frame_skip)
        writer.close()
        env.close()
        eval_env.close()
        print(f"Done in {(time.time() - t_start) / 60:.1f} min; best: {best_path}")

    return best_path
