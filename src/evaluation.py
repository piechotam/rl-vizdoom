"""Shared evaluation loop: run episodes, collect metrics, iterate over seeds."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from src.agents import Agent
from src.env import DoomEnv
from src.metrics import TICS_PER_SECOND, EpisodeMetrics, MetricsCollector

# (num_actions, seed) -> agent
AgentFactory = Callable[[int, int], Agent]


def run_episode(
    env: DoomEnv, agent: Agent, seed: int = 0, episode: int = 0
) -> EpisodeMetrics:
    """Play one episode and compute all metrics step by step."""
    agent.reset()
    obs = env.reset()
    prev = env.get_info()

    total_reward = 0.0
    shots = ammo_collected = 0
    health_collected = armor_collected = distance = 0.0
    action_counts = np.zeros(env.num_actions, dtype=int)

    while True:
        action = agent.act(obs)
        action_counts[action] += 1
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Ammo deltas are only comparable when the weapon did not change
        if info["selected_weapon"] == prev["selected_weapon"]:
            d_ammo = info["selected_weapon_ammo"] - prev["selected_weapon_ammo"]
            if d_ammo < 0:
                shots += int(-d_ammo)
            else:
                ammo_collected += int(d_ammo)
        if not info["dead"]:
            health_collected += max(0.0, info["health"] - prev["health"])
            armor_collected += max(0.0, info["armor"] - prev["armor"])
            distance += float(
                np.hypot(
                    info["position_x"] - prev["position_x"],
                    info["position_y"] - prev["position_y"],
                )
            )
        prev = info
        if done:
            break

    kills = int(info["killcount"])
    died = bool(info["dead"]) or info["health"] <= 0
    survival_time = int(info["episode_time"])
    probs = action_counts / action_counts.sum()
    probs = probs[probs > 0]

    return EpisodeMetrics(
        seed=seed,
        episode=episode,
        kill_count=kills,
        died=died,
        timed_out=not died and survival_time >= env.episode_timeout,
        survival_time=survival_time,
        total_reward=total_reward,
        kills_per_minute=kills / max(survival_time, 1) * TICS_PER_SECOND * 60,
        damage_taken=info["damage_taken"],
        hits_taken=int(info["hits_taken"]),
        shots_fired=shots,
        kills_per_shot=kills / shots if shots else 0.0,
        ammo_collected=ammo_collected,
        health_collected=health_collected,
        armor_collected=armor_collected,
        distance_travelled=distance,
        action_entropy=float(-(probs * np.log2(probs)).sum()),
        final_health=info["health"],
        final_armor=info["armor"],
    )


def evaluate(
    env: DoomEnv,
    agent_factory: AgentFactory,
    seeds: Iterable[int],
    episodes_per_seed: int,
    verbose: bool = True,
) -> MetricsCollector:
    """Evaluate a freshly built agent for every seed.

    The same seed drives both the policy (``agent_factory``) and the game
    engine (``env.seed``), so each ``(seed, episode)`` pair is reproducible.
    """
    collector = MetricsCollector()
    for seed in seeds:
        env.seed(seed)
        agent = agent_factory(env.num_actions, seed)
        for ep in range(1, episodes_per_seed + 1):
            collector.record(run_episode(env, agent, seed=seed, episode=ep))
            if verbose and (ep % 10 == 0 or ep == episodes_per_seed):
                print(f"  seed {seed}: [{ep:>{len(str(episodes_per_seed))}d}/{episodes_per_seed}] done")
    return collector
