"""Integration tests that run the real ViZDoom engine (headless)."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from src.agents import RandomAgent
from src.env import DoomEnv
from src.evaluation import evaluate, run_episode


@pytest.fixture(scope="module")
def env():
    env = DoomEnv(visible=False, frame_skip=4)
    yield env
    env.close()


def random_agent(num_actions: int, seed: int) -> RandomAgent:
    return RandomAgent(num_actions, seed=seed)


def test_reset_returns_screen(env):
    obs = env.reset()
    assert obs.ndim == 3
    assert obs.shape[2] == 3
    assert obs.dtype == np.uint8


def test_action_space(env):
    # no-op + one action per button
    assert env.num_actions == env.game.get_available_buttons_size() + 1
    assert not any(env.action_space[0])
    assert all(sum(a) == 1 for a in env.action_space[1:])


def test_info_contains_game_variables(env):
    env.reset()
    _, _, done, info = env.step(0)
    assert not done
    for key in ("killcount", "health", "armor", "dead", "episode_time"):
        assert key in info
    assert info["health"] > 0


def test_episode_metrics_are_consistent(env):
    env.seed(0)
    m = run_episode(env, RandomAgent(env.num_actions, seed=0))
    assert m.died != m.timed_out
    assert 0 < m.survival_time <= env.episode_timeout
    assert m.kill_count >= 0
    assert m.shots_fired >= 0
    assert 0 <= m.action_entropy <= np.log2(env.num_actions) + 1e-9


def test_same_seed_is_reproducible(env):
    runs = [
        [asdict(m) for m in evaluate(env, random_agent, [0], 2, verbose=False).episodes]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


def test_different_seeds_differ(env):
    episodes = evaluate(env, random_agent, [0, 1], 1, verbose=False).episodes
    a, b = (asdict(m) | {"seed": 0} for m in episodes)
    assert a != b
