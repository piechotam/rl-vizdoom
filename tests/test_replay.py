"""Tests for the frame-based replay buffer."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.dqn import ReplayBuffer


def frame(value: int) -> np.ndarray:
    return np.full((3, 45, 60), value, dtype=np.uint8)


def stack_values(state: np.ndarray) -> list[int]:
    return [int(state[3 * i, 0, 0]) for i in range(state.shape[0] // 3)]


def fill_episode(buf: ReplayBuffer, values: list[int]) -> None:
    """Episode whose frames carry ``values``; the last step ends it."""
    for j, v in enumerate(values):
        buf.add_frame(frame(v), episode_start=j == 0)
        buf.add_transition(action=v % 3, reward=float(v), done=j == len(values) - 1)


def test_stacks_do_not_cross_episodes():
    buf = ReplayBuffer(capacity=100, stack_size=4)
    fill_episode(buf, [1, 2, 3])
    fill_episode(buf, [11, 12, 13, 14, 15])
    expected = {
        1: [1, 1, 1, 1],
        2: [1, 1, 1, 2],
        3: [1, 1, 2, 3],
        11: [11, 11, 11, 11],
        12: [11, 11, 11, 12],
        15: [12, 13, 14, 15],
    }
    batch = buf.sample(256)
    for state, reward in zip(batch.states, batch.rewards, strict=True):
        if int(reward) in expected:
            assert stack_values(state) == expected[int(reward)]


def test_next_state_and_done():
    buf = ReplayBuffer(capacity=100, stack_size=2)
    fill_episode(buf, [1, 2, 3])
    batch = buf.sample(128)
    for s, ns, r, d in zip(
        batch.states, batch.next_states, batch.rewards, batch.dones, strict=True
    ):
        if r == 3:
            assert d == 1.0  # terminal: next state is masked
        else:
            assert d == 0.0
            assert stack_values(ns)[-1] == stack_values(s)[-1] + 1


def test_frame_without_transition_is_not_sampled():
    buf = ReplayBuffer(capacity=100, stack_size=2)
    buf.add_frame(frame(1), episode_start=True)
    with pytest.raises(ValueError):
        buf.sample(1)
    buf.add_transition(0, 1.0, done=False)
    buf.add_frame(frame(2), episode_start=False)  # no transition yet
    assert len(buf) == 1
    assert set(buf.sample(64).rewards.tolist()) == {1.0}


def test_wraparound_keeps_valid_stacks():
    buf = ReplayBuffer(capacity=10, stack_size=3)
    fill_episode(buf, list(range(1, 26)))
    batch = buf.sample(512)
    for state in batch.states:
        values = stack_values(state)
        # consecutive frames from the same episode, never overwritten data
        assert values == list(range(values[0], values[0] + 3))


def test_n_step_returns_stop_at_episode_end():
    gamma = 0.5
    buf = ReplayBuffer(capacity=100, stack_size=2, n_step=3, gamma=gamma)
    fill_episode(buf, [1, 2, 3, 4])
    fill_episode(buf, [11, 12])
    expected = {
        # first frame value -> (return, done, discount, bootstrap frame)
        1: (1 + 0.5 * 2 + 0.25 * 3, 0.0, 0.125, 4),
        2: (2 + 0.5 * 3 + 0.25 * 4, 1.0, 0.125, None),
        3: (3 + 0.5 * 4, 1.0, 0.25, None),
        4: (4, 1.0, 0.5, None),
        11: (11 + 0.5 * 12, 1.0, 0.25, None),
        12: (12, 1.0, 0.5, None),
    }
    batch = buf.sample(512)
    seen = set()
    for s, ns, r, d, g in zip(
        batch.states,
        batch.next_states,
        batch.rewards,
        batch.dones,
        batch.discounts,
        strict=True,
    ):
        first = stack_values(s)[-1]
        ret, done, discount, boot = expected[first]
        assert r == pytest.approx(ret)
        assert d == done
        assert g == pytest.approx(discount)
        if boot is not None:
            assert stack_values(ns)[-1] == boot
        seen.add(first)
    assert seen == set(expected)


def test_n_step_truncates_at_newest_transition():
    buf = ReplayBuffer(capacity=100, stack_size=1, n_step=3, gamma=0.5)
    for j, v in enumerate([1, 2, 3]):
        buf.add_frame(frame(v), episode_start=j == 0)
        buf.add_transition(action=0, reward=float(v), done=False)
    buf.add_frame(frame(4), episode_start=False)  # current state, no action yet
    batch = buf.sample(256)
    for s, ns, r, g in zip(
        batch.states, batch.next_states, batch.rewards, batch.discounts, strict=True
    ):
        first = stack_values(s)[-1]
        # every sum ends at the newest transition and bootstraps from frame 4
        assert stack_values(ns)[-1] == 4
        assert r == pytest.approx(
            sum(0.5**k * v for k, v in enumerate(range(first, 4)))
        )
        assert g == pytest.approx(0.5 ** (4 - first))


def test_features_and_aux_follow_states():
    buf = ReplayBuffer(capacity=100, stack_size=2, n_step=2, feature_dim=3, aux_dim=2)
    for j, v in enumerate([1, 2, 3, 4]):
        buf.add_frame(
            frame(v),
            episode_start=j == 0,
            features=np.full(3, v, dtype=np.float32),
            aux=np.array([v % 2, 1], dtype=np.uint8),
        )
        buf.add_transition(action=0, reward=0.0, done=j == 3)
    batch = buf.sample(256)
    for s, ns, f, nf, aux in zip(
        batch.states,
        batch.next_states,
        batch.features,
        batch.next_features,
        batch.aux,
        strict=True,
    ):
        v = stack_values(s)[-1]
        assert f.tolist() == [v] * 3
        assert aux.tolist() == [v % 2, 1]
        if v <= 2:  # 2-step bootstrap from frame v + 2
            assert nf.tolist() == [stack_values(ns)[-1]] * 3 == [v + 2] * 3


def test_buffer_with_features_requires_them():
    buf = ReplayBuffer(capacity=100, stack_size=2, feature_dim=3)
    with pytest.raises(ValueError):
        buf.add_frame(frame(1), episode_start=True)
