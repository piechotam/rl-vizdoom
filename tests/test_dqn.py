"""Tests for the DQN network and agent (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.dqn import Batch, DQNAgent, DQNConfig, DuelingQNetwork
from src.game_state import GAME_FEATURE_DIM

INFO = {
    "health": 100.0,
    "armor": 0.0,
    "selected_weapon": 2.0,
    "selected_weapon_ammo": 50.0,
    "ammo2": 50.0,
    "ammo3": 0.0,
    "ammo5": 0.0,
    "ammo6": 0.0,
}


def random_batch(n: int, num_actions: int, seed: int = 0) -> Batch:
    rng = np.random.default_rng(seed)
    shape = (n, 12, 45, 60)
    return Batch(
        states=rng.integers(0, 256, shape, dtype=np.uint8),
        actions=rng.integers(num_actions, size=n),
        rewards=rng.normal(size=n).astype(np.float32),
        next_states=rng.integers(0, 256, shape, dtype=np.uint8),
        dones=(rng.random(n) < 0.1).astype(np.float32),
        discounts=np.full(n, 0.99, dtype=np.float32),
        features=rng.random((n, GAME_FEATURE_DIM), dtype=np.float32),
        next_features=rng.random((n, GAME_FEATURE_DIM), dtype=np.float32),
        aux=(rng.random((n, 2)) < 0.3).astype(np.float32),
    )


def test_network_output_shape():
    net = DuelingQNetwork(num_actions=31)
    x = torch.zeros(5, 12, 45, 60, dtype=torch.uint8)
    assert net(x).shape == (5, 31)


def test_network_with_features_and_aux_head():
    net = DuelingQNetwork(num_actions=31, feature_dim=GAME_FEATURE_DIM, aux_dim=2)
    x = torch.zeros(5, 12, 45, 60, dtype=torch.uint8)
    f = torch.zeros(5, GAME_FEATURE_DIM)
    q, aux = net.q_and_aux(x, f)
    assert q.shape == (5, 31)
    assert aux is not None and aux.shape == (5, 2)
    torch.testing.assert_close(net(x, f), q)
    with pytest.raises(ValueError):
        net(x)


def test_aux_loss_decreases_on_fixed_batch():
    agent = DQNAgent(7, DQNConfig(lr=1e-3), device="cpu")
    batch = random_batch(32, 7)
    first = agent.update(batch)["aux_loss"]
    last = first
    for _ in range(30):
        last = agent.update(batch)["aux_loss"]
    assert last < first


def test_update_reduces_loss_on_fixed_batch():
    agent = DQNAgent(7, DQNConfig(lr=1e-3), device="cpu")
    batch = random_batch(32, 7)
    first = agent.update(batch)["loss"]
    last = first
    for _ in range(30):
        last = agent.update(batch)["loss"]
    assert np.isfinite(first)
    assert last < first


def test_epsilon_schedule():
    agent = DQNAgent(4, DQNConfig(eps_fraction=0.5), device="cpu")
    assert agent.epsilon_at(0.0) == 1.0
    assert agent.epsilon_at(0.25) == pytest.approx(0.525)
    assert agent.epsilon_at(0.5) == pytest.approx(0.05)
    assert agent.epsilon_at(1.0) == pytest.approx(0.05)


def test_lr_schedule():
    agent = DQNAgent(4, DQNConfig(lr=5e-5, lr_end=1e-5), device="cpu")
    assert agent.lr_at(0.0) == pytest.approx(5e-5)
    assert agent.lr_at(0.5) == pytest.approx(3e-5)
    assert agent.lr_at(1.0) == pytest.approx(1e-5)
    agent.set_lr(agent.lr_at(1.0))
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_act_uses_frame_stack_and_is_greedy():
    agent = DQNAgent(5, device="cpu")
    agent.epsilon = 0.0
    obs = np.zeros((240, 320, 3), dtype=np.uint8)
    agent.reset()
    a = agent.act(obs, INFO)
    assert a == agent.act(obs, INFO)
    assert 0 <= a < 5


def test_save_load_roundtrip(tmp_path):
    agent = DQNAgent(6, DQNConfig(hidden=64), device="cpu")
    state = np.random.default_rng(0).integers(0, 256, (12, 45, 60), dtype=np.uint8)
    agent.save(tmp_path / "m.pt", step=3)
    loaded = DQNAgent.load(tmp_path / "m.pt", device="cpu")
    features = agent.features_of(INFO)
    np.testing.assert_allclose(
        agent.q_values(state, features), loaded.q_values(state, features), rtol=1e-6
    )
    assert loaded.config.hidden == 64


def test_load_checkpoint_without_features(tmp_path):
    """Checkpoints saved before game-state input still load as frames-only."""
    agent = DQNAgent(6, DQNConfig(feature_dim=0, aux_dim=0), device="cpu")
    agent.save(tmp_path / "old.pt")
    checkpoint = torch.load(tmp_path / "old.pt", weights_only=True)
    del checkpoint["config"]["feature_dim"], checkpoint["config"]["aux_dim"]
    torch.save(checkpoint, tmp_path / "old.pt")
    loaded = DQNAgent.load(tmp_path / "old.pt", device="cpu")
    assert loaded.config.feature_dim == 0
    assert loaded.features_of(INFO) is None
    loaded.reset()
    assert 0 <= loaded.act(np.zeros((240, 320, 3), dtype=np.uint8), INFO) < 6
