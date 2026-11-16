"""Double DQN agent with a dueling network."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.dqn.network import DuelingQNetwork
from src.agents.dqn.replay import Batch
from src.game_state import AUX_KEYS, GAME_FEATURE_DIM, game_features
from src.preprocessing import FrameStack, preprocess


@dataclass
class DQNConfig:
    gamma: float = 0.99
    n_step: int = 3  # multi-step returns propagate sparse kill rewards faster
    lr: float = 5e-5
    lr_end: float = 1e-5  # linear decay over training, against late forgetting
    batch_size: int = 64
    buffer_size: int = 400_000
    learning_starts: int = 20_000  # agent steps before the first update
    train_freq: int = 4  # agent steps between updates
    target_update: int = 10_000  # agent steps between target syncs
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.25  # fraction of training to anneal epsilon
    eval_epsilon: float = 0.05
    grad_clip: float = 10.0
    stack_size: int = 4
    hidden: int = 512
    feature_dim: int = GAME_FEATURE_DIM  # game-state input (0 = frames only)
    aux_dim: int = len(AUX_KEYS)  # auxiliary enemy-detection targets (0 = off)
    aux_weight: float = 1.0  # weight of the auxiliary BCE loss


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DQNAgent:
    """Double DQN (van Hasselt et al. 2016) with a dueling Q-network.

    ``select_action`` works on preprocessed frame stacks and game-state
    features (used in training). ``act``/``reset`` implement the ``Agent``
    protocol on raw screens and ``info`` with an internal frame stack and
    ``epsilon`` exploration, so the agent plugs into
    ``src.evaluation.evaluate`` unchanged.
    """

    def __init__(
        self,
        num_actions: int,
        config: DQNConfig | None = None,
        device: torch.device | str | None = None,
        seed: int = 0,
    ) -> None:
        self.num_actions = num_actions
        self.config = config or DQNConfig()
        self.device = torch.device(device) if device else default_device()
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        c = self.config
        self.online = DuelingQNetwork(
            num_actions, c.stack_size, c.hidden, c.feature_dim, c.aux_dim
        ).to(self.device)
        self.target = DuelingQNetwork(
            num_actions, c.stack_size, c.hidden, c.feature_dim, c.aux_dim
        ).to(self.device)
        self.sync_target()
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=c.lr)

        self.epsilon = c.eval_epsilon
        self._stack = FrameStack(c.stack_size)
        self._episode_started = False

    # --- acting -----------------------------------------------------------

    def features_of(self, info: dict) -> np.ndarray | None:
        """Network input built from game variables (``None`` if unused)."""
        return game_features(info) if self.config.feature_dim else None

    @torch.no_grad()
    def q_values(
        self, state: np.ndarray, features: np.ndarray | None = None
    ) -> np.ndarray:
        x = torch.as_tensor(state, device=self.device).unsqueeze(0)
        f = (
            torch.as_tensor(features, device=self.device).unsqueeze(0)
            if features is not None
            else None
        )
        return self.online(x, f).squeeze(0).cpu().numpy()

    def select_action(
        self, state: np.ndarray, features: np.ndarray | None, epsilon: float
    ) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.num_actions))
        return int(np.argmax(self.q_values(state, features)))

    def reset(self) -> None:
        self._episode_started = False

    def act(self, observation: np.ndarray, info: dict) -> int:
        frame = preprocess(observation)
        if self._episode_started:
            state = self._stack.push(frame)
        else:
            state = self._stack.reset(frame)
            self._episode_started = True
        return self.select_action(state, self.features_of(info), self.epsilon)

    # --- learning ---------------------------------------------------------

    def epsilon_at(self, progress: float) -> float:
        """Linear epsilon schedule; ``progress`` is the fraction of training done."""
        c = self.config
        frac = min(1.0, progress / c.eps_fraction) if c.eps_fraction > 0 else 1.0
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def lr_at(self, progress: float) -> float:
        """Linear learning-rate decay from ``lr`` to ``lr_end`` over training."""
        c = self.config
        return c.lr + min(1.0, max(0.0, progress)) * (c.lr_end - c.lr)

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def update(self, batch: Batch) -> dict[str, float]:
        """One gradient step on the n-step Double DQN TD error (Huber loss)."""
        c = self.config
        states = torch.as_tensor(batch.states, device=self.device)
        next_states = torch.as_tensor(batch.next_states, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        dones = torch.as_tensor(batch.dones, device=self.device)
        discounts = torch.as_tensor(batch.discounts, device=self.device)
        features = next_features = None
        if c.feature_dim:
            features = torch.as_tensor(batch.features, device=self.device)
            next_features = torch.as_tensor(batch.next_features, device=self.device)

        q_all, aux_logits = self.online.q_and_aux(states, features)
        q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online net picks the action, target net evaluates it
            next_actions = self.online(next_states, next_features).argmax(
                dim=1, keepdim=True
            )
            next_q = (
                self.target(next_states, next_features)
                .gather(1, next_actions)
                .squeeze(1)
            )
            target = rewards + discounts * (1.0 - dones) * next_q

        td_loss = F.smooth_l1_loss(q, target)
        loss = td_loss
        stats: dict[str, float] = {}
        if aux_logits is not None:
            aux_targets = torch.as_tensor(batch.aux, device=self.device)
            aux_loss = F.binary_cross_entropy_with_logits(aux_logits, aux_targets)
            loss = td_loss + c.aux_weight * aux_loss
            with torch.no_grad():
                correct = ((aux_logits > 0).float() == aux_targets).float().mean(0)
            stats["aux_loss"] = float(aux_loss.item())
            for i, key in enumerate(AUX_KEYS[: c.aux_dim]):
                stats[f"aux_acc_{key}"] = float(correct[i].item())
                stats[f"aux_rate_{key}"] = float(aux_targets[:, i].mean().item())

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), c.grad_clip
        )
        self.optimizer.step()
        return {
            "loss": float(td_loss.item()),
            "q_mean": float(q.mean().item()),
            "grad_norm": float(grad_norm.item()),
            **stats,
        }

    # --- persistence ------------------------------------------------------

    def save(self, path: str | Path, **metadata: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.online.state_dict(),
                "num_actions": self.num_actions,
                "config": asdict(self.config),
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device | str | None = None,
        seed: int = 0,
    ) -> DQNAgent:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        # checkpoints from before game-state input / aux head had neither
        config = {"feature_dim": 0, "aux_dim": 0, **checkpoint["config"]}
        agent = cls(
            checkpoint["num_actions"],
            DQNConfig(**config),
            device=device,
            seed=seed,
        )
        agent.online.load_state_dict(checkpoint["model"])
        agent.sync_target()
        return agent
