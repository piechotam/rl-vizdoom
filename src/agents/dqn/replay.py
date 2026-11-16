"""Experience replay storing single frames and rebuilding frame stacks on demand.

Storing each 3x45x60 frame once (instead of two 12-channel stacks per
transition) cuts memory ~8x: 200k transitions take ~1.6 GB.

Slot ``i`` holds the newest frame of state ``s_i`` and the transition taken
from it: ``(a_i, r_i, done_i)``. The next state ``s_{i+1}`` is rebuilt from slot
``i + 1``; it is ignored by the TD target when ``done_i``. Stacks never cross
an episode start: missing older frames repeat the episode's first frame, which
matches ``FrameStack.reset``.

With ``n_step > 1`` a sample from slot ``i`` sums the discounted rewards of up
to ``n`` consecutive transitions and bootstraps from the state after the last
one; the sum stops early at the end of an episode or at the newest transition.

Each slot may also hold a game-state feature vector and auxiliary targets
(``feature_dim`` / ``aux_dim`` > 0); both describe the slot's newest frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.preprocessing import FRAME_CHANNELS, FRAME_HEIGHT, FRAME_WIDTH


@dataclass
class Batch:
    states: np.ndarray  # (B, stack * 3, 45, 60) uint8
    actions: np.ndarray  # (B,) int64
    rewards: np.ndarray  # (B,) float32
    next_states: np.ndarray  # (B, stack * 3, 45, 60) uint8
    dones: np.ndarray  # (B,) float32
    discounts: np.ndarray  # (B,) float32, gamma ** steps summed into the reward
    features: np.ndarray  # (B, feature_dim) float32, game state of ``states``
    next_features: np.ndarray  # (B, feature_dim) float32
    aux: np.ndarray  # (B, aux_dim) float32, auxiliary targets of ``states``


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        stack_size: int = 4,
        seed: int = 0,
        n_step: int = 1,
        gamma: float = 0.99,
        feature_dim: int = 0,
        aux_dim: int = 0,
    ) -> None:
        if capacity <= stack_size + n_step:
            raise ValueError("capacity must exceed stack_size + n_step")
        if n_step < 1:
            raise ValueError("n_step must be at least 1")
        self.capacity = capacity
        self.stack_size = stack_size
        self.n_step = n_step
        self.gamma = gamma
        self.frames = np.zeros(
            (capacity, FRAME_CHANNELS, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8
        )
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.features = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.aux = np.zeros((capacity, aux_dim), dtype=np.uint8)
        self.starts = np.zeros(capacity, dtype=bool)  # first frame of an episode
        self.has_transition = np.zeros(capacity, dtype=bool)
        self.head = 0  # next slot to write
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Number of transitions that can be sampled."""
        return int(self.has_transition[: self.size].sum())

    def add_frame(
        self,
        frame: np.ndarray,
        episode_start: bool,
        features: np.ndarray | None = None,
        aux: np.ndarray | None = None,
    ) -> None:
        """Store the newest frame of the current state (and its extras)."""
        self.frames[self.head] = frame
        if self.features.shape[1]:
            if features is None:
                raise ValueError("buffer stores features: pass them")
            self.features[self.head] = features
        if self.aux.shape[1]:
            if aux is None:
                raise ValueError("buffer stores auxiliary targets: pass them")
            self.aux[self.head] = aux
        self.starts[self.head] = episode_start
        self.has_transition[self.head] = False
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_transition(self, action: int, reward: float, done: bool) -> None:
        """Record the transition taken from the most recently added frame."""
        i = (self.head - 1) % self.capacity
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.has_transition[i] = True

    def _stack(self, idx: np.ndarray) -> np.ndarray:
        """Stacks ending at slots ``idx``, oldest frame first."""
        cur = idx.copy()
        slots = [cur]
        for _ in range(self.stack_size - 1):
            prev = (cur - 1) % self.capacity
            cur = np.where(self.starts[cur], cur, prev)
            slots.append(cur)
        frames = [self.frames[s] for s in reversed(slots)]
        return np.concatenate(frames, axis=1)

    def _has_next(self, idx: np.ndarray) -> np.ndarray:
        """Slots holding a transition whose next frame is written (or terminal)."""
        nxt = (idx + 1) % self.capacity
        return self.has_transition[idx] & (self.dones[idx] | (nxt != self.head))

    def _valid(self, idx: np.ndarray) -> np.ndarray:
        valid = self._has_next(idx)
        if self.size == self.capacity:
            # the oldest slots may stack frames that were already overwritten
            age = (idx - self.head) % self.capacity
            valid &= age >= self.stack_size - 1
        return valid

    def sample(self, batch_size: int) -> Batch:
        if len(self) == 0:
            raise ValueError("replay buffer is empty")
        idx = self.rng.integers(self.size, size=batch_size)
        invalid = ~self._valid(idx)
        while invalid.any():
            idx[invalid] = self.rng.integers(self.size, size=int(invalid.sum()))
            invalid = ~self._valid(idx)
        returns, last, dones, discounts = self._n_step(idx)
        boot = (last + 1) % self.capacity
        return Batch(
            states=self._stack(idx),
            actions=self.actions[idx],
            rewards=returns,
            next_states=self._stack(boot),
            dones=dones.astype(np.float32),
            discounts=discounts,
            features=self.features[idx],
            next_features=self.features[boot],
            aux=self.aux[idx].astype(np.float32),
        )

    def _n_step(
        self, idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Discounted n-step returns from ``idx``.

        Returns ``(returns, last, dones, discounts)``: ``last`` is the slot of the
        final summed transition, so bootstrapping uses the state at ``last + 1``
        discounted by ``discounts``.
        """
        returns = self.rewards[idx].astype(np.float64)
        last = idx.copy()
        dones = self.dones[idx].copy()
        discounts = np.full(idx.shape, self.gamma)
        for _ in range(self.n_step - 1):
            nxt = (last + 1) % self.capacity
            # the next slot belongs to the same episode unless this one ended
            extend = ~dones & self._has_next(nxt)
            if not extend.any():
                break
            returns += np.where(extend, discounts * self.rewards[nxt], 0.0)
            last = np.where(extend, nxt, last)
            dones |= extend & self.dones[nxt]
            discounts = np.where(extend, discounts * self.gamma, discounts)
        return (
            returns.astype(np.float32),
            last,
            dones,
            discounts.astype(np.float32),
        )
