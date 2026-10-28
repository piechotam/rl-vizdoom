"""RandomAgent — selects actions uniformly at random."""

from __future__ import annotations

import numpy as np


class RandomAgent:
    """Baseline agent that picks a random action every step."""

    def __init__(self, num_actions: int, seed: int | None = None) -> None:
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> int:
        """Return a uniformly random action index."""
        return int(self.rng.integers(self.num_actions))

    def reset(self) -> None:
        """Stateless — nothing to reset between episodes."""
