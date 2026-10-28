"""RandomAgent — selects actions uniformly at random."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class RandomAgent:
    """Baseline agent that picks a random action every step.

    ``excluded_actions`` removes actions from the uniform draw, e.g. the attack
    action for a "never shoots" control baseline.
    """

    def __init__(
        self,
        num_actions: int,
        seed: int | None = None,
        excluded_actions: Iterable[int] = (),
    ) -> None:
        self.num_actions = num_actions
        excluded = set(excluded_actions)
        self.allowed_actions = np.array(
            [a for a in range(num_actions) if a not in excluded]
        )
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> int:
        """Return a uniformly random allowed action index."""
        return int(self.rng.choice(self.allowed_actions))

    def reset(self) -> None:
        """Stateless — nothing to reset between episodes."""
