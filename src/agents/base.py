"""Common agent interface used by the evaluation pipeline."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Agent(Protocol):
    """Anything that maps an observation to a discrete action index."""

    def act(self, observation: np.ndarray) -> int:
        """Return the action index for the given observation."""
        ...

    def reset(self) -> None:
        """Called at the start of every episode (e.g. to clear frame stacks)."""
        ...
