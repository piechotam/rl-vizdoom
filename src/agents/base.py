"""Common agent interface used by the evaluation pipeline."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Agent(Protocol):
    """Anything that maps an observation to a discrete action index."""

    def act(self, observation: np.ndarray, info: dict) -> int:
        """Return the action index for the screen and game variables."""
        ...

    def reset(self) -> None:
        """Called at the start of every episode (e.g. to clear frame stacks)."""
        ...
