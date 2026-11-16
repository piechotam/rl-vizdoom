"""Observation preprocessing: downscaling and frame stacking."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

FRAME_HEIGHT = 45
FRAME_WIDTH = 60
FRAME_CHANNELS = 3


def preprocess(frame: np.ndarray) -> np.ndarray:
    """``(H, W, 3)`` RGB screen -> ``(3, 45, 60)`` uint8, channels first."""
    small = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(np.transpose(small, (2, 0, 1)), dtype=np.uint8)


class FrameStack:
    """Keeps the last ``k`` preprocessed frames, concatenated along channels.

    After ``reset`` the stack is filled with copies of the first frame, so the
    output always has ``k * 3`` channels.
    """

    def __init__(self, k: int = 4) -> None:
        self.k = k
        self._frames: deque[np.ndarray] = deque(maxlen=k)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """Start a new episode from an already preprocessed frame."""
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(frame)
        return self.state()

    def push(self, frame: np.ndarray) -> np.ndarray:
        """Append a preprocessed frame and return the new stacked state."""
        self._frames.append(frame)
        return self.state()

    def state(self) -> np.ndarray:
        return np.concatenate(self._frames, axis=0)
