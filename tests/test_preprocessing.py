"""Tests for frame preprocessing and stacking."""

from __future__ import annotations

import numpy as np

from src.preprocessing import FrameStack, preprocess


def test_preprocess_shape_and_dtype():
    frame = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    out = preprocess(frame)
    assert out.shape == (3, 45, 60)
    assert out.dtype == np.uint8


def test_preprocess_keeps_colour():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[..., 0] = 200
    out = preprocess(frame)
    assert (out[0] == 200).all()
    assert (out[1:] == 0).all()


def test_frame_stack():
    stack = FrameStack(k=4)
    frames = [np.full((3, 2, 2), i, dtype=np.uint8) for i in range(6)]
    state = stack.reset(frames[0])
    assert state.shape == (12, 2, 2)
    assert (state == 0).all()
    for f in frames[1:]:
        state = stack.push(f)
    # oldest first: frames 2, 3, 4, 5
    assert [int(state[3 * i, 0, 0]) for i in range(4)] == [2, 3, 4, 5]
