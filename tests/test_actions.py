"""Tests for the action space definition — no game instance required."""

from __future__ import annotations

import pytest
import vizdoom as vzd

from src.actions import ACTION_NAMES, ACTIONS, build_button_vectors

BUTTONS = [vzd.Button.ATTACK, vzd.Button.SPEED, vzd.Button.TURN_LEFT]


def test_actions_are_unique():
    assert len(ACTIONS) == 31
    assert len({frozenset(a) for a in ACTIONS}) == len(ACTIONS)
    assert ACTION_NAMES[0] == "NOOP"
    assert "SPEED+MOVE_FORWARD+ATTACK" in ACTION_NAMES


def test_build_button_vectors():
    vectors = build_button_vectors(BUTTONS, [(), ("ATTACK",), ("SPEED", "TURN_LEFT")])
    assert vectors == [[0, 0, 0], [1, 0, 0], [0, 1, 1]]


def test_unknown_button_raises():
    with pytest.raises(ValueError, match="MOVE_FORWARD"):
        build_button_vectors(BUTTONS, [("MOVE_FORWARD",)])
