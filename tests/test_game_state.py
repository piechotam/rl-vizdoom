"""Tests for game-state features and enemy labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.game_state import GAME_FEATURE_DIM, aux_targets, enemy_flags, game_features


@dataclass
class FakeLabel:
    object_name: str
    x: int
    y: int
    width: int
    height: int


def info(**overrides: float) -> dict:
    base = {
        "health": 100.0,
        "armor": 50.0,
        "selected_weapon": 3.0,
        "selected_weapon_ammo": 25.0,
        "ammo2": 100.0,
        "ammo3": 25.0,
        "ammo5": 0.0,
        "ammo6": 600.0,  # backpack doubles capacity; clipped to 1
    }
    return base | overrides


def test_game_features_are_normalized():
    f = game_features(info())
    assert f.shape == (GAME_FEATURE_DIM,)
    assert f.dtype == np.float32
    assert f[:3].tolist() == pytest.approx([0.5, 0.25, 0.5])  # hp, armor, shells
    assert f[3:10].tolist() == [0, 0, 1, 0, 0, 0, 0]  # shotgun = slot 3
    assert f[10:].tolist() == pytest.approx([0.5, 0.5, 0.0, 1.0])
    assert f.min() >= 0.0 and f.max() <= 1.0


def test_fist_has_no_weapon_ammo():
    f = game_features(info(selected_weapon=1, selected_weapon_ammo=0))
    assert f[2] == 0.0
    assert f[3] == 1.0


def test_enemy_flags():
    w, h = 320, 240
    assert enemy_flags([], w, h) == {"enemy_visible": 0.0, "enemy_in_crosshair": 0.0}
    # the agent itself and items are not enemies
    own = [FakeLabel("DoomPlayer", 140, 100, 40, 40), FakeLabel("Medikit", 0, 0, 9, 9)]
    assert enemy_flags(own, w, h)["enemy_visible"] == 0.0
    side = [FakeLabel("Zombieman", 10, 100, 20, 40)]
    assert enemy_flags(side, w, h) == {"enemy_visible": 1.0, "enemy_in_crosshair": 0.0}
    centre = side + [FakeLabel("Demon", 150, 110, 20, 20)]
    assert enemy_flags(centre, w, h) == {
        "enemy_visible": 1.0,
        "enemy_in_crosshair": 1.0,
    }
    assert aux_targets(enemy_flags(centre, w, h)).tolist() == [1, 1]
