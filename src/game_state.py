"""Game-state features and enemy labels used by the DQN agent.

The HUD is disabled (``render_hud = false``), so the screen does not show the
agent's health or ammo. ``game_features`` turns the game variables into a
small normalized vector fed to the network next to the frames.

``enemy_flags`` reads the engine's labels buffer and gives two binary targets
for an auxiliary loss (as in Arnold, Lample & Chaplot 2017): an enemy is
anywhere in view, and an enemy's bounding box covers the crosshair.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np

from src.reward import AMMO_MAX, AMMO_POOLS

# Monsters seen in deathmatch.wad (measured with a random agent) plus the other
# classic Doom monsters, so a different spawn list is still covered.
# ``DoomPlayer`` is the agent itself and is never an enemy.
ENEMY_NAMES = frozenset(
    {
        "Zombieman",
        "ShotgunGuy",
        "ChaingunGuy",
        "Demon",
        "Spectre",
        "DoomImp",
        "LostSoul",
        "Cacodemon",
        "HellKnight",
        "BaronOfHell",
        "Arachnotron",
        "PainElemental",
        "Revenant",
        "Fatso",
        "Archvile",
        "Cyberdemon",
        "SpiderMastermind",
        "WolfensteinSS",
        "MarineChainsawVzd",
    }
)

# Info keys of the auxiliary targets (added by DoomEnv when labels are enabled)
AUX_KEYS = ("enemy_visible", "enemy_in_crosshair")

NUM_WEAPON_SLOTS = 7
# Ammo pool used by each weapon slot (1 = fist/chainsaw needs none)
_WEAPON_POOL = {2: "ammo2", 3: "ammo3", 4: "ammo2", 5: "ammo5", 6: "ammo6", 7: "ammo6"}
MAX_HEALTH = 200.0  # with bonuses
MAX_ARMOR = 200.0

# health, armor, selected weapon ammo, weapon one-hot, ammo pools
GAME_FEATURE_DIM = 3 + NUM_WEAPON_SLOTS + len(AMMO_POOLS)


class Label(Protocol):
    """The fields of ``vizdoom.Label`` used here."""

    @property
    def object_name(self) -> str: ...
    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...


def game_features(info: dict) -> np.ndarray:
    """Normalized game variables, shape ``(GAME_FEATURE_DIM,)``, in ``[0, 1]``."""
    weapon = int(info["selected_weapon"])
    pool = _WEAPON_POOL.get(weapon)
    weapon_ammo = info["selected_weapon_ammo"] / AMMO_MAX[pool] if pool else 0.0
    one_hot = np.zeros(NUM_WEAPON_SLOTS, dtype=np.float32)
    if 1 <= weapon <= NUM_WEAPON_SLOTS:
        one_hot[weapon - 1] = 1.0
    values = np.concatenate(
        [
            [
                info["health"] / MAX_HEALTH,
                info["armor"] / MAX_ARMOR,
                weapon_ammo,
            ],
            one_hot,
            [info[k] / AMMO_MAX[k] for k in AMMO_POOLS],
        ]
    )
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def enemy_flags(labels: Iterable[Label], width: int, height: int) -> dict[str, float]:
    """``enemy_visible`` and ``enemy_in_crosshair`` (0/1) from screen labels."""
    cx, cy = width / 2, height / 2
    visible = crosshair = False
    for label in labels:
        if label.object_name not in ENEMY_NAMES:
            continue
        visible = True
        if (
            label.x <= cx < label.x + label.width
            and label.y <= cy < label.y + label.height
        ):
            crosshair = True
            break
    return {"enemy_visible": float(visible), "enemy_in_crosshair": float(crosshair)}


def aux_targets(info: dict) -> np.ndarray:
    """Auxiliary targets from ``info``, shape ``(len(AUX_KEYS),)`` uint8."""
    return np.array([info[k] for k in AUX_KEYS], dtype=np.uint8)
