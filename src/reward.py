"""Reward shaping for training, computed from game-variable deltas.

The game's own reward only pays for kills, and it also counts monsters killed by
other monsters (only ~9% of a random agent's kills follow its own hit). The
shaped reward adds dense signals and pays for a kill only when the agent hit
something shortly before. Evaluation always uses the original game reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Distinct ammo pools (slots 4 and 7 share the pools of slots 2 and 6) and their
# capacity without a backpack: bullets, shells, rockets, cells
AMMO_POOLS = ("ammo2", "ammo3", "ammo5", "ammo6")
AMMO_MAX = {"ammo2": 200, "ammo3": 50, "ammo5": 50, "ammo6": 300}


@dataclass(frozen=True)
class RewardConfig:
    """Weights of the shaped reward components (before ``scale``)."""

    kill: float = 1.0  # multiplier of the game reward for an attributed kill
    # v1 (damage 0.01, health -0.05, death -5): unavoidable penalties (~-11.5
    # per episode) drowned the combat signal (~+0.6) and nothing was learned.
    # v2 (health -0.01, death -1): learned to fight (1.7 attributed kills vs
    # 0.14 random) but not to survive. v3 (health -0.02, death -3, +0.001 per
    # tic alive) made it shoot half as often without surviving longer, so the
    # v2 weights are back; the survival bonus stays available but off.
    damage_dealt: float = 0.05  # per point of damage dealt
    hit: float = 0.1  # per HITCOUNT increase
    health_lost: float = -0.01  # per HP lost
    alive: float = 0.0  # per tic survived (0.001 = 0.035/s, frame-skip independent)
    missed_shot: float = -0.01  # per ammo spent without a hit
    health_gained: float = 0.02  # per HP healed
    # Fixed reward per ammo pickup, only into a pool that was nearly empty.
    # (A per-unit reward was farmed: items respawn and a cell pack is +100.)
    low_ammo_pickup: float = 0.1
    low_ammo_fraction: float = 0.1  # "nearly empty" = below 10% of capacity
    armor_collected: float = 0.01  # per armor point picked up
    death: float = -1.0
    distance: float = 0.0005  # per map unit travelled
    scale: float = 0.1  # applied to the sum of all components
    attribution_window_tics: int = 70  # 2 s at 35 tics/s


class KillAttribution:
    """Counts kills that follow the agent's own hit within a time window.

    ``KILLCOUNT`` includes monsters killed by other monsters; a kill is
    attributed to the agent when ``HITCOUNT`` increased at most
    ``window_tics`` before it.
    """

    def __init__(self, window_tics: int = 70) -> None:
        self.window_tics = window_tics
        self._last_hit_tic: float = -np.inf

    def reset(self) -> None:
        self._last_hit_tic = -np.inf

    def step(self, prev: dict, info: dict) -> tuple[int, int]:
        """Return ``(new_kills, new_attributed_kills)`` for this step."""
        if info["hitcount"] > prev["hitcount"]:
            self._last_hit_tic = info["episode_time"]
        kills = int(info["killcount"] - prev["killcount"])
        if kills <= 0:
            return 0, 0
        recent = info["episode_time"] - self._last_hit_tic <= self.window_tics
        return kills, kills if recent else 0


def shots_fired(prev: dict, info: dict) -> int:
    """Ammo spent by the selected weapon (0 if the weapon changed)."""
    if info["selected_weapon"] != prev["selected_weapon"]:
        return 0
    return max(0, int(prev["selected_weapon_ammo"] - info["selected_weapon_ammo"]))


def ammo_collected(prev: dict, info: dict) -> float:
    """Ammo picked up across all distinct pools."""
    return float(sum(max(0.0, info[k] - prev[k]) for k in AMMO_POOLS))


def low_ammo_pickups(prev: dict, info: dict, fraction: float) -> int:
    """Pools that gained ammo while below ``fraction`` of their capacity."""
    return sum(
        info[k] > prev[k] and prev[k] < fraction * AMMO_MAX[k] for k in AMMO_POOLS
    )


class RewardShaper:
    """Turns consecutive ``info`` dicts into a shaped training reward."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self._attribution = KillAttribution(self.config.attribution_window_tics)
        self._prev: dict | None = None
        self.attributed_kills = 0  # in the current episode

    def reset(self, info: dict) -> None:
        """Call after ``env.reset()`` with ``env.get_info()``."""
        self._attribution.reset()
        self._prev = info
        self.attributed_kills = 0

    def step(
        self, info: dict, game_reward: float, done: bool
    ) -> tuple[float, dict[str, float]]:
        """Return ``(shaped_reward, components)``; components are unscaled."""
        if self._prev is None:
            raise RuntimeError("RewardShaper.reset() must be called first")
        c = self.config
        prev = self._prev
        dead = bool(info["dead"]) or info["health"] <= 0

        _, attributed = self._attribution.step(prev, info)
        self.attributed_kills += attributed
        hit = info["hitcount"] > prev["hitcount"]
        d_health = info["health"] - prev["health"]

        components = {
            "kill": c.kill * max(0.0, game_reward) if attributed else 0.0,
            "damage_dealt": c.damage_dealt
            * max(0.0, info["damagecount"] - prev["damagecount"]),
            "hit": c.hit * max(0.0, info["hitcount"] - prev["hitcount"]),
            "health_lost": c.health_lost * max(0.0, -d_health),
            "missed_shot": 0.0 if hit else c.missed_shot * shots_fired(prev, info),
            "health_gained": 0.0 if dead else c.health_gained * max(0.0, d_health),
            "ammo_collected": c.low_ammo_pickup
            * low_ammo_pickups(prev, info, c.low_ammo_fraction),
            "armor_collected": c.armor_collected
            * max(0.0, info["armor"] - prev["armor"]),
            "alive": 0.0
            if dead
            else c.alive * max(0.0, info["episode_time"] - prev["episode_time"]),
            "death": c.death if done and dead else 0.0,
            "distance": 0.0
            if dead
            else c.distance
            * float(
                np.hypot(
                    info["position_x"] - prev["position_x"],
                    info["position_y"] - prev["position_y"],
                )
            ),
        }
        self._prev = info
        return c.scale * sum(components.values()), components
