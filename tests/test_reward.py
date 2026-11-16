"""Tests for reward shaping on synthetic game-variable dicts."""

from __future__ import annotations

from typing import Any

import pytest

from src.reward import KillAttribution, RewardConfig, RewardShaper


def info(**overrides: float) -> dict:
    base = {
        "killcount": 0.0,
        "hitcount": 0.0,
        "damagecount": 0.0,
        "health": 100.0,
        "armor": 0.0,
        "dead": 0.0,
        "selected_weapon": 2.0,
        "selected_weapon_ammo": 50.0,
        "ammo2": 50.0,
        "ammo3": 0.0,
        "ammo5": 0.0,
        "ammo6": 0.0,
        "position_x": 0.0,
        "position_y": 0.0,
        "episode_time": 1.0,
    }
    return base | overrides


def shaper(**weights: Any) -> RewardShaper:
    s = RewardShaper(RewardConfig(scale=1.0, **weights))
    s.reset(info())
    return s


def test_kill_without_own_hit_is_not_rewarded():
    _, c = shaper().step(info(killcount=1, episode_time=10), 1.0, False)
    assert c["kill"] == 0.0


def test_kill_after_own_hit_is_rewarded():
    s = shaper()
    s.step(info(hitcount=1, damagecount=10, episode_time=10), 0.0, False)
    _, c = s.step(
        info(hitcount=1, damagecount=10, killcount=1, episode_time=50), 2.0, False
    )
    assert c["kill"] == 2.0
    assert s.attributed_kills == 1


def test_kill_attribution_window_expires():
    a = KillAttribution(window_tics=70)
    prev = info()
    hit = info(hitcount=1, episode_time=10)
    assert a.step(prev, hit) == (0, 0)
    assert a.step(hit, info(hitcount=1, killcount=1, episode_time=81)) == (1, 0)


def test_damage_and_missed_shots():
    s = shaper()
    _, c = s.step(info(selected_weapon_ammo=49, ammo2=49), 0.0, False)
    assert c["missed_shot"] == pytest.approx(-0.01)
    _, c = s.step(
        info(selected_weapon_ammo=48, ammo2=48, hitcount=1, damagecount=15), 0.0, False
    )
    assert c["missed_shot"] == 0.0
    assert c["damage_dealt"] == pytest.approx(0.75)
    assert c["hit"] == pytest.approx(0.1)


def test_pickups_health_and_death():
    s = shaper()
    _, c = s.step(info(health=80, armor=100, ammo3=8), 0.0, False)
    assert c["health_lost"] == pytest.approx(-0.2)
    assert c["armor_collected"] == pytest.approx(1.0)
    assert c["ammo_collected"] == pytest.approx(0.1)  # shells were empty
    _, c = s.step(info(health=100, armor=100, ammo3=8), 0.0, False)
    assert c["health_gained"] == pytest.approx(0.4)
    _, c = s.step(info(health=0, armor=100, ammo3=8, dead=1), 0.0, True)
    assert c["death"] == -1.0
    assert c["health_lost"] == pytest.approx(-1.0)
    assert c["alive"] == 0.0


def test_ammo_rewarded_only_when_nearly_empty():
    s = RewardShaper(RewardConfig(scale=1.0))
    s.reset(info(ammo2=50, ammo6=0))
    # bullets at 50/200 are not low; cells at 0/300 are
    _, c = s.step(info(ammo2=60, ammo6=100), 0.0, False)
    assert c["ammo_collected"] == pytest.approx(0.1)
    # cells now at 100/300: another pack is not rewarded
    _, c = s.step(info(ammo2=60, ammo6=200), 0.0, False)
    assert c["ammo_collected"] == 0.0


def test_scale_and_distance():
    s = RewardShaper(RewardConfig(scale=0.1))
    s.reset(info())
    reward, c = s.step(info(position_x=300, position_y=400), 0.0, False)
    assert c["distance"] == pytest.approx(500 * 0.0005)
    assert reward == pytest.approx(0.1 * sum(c.values()))


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        RewardShaper().step(info(), 0.0, False)


def test_alive_bonus_counts_tics():
    s = shaper(alive=0.001)
    _, c = s.step(info(episode_time=9), 0.0, False)
    assert c["alive"] == pytest.approx(8 * 0.001)
