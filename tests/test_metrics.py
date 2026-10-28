"""Tests for statistics in src/metrics.py — no ViZDoom required."""

from __future__ import annotations

import csv
import json
from dataclasses import fields, replace
from typing import Any

import numpy as np
import pytest

from src.metrics import EpisodeMetrics, MetricsCollector, bootstrap_ci, describe, iqm


def make_episode(seed: int, episode: int, kill_count: int) -> EpisodeMetrics:
    defaults: dict[str, Any] = {f.name: 0 for f in fields(EpisodeMetrics)}
    defaults.update(died=True, timed_out=False)
    return EpisodeMetrics(
        **{**defaults, "seed": seed, "episode": episode, "kill_count": kill_count}
    )


class TestIqm:
    def test_ignores_outer_quartiles(self):
        assert iqm(np.array([0, 1, 2, 3, 4, 5, 6, 100])) == pytest.approx(3.5)

    def test_constant(self):
        assert iqm(np.full(10, 7.0)) == pytest.approx(7.0)

    def test_robust_to_outlier(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])
        assert iqm(values) < np.mean(values)

    def test_single_value(self):
        assert iqm(np.array([5.0])) == pytest.approx(5.0)


class TestBootstrapCi:
    def test_contains_mean(self):
        values = np.random.default_rng(0).normal(10, 2, size=200)
        low, high = bootstrap_ci(values)
        assert low < np.mean(values) < high

    def test_narrows_with_more_data(self):
        rng = np.random.default_rng(0)
        small = bootstrap_ci(rng.normal(0, 1, size=20))
        large = bootstrap_ci(rng.normal(0, 1, size=2000))
        assert large[1] - large[0] < small[1] - small[0]

    def test_deterministic(self):
        values = np.arange(50, dtype=float)
        assert bootstrap_ci(values) == bootstrap_ci(values)

    def test_constant_values_collapse(self):
        assert bootstrap_ci(np.full(10, 3.0)) == (3.0, 3.0)

    def test_single_value(self):
        assert bootstrap_ci(np.array([4.0])) == (4.0, 4.0)


def test_describe_keys_and_values():
    stats = describe(np.array([1.0, 2.0, 3.0]))
    assert set(stats) == {"mean", "std", "min", "max", "iqm", "ci_low", "ci_high"}
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
    assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]


class TestMetricsCollector:
    @pytest.fixture
    def collector(self) -> MetricsCollector:
        c = MetricsCollector()
        # seed 0: kills 0, 2 (mean 1); seed 1: kills 4, 4, 4, 4 (mean 4)
        for ep, k in enumerate([0, 2], start=1):
            c.record(make_episode(0, ep, k))
        for ep in range(1, 5):
            c.record(make_episode(1, ep, 4))
        return c

    def test_empty(self):
        c = MetricsCollector()
        assert c.summary() == {}
        assert c.summary_by_seed() == {}
        assert c.aggregate_over_seeds() == {}

    def test_seeds(self, collector):
        assert collector.seeds == [0, 1]

    def test_summary_by_seed(self, collector):
        by_seed = collector.summary_by_seed()
        assert by_seed[0]["kill_count"]["mean"] == pytest.approx(1.0)
        assert by_seed[1]["kill_count"]["mean"] == pytest.approx(4.0)

    def test_aggregate_weights_seeds_equally(self, collector):
        # Mean of per-seed means (1 + 4) / 2, not pooled mean 18 / 6
        agg = collector.aggregate_over_seeds()["kill_count"]
        assert agg["mean"] == pytest.approx(2.5)
        assert agg["std"] == pytest.approx(1.5)
        assert collector.summary()["kill_count"]["mean"] == pytest.approx(3.0)

    def test_bool_metric_is_rate(self, collector):
        collector.record(replace(make_episode(0, 3, 0), died=False))
        assert collector.summary_by_seed()[0]["died"]["mean"] == pytest.approx(2 / 3)

    def test_to_csv(self, collector, tmp_path):
        path = tmp_path / "out" / "episodes.csv"
        collector.to_csv(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6
        assert list(rows[0]) == [f.name for f in fields(EpisodeMetrics)]

    def test_summary_to_json(self, collector, tmp_path):
        path = tmp_path / "summary.json"
        collector.summary_to_json(path, agent="random")
        data = json.loads(path.read_text())
        assert data["metadata"] == {
            "agent": "random",
            "seeds": [0, 1],
            "num_episodes": 6,
        }
        assert set(data["by_seed"]) == {"0", "1"}
        assert data["across_seeds"]["kill_count"]["mean"] == pytest.approx(2.5)
