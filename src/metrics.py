"""Metrics collection and statistical summary for ViZDoom evaluation runs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics recorded for a single episode."""

    episode: int
    kill_count: int
    death_count: int
    survival_time: int  # tics survived
    total_reward: float
    final_health: float
    final_armor: float


# Numeric fields used for aggregate statistics
_STAT_FIELDS = [
    "kill_count",
    "death_count",
    "survival_time",
    "total_reward",
    "final_health",
    "final_armor",
]


class MetricsCollector:
    """Collects per-episode metrics and computes aggregate statistics."""

    def __init__(self) -> None:
        self._episodes: list[EpisodeMetrics] = []

    def record(self, metrics: EpisodeMetrics) -> None:
        """Record metrics for a completed episode."""
        self._episodes.append(metrics)

    @property
    def episodes(self) -> list[EpisodeMetrics]:
        return list(self._episodes)

    def summary(self) -> dict[str, dict[str, float]]:
        """Compute mean, std, min, max for each numeric metric."""
        if not self._episodes:
            return {}

        result: dict[str, dict[str, float]] = {}
        for field_name in _STAT_FIELDS:
            values = np.array(
                [getattr(m, field_name) for m in self._episodes], dtype=float
            )
            result[field_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        return result

    def to_csv(self, path: str | Path) -> None:
        """Write all episode metrics to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        field_names = [f.name for f in fields(EpisodeMetrics)]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for ep in self._episodes:
                writer.writerow(asdict(ep))

    def print_summary(self, title: str = "Evaluation Summary") -> None:
        """Print a formatted summary table to stdout."""
        stats = self.summary()
        if not stats:
            print("No episodes recorded.")
            return

        labels = {
            "kill_count": "Kill Count",
            "death_count": "Death Count",
            "survival_time": "Survival Time (tics)",
            "total_reward": "Total Reward",
            "final_health": "Final Health",
            "final_armor": "Final Armor",
        }

        n = len(self._episodes)
        header = f"{title} ({n} episodes)"
        print(f"\n{header}")
        print("─" * 60)
        print(
            f"{'Metric':<25s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}"
        )
        print("─" * 60)
        for key in _STAT_FIELDS:
            s = stats[key]
            label = labels.get(key, key)
            print(
                f"{label:<25s} {s['mean']:8.2f} {s['std']:8.2f} {s['min']:8.1f} {s['max']:8.1f}"
            )
        print("─" * 60)
