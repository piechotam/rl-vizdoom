"""Metrics collection and statistical summary for ViZDoom evaluation runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

TICS_PER_SECOND = 35


@dataclass
class EpisodeMetrics:
    """Metrics recorded for a single episode."""

    seed: int
    episode: int
    kill_count: int
    died: bool
    timed_out: bool
    survival_time: int  # tics survived
    total_reward: float
    kills_per_minute: float
    damage_taken: float
    hits_taken: int
    shots_fired: int  # ammo spent on the selected weapon
    kills_per_shot: float
    ammo_collected: int
    health_collected: float
    armor_collected: float
    distance_travelled: float  # map units, XY plane
    action_entropy: float  # bits; log2(num_actions) = uniform policy
    final_health: float
    final_armor: float


# Numeric fields used for aggregate statistics, with display labels
_STAT_LABELS = {
    "kill_count": "Kill Count",
    "died": "Death Rate",
    "survival_time": "Survival Time (tics)",
    "total_reward": "Total Reward",
    "kills_per_minute": "Kills / Minute",
    "damage_taken": "Damage Taken",
    "hits_taken": "Hits Taken",
    "shots_fired": "Shots Fired",
    "kills_per_shot": "Kills / Shot",
    "ammo_collected": "Ammo Collected",
    "health_collected": "Health Collected",
    "armor_collected": "Armor Collected",
    "distance_travelled": "Distance Travelled",
    "action_entropy": "Action Entropy (bits)",
    "final_health": "Final Health",
    "final_armor": "Final Armor",
}
_STAT_FIELDS = list(_STAT_LABELS)


def iqm(values: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50% of values."""
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(np.mean(values[lo:hi] if hi > lo else values))


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        v = float(values[0]) if len(values) else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def describe(values: np.ndarray) -> dict[str, float]:
    """Mean, std, min, max, IQM and 95% bootstrap CI of the mean."""
    values = np.asarray(values, dtype=float)
    ci_low, ci_high = bootstrap_ci(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "iqm": iqm(values),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


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

    @property
    def seeds(self) -> list[int]:
        return sorted({m.seed for m in self._episodes})

    def _values(
        self, field_name: str, episodes: list[EpisodeMetrics]
    ) -> np.ndarray:
        return np.array([getattr(m, field_name) for m in episodes], dtype=float)

    def summary(self) -> dict[str, dict[str, float]]:
        """Statistics over all episodes pooled together."""
        if not self._episodes:
            return {}
        return {
            f: describe(self._values(f, self._episodes)) for f in _STAT_FIELDS
        }

    def summary_by_seed(self) -> dict[int, dict[str, dict[str, float]]]:
        """Statistics over episodes, computed separately for each seed."""
        result = {}
        for seed in self.seeds:
            eps = [m for m in self._episodes if m.seed == seed]
            result[seed] = {f: describe(self._values(f, eps)) for f in _STAT_FIELDS}
        return result

    def aggregate_over_seeds(self) -> dict[str, dict[str, float]]:
        """Statistics of per-seed means — the headline multi-seed result.

        Each seed contributes one number (its mean over episodes), so ``std``
        and the CI describe variability between policy seeds.
        """
        by_seed = self.summary_by_seed()
        if not by_seed:
            return {}
        return {
            f: describe(np.array([s[f]["mean"] for s in by_seed.values()]))
            for f in _STAT_FIELDS
        }

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

    def summary_to_json(self, path: str | Path, **metadata) -> None:
        """Write pooled, per-seed and across-seed statistics to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "metadata": {
                **metadata,
                "seeds": self.seeds,
                "num_episodes": len(self._episodes),
            },
            "across_seeds": self.aggregate_over_seeds(),
            "pooled": self.summary(),
            "by_seed": {str(k): v for k, v in self.summary_by_seed().items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self, title: str = "Evaluation Summary") -> None:
        """Print per-seed means and the across-seed aggregate to stdout."""
        if not self._episodes:
            print("No episodes recorded.")
            return

        by_seed = self.summary_by_seed()
        n = len(self._episodes)
        print(f"\n{title} ({n} episodes, {len(by_seed)} seeds)")

        # Per-seed means
        width = 25 + 9 * len(by_seed)
        print("\nPer-seed mean")
        print("─" * width)
        print(f"{'Metric':<25s}" + "".join(f"{f'seed {s}':>9s}" for s in by_seed))
        print("─" * width)
        for key in _STAT_FIELDS:
            row = "".join(f"{s[key]['mean']:9.2f}" for s in by_seed.values())
            print(f"{_STAT_LABELS[key]:<25s}{row}")
        print("─" * width)

        # Aggregate: across-seed mean/std/CI + pooled IQM
        stats = self.aggregate_over_seeds()
        pooled = self.summary()
        print("\nAcross seeds (mean ± std of per-seed means, 95% bootstrap CI; IQM pooled)")
        print("─" * 78)
        print(
            f"{'Metric':<25s} {'Mean':>9s} {'Std':>8s} {'CI low':>9s} {'CI high':>9s} {'IQM':>9s}"
        )
        print("─" * 78)
        for key in _STAT_FIELDS:
            s = stats[key]
            print(
                f"{_STAT_LABELS[key]:<25s} {s['mean']:9.2f} {s['std']:8.2f} "
                f"{s['ci_low']:9.2f} {s['ci_high']:9.2f} {pooled[key]['iqm']:9.2f}"
            )
        print("─" * 78)
