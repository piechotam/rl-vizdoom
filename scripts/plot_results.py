"""Plot learning curves and evaluation comparisons for the presentation.

Reads ``results/`` (written by ``train.py`` and ``evaluate.py``) and saves PDF +
PNG figures to ``docs/presentations/img/``. Missing inputs are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.metrics import TICS_PER_SECOND

# Presentation palette (validated for CVD separation on the slide background)
INK = "#1F2328"
MUTED = "#646B75"
FAINT = "#D7D9DD"
PAPER = "#FAFAF8"
FS_COLORS = {2: "#3F6FB5", 4: "#D9480F", 8: "#1A9E8F"}
DQN_COLOR = FS_COLORS[4]
RANDOM_COLOR = "#646B75"
NO_ATTACK_COLOR = "#A9AEB6"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": FAINT,
        "axes.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": FAINT,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
    }
)


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save(fig: Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / name}.pdf/.png")


def fs_dirs(results: Path) -> dict[int, Path]:
    dirs = {}
    for d in sorted((results / "dqn").glob("fs*")):
        if d.name[2:].isdigit():
            dirs[int(d.name[2:])] = d
    return dirs


def baseline_mean(results: Path, agent: str, metric: str) -> float | None:
    s = read_summary(results / agent / "summary.json")
    return None if s is None else s["across_seeds"][metric]["mean"]


# --- learning curves -------------------------------------------------------

CURVE_METRICS = {
    "attributed_kills": ("Zabójstwa przypisane agentowi", "attributed_kills", 1.0),
    "survival_s": ("Przeżycie [s]", "survival_time", 1 / TICS_PER_SECOND),
    "kills": ("Wszystkie zabójstwa (KILLCOUNT)", "kill_count", 1.0),
}


def plot_learning_curves(results: Path, out: Path) -> None:
    runs = {
        fs: [read_csv(p) for p in sorted(d.glob("seed*/eval.csv"))]
        for fs, d in fs_dirs(results).items()
    }
    runs = {fs: r for fs, r in runs.items() if r}
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    for ax, (key, (title, base_key, base_scale)) in zip(
        axes, CURVE_METRICS.items(), strict=True
    ):
        for fs, seeds in runs.items():
            n = min(len(s["frames"]) for s in seeds)
            x = seeds[0]["frames"][:n] / 1e6
            ys = np.stack([s[key][:n] for s in seeds])
            color = FS_COLORS.get(fs, INK)
            ax.plot(x, ys.mean(0), color=color, lw=2, label=f"DQN, frame skip {fs}")
            if len(seeds) > 1:
                ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.15, lw=0)
        for agent, color, ls, label in (
            ("random", RANDOM_COLOR, "--", "losowy"),
            ("no_attack", NO_ATTACK_COLOR, ":", "bez strzelania"),
        ):
            value = baseline_mean(results, agent, base_key)
            if value is not None:
                ax.axhline(value * base_scale, color=color, ls=ls, lw=1.2, label=label)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("klatki gry [mln]")
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.1),
    )
    fig.tight_layout()
    save(fig, out, "learning_curves")


def plot_train_reward(results: Path, out: Path, window: int = 100) -> None:
    runs = {
        fs: [read_csv(p) for p in sorted(d.glob("seed*/train_episodes.csv"))]
        for fs, d in fs_dirs(results).items()
    }
    runs = {fs: r for fs, r in runs.items() if r}
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(5, 3.0))
    grid = np.linspace(0, max(s["frames"].max() for r in runs.values() for s in r), 200)
    for fs, seeds in runs.items():
        curves = []
        for s in seeds:
            r = s["shaped_reward"]
            k = min(window, len(r))
            smooth = np.convolve(r, np.ones(k) / k, mode="valid")
            curves.append(np.interp(grid, s["frames"][k - 1 :], smooth, right=np.nan))
        ys = np.stack(curves)
        color = FS_COLORS.get(fs, INK)
        ax.plot(grid / 1e6, np.nanmean(ys, 0), color=color, lw=2, label=f"FS {fs}")
        if len(seeds) > 1:
            ax.fill_between(
                grid / 1e6,
                np.nanmin(ys, 0),
                np.nanmax(ys, 0),
                color=color,
                alpha=0.15,
                lw=0,
            )
    ax.set_title(
        f"Nagroda kształtowana (średnia krocząca, {window} epizodów)",
        fontsize=9,
        color=INK,
        loc="left",
    )
    ax.set_xlabel("klatki gry [mln]")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, out, "train_reward")


# --- TensorBoard-style grids (eval/*, train/*) -----------------------------


def ema(y: np.ndarray, weight: float = 0.6) -> np.ndarray:
    """Exponential moving average, as TensorBoard's smoothing slider."""
    out = np.empty(len(y))
    last = y[0]
    for i, v in enumerate(y):
        last = weight * last + (1 - weight) * v
        out[i] = last
    return out


def rolling(r: np.ndarray, window: int) -> tuple[np.ndarray, int]:
    k = min(window, len(r))
    return np.convolve(r, np.ones(k) / k, mode="valid"), k


def grid_figure(n: int, ncols: int) -> tuple[Figure, list]:
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.4 * nrows))
    axes = list(np.ravel(axes))
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def finish_grid(fig: Figure, axes: list, name: str, out: Path) -> None:
    for ax in axes:
        ax.set_xlabel("klatki gry [mln]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, name)


def accuracy(s: dict[str, np.ndarray]) -> np.ndarray:
    return np.divide(
        s["hits"] * 100, s["shots"], out=np.zeros_like(s["hits"]), where=s["shots"] > 0
    )


EVAL_PANELS = {
    "Zabójstwa przypisane agentowi": lambda s: s["attributed_kills"],
    "Wszystkie zabójstwa": lambda s: s["kills"],
    "Przeżycie [s]": lambda s: s["survival_s"],
    "Nagroda z gry": lambda s: s["game_reward"],
    "Trafienia": lambda s: s["hits"],
    "Strzały": lambda s: s["shots"],
    "Celność [%]": accuracy,
    "Entropia akcji": lambda s: s["action_entropy"],
    "Odsetek epizodów do limitu czasu": lambda s: s["timeout_rate"],
}


def plot_eval_metrics(results: Path, out: Path) -> None:
    """Every ``eval/*`` scalar: raw (thin) and EMA-smoothed (bold)."""
    runs = {
        fs: [read_csv(p) for p in sorted(d.glob("seed*/eval.csv"))]
        for fs, d in fs_dirs(results).items()
    }
    runs = {fs: r for fs, r in runs.items() if r}
    if not runs:
        return
    fig, axes = grid_figure(len(EVAL_PANELS), 3)
    for ax, (title, get) in zip(axes, EVAL_PANELS.items(), strict=True):
        for fs, seeds in runs.items():
            n = min(len(s["frames"]) for s in seeds)
            x = seeds[0]["frames"][:n] / 1e6
            y = np.stack([get(s)[:n] for s in seeds]).mean(0)
            color = FS_COLORS.get(fs, INK)
            ax.plot(x, y, color=color, lw=0.6, alpha=0.3)
            ax.plot(x, ema(y), color=color, lw=1.8, label=f"FS {fs}")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    finish_grid(fig, axes, "eval_metrics", out)


TRAIN_PANELS = {
    "shaped_reward": "Nagroda kształtowana",
    "game_reward": "Nagroda z gry",
    "attributed_kills": "Zabójstwa przypisane agentowi",
    "kills": "Wszystkie zabójstwa",
    "survival_s": "Przeżycie [s]",
    "epsilon": "Epsilon (eksploracja)",
}

REWARD_PANELS = {
    "r_kill": "Zabójstwo",
    "r_damage_dealt": "Zadane obrażenia",
    "r_hit": "Trafienie",
    "r_missed_shot": "Chybiony strzał",
    "r_health_lost": "Utrata zdrowia",
    "r_death": "Śmierć",
    "r_health_gained": "Zebrane zdrowie",
    "r_armor_collected": "Zebrany pancerz",
    "r_ammo_collected": "Zebrana amunicja",
    "r_distance": "Przebyty dystans",
}


def plot_train_grid(
    results: Path,
    out: Path,
    panels: dict[str, str],
    name: str,
    ncols: int,
    window: int = 100,
) -> None:
    """Per-episode ``train/*`` scalars, rolling mean over ``window`` episodes."""
    runs = {
        fs: [read_csv(p) for p in sorted(d.glob("seed*/train_episodes.csv"))]
        for fs, d in fs_dirs(results).items()
    }
    runs = {fs: r for fs, r in runs.items() if r}
    if not runs:
        return
    fig, axes = grid_figure(len(panels), ncols)
    for ax, (key, title) in zip(axes, panels.items(), strict=True):
        for fs, seeds in runs.items():
            grid = np.linspace(0, max(s["frames"].max() for s in seeds), 300)
            curves = []
            for s in seeds:
                smooth, k = rolling(s[key], window)
                curves.append(
                    np.interp(grid, s["frames"][k - 1 :], smooth, right=np.nan)
                )
            ax.plot(
                grid / 1e6,
                np.nanmean(np.stack(curves), 0),
                color=FS_COLORS.get(fs, INK),
                lw=1.8,
                label=f"FS {fs}",
            )
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.suptitle(
        f"Trening: średnia krocząca z {window} epizodów",
        fontsize=9,
        color=MUTED,
        x=0.01,
        ha="left",
    )
    finish_grid(fig, axes, name, out)


def plot_train_metrics(results: Path, out: Path) -> None:
    plot_train_grid(results, out, TRAIN_PANELS, "train_metrics", ncols=3)


def plot_reward_components(results: Path, out: Path) -> None:
    plot_train_grid(results, out, REWARD_PANELS, "reward_components", ncols=5)


# --- final evaluation ------------------------------------------------------

BAR_METRICS = {
    "attributed_kills": ("Zabójstwa przypisane", 1.0),
    "kill_count": ("Wszystkie zabójstwa", 1.0),
    "survival_time": ("Przeżycie [s]", 1 / TICS_PER_SECOND),
}


def plot_final_comparison(results: Path, out: Path) -> None:
    entries: list[tuple[str, str, dict]] = []
    for agent, color, label in (
        ("random", RANDOM_COLOR, "losowy"),
        ("no_attack", NO_ATTACK_COLOR, "bez strzelania"),
    ):
        s = read_summary(results / agent / "summary.json")
        if s:
            entries.append((label, color, s))
    for fs, d in fs_dirs(results).items():
        s = read_summary(d / "eval_skill3" / "summary.json")
        if s:
            entries.append((f"DQN FS {fs}", FS_COLORS.get(fs, INK), s))
    if not any(label.startswith("DQN") for label, _, _ in entries):
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, (key, (title, scale)) in zip(axes, BAR_METRICS.items(), strict=True):
        x = np.arange(len(entries))
        for i, (_label, color, s) in enumerate(entries):
            st = s["across_seeds"][key]
            mean = st["mean"] * scale
            err = [[mean - st["ci_low"] * scale], [st["ci_high"] * scale - mean]]
            ax.bar(i, mean, width=0.7, color=color, edgecolor=PAPER, lw=2)
            ax.errorbar(i, mean, yerr=err, color=INK, lw=1, capsize=3)
            ax.text(
                i,
                st["ci_high"] * scale,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=MUTED,
            )
        ax.set_xticks(x, [e[0] for e in entries], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    save(fig, out, "final_comparison")


def plot_skill(results: Path, out: Path) -> None:
    def series(get_dir) -> tuple[list[int], dict[str, list[float]]]:
        skills, values = [], {k: [] for k in BAR_METRICS}
        for skill in range(1, 6):
            s = read_summary(get_dir(skill) / "summary.json")
            if s is None:
                continue
            skills.append(skill)
            for k, (_, scale) in BAR_METRICS.items():
                values[k].append(s["across_seeds"][k]["mean"] * scale)
        return skills, values

    fs4 = results / "dqn" / "fs4"
    dqn = series(lambda k: fs4 / f"eval_skill{k}")
    rnd = series(lambda k: results / "random" / ("" if k == 3 else f"skill{k}"))
    if len(dqn[0]) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, (key, (title, _)) in zip(axes, BAR_METRICS.items(), strict=True):
        for (skills, values), color, ls, label in (
            (dqn, DQN_COLOR, "-", "DQN (FS 4, trenowany na skill 3)"),
            (rnd, RANDOM_COLOR, "--", "losowy"),
        ):
            if skills:
                ax.plot(
                    skills,
                    values[key],
                    color=color,
                    ls=ls,
                    lw=2,
                    marker="o",
                    ms=5,
                    label=label,
                )
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("doom_skill")
        ax.set_xticks(range(1, 6))
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, out, "skill_comparison")


def plot_actions(results: Path, out: Path) -> None:
    def load(path: Path) -> dict[str, float] | None:
        s = read_summary(path)
        if s is None:
            return None
        total = sum(s.values())
        return {k: v / total for k, v in s.items()}

    dqn = load(results / "dqn" / "fs4" / "eval_skill3" / "action_counts.json")
    rnd = load(results / "random" / "action_counts.json")
    if dqn is None:
        return
    names = sorted(dqn, key=lambda k: dqn[k])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.barh(
        y + 0.2, [dqn[n] * 100 for n in names], 0.4, color=DQN_COLOR, label="DQN (FS 4)"
    )
    if rnd:
        ax.barh(
            y - 0.2,
            [rnd[n] * 100 for n in names],
            0.4,
            color=RANDOM_COLOR,
            label="losowy",
        )
    ax.set_yticks(y, names, fontsize=7)
    ax.set_xlabel("udział akcji [%]")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, out, "action_histogram")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("docs/presentations/img"))
    args = parser.parse_args()
    for plot in (
        plot_learning_curves,
        plot_train_reward,
        plot_eval_metrics,
        plot_train_metrics,
        plot_reward_components,
        plot_final_comparison,
        plot_skill,
        plot_actions,
    ):
        plot(args.results, args.out)


if __name__ == "__main__":
    main()
