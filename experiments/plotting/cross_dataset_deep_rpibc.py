"""
Cross-Dataset Comparison: Deep RPIBC vs ABC vs RLHF.
Loads results from all three datasets (IMDb, TL;DR, Anthropic HH / OpenLLaMA)
and plots normalised reward curves side-by-side in a 1×3 grid.

Usage:
  python3 experiments/plotting/cross_dataset_deep_rpibc.py [--save_fig]
"""
import argparse
import glob
import os
import pickle
import re

import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--save_fig", action="store_true", help="Save to results/figures/")
args = parser.parse_args()

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 13,
    "font.family": "DejaVu Sans",
    "axes.labelsize": 13,
    "axes.labelweight": "bold",
    "axes.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": ":",
    "grid.linewidth": 1.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
})

# ── Colours / markers ─────────────────────────────────────────────────────────
C = {
    "rlhf":       "#94ba22",
    "abc":        "#1fbdaa",
    "rpibc_deep": "#e41a1c",
}
MK = {"rlhf": "s", "abc": "o", "rpibc_deep": "D"}
LABELS = {"rlhf": "RLHF", "abc": "ABC", "rpibc_deep": "Deep RPIBC"}

BASE_PATH   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NUMERICS    = os.path.join(BASE_PATH, "results", "numerics")

# ── Dataset configs ────────────────────────────────────────────────────────────
# pattern: glob prefix that selects every pkl for that dataset
# key_fn:  maps a run_name → method label (one of C.keys()) or None to skip
DATASETS = [
    {
        "title": "IMDb Sentiment",
        "glob":  "IMDb_*.pkl",
        "max_steps": 151,
        "key_fn": lambda k: (
            "rpibc_deep" if re.match(r"rpibc_deep_", k) else
            "abc"        if k.startswith("abc_")         else
            "rlhf"       if k.startswith("rlhf_")        else None
        ),
    },
    {
        "title": "TL;DR Summarization",
        "glob":  "TLDr_*.pkl",
        "max_steps": 201,
        "key_fn": lambda k: (
            "rpibc_deep" if re.match(r"rpibc_deep_", k) else
            "abc"        if k.startswith("abc_")         else
            "rlhf"       if k.startswith("rlhf_")        else None
        ),
    },
    {
        "title": "Anthropic HH (OpenLLaMA)",
        "glob":  "OpenLLaMA_*.pkl",
        "max_steps": 201,
        "key_fn": lambda k: (
            "rpibc_deep" if re.match(r"rpibc_deep_", k) else
            "abc"        if k.startswith("abc_")         else
            "rlhf"       if k.startswith("rlhf_")        else None
        ),
    },
]


def load_dataset_results(glob_pat, key_fn, max_steps):
    """Load all pkl files matching glob_pat, group by method."""
    groups = {}
    files = glob.glob(os.path.join(NUMERICS, glob_pat))
    print(f"\n[LOAD] {glob_pat} → {len(files)} file(s)")
    for f in sorted(files):
        try:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
        except Exception as e:
            print(f"  ERROR {f}: {e}")
            continue

        # Support both dict {run_name: [rewards]} and raw list formats
        if isinstance(data, dict):
            items = data.items()
        elif isinstance(data, list):
            items = [(os.path.splitext(os.path.basename(f))[0], data)]
        else:
            continue

        for run_name, rewards in items:
            method = key_fn(run_name)
            if method is None:
                print(f"  SKIP  {run_name}")
                continue
            print(f"  {method:<12} ← {run_name}  (len={len(rewards)})")
            groups.setdefault(method, []).append(rewards)

    # Aggregate
    out = {}
    for method, runs in groups.items():
        target = min(max(len(r) for r in runs), max_steps)
        valid  = [r[:target] for r in runs if len(r) >= target]
        if not valid:
            continue
        arr          = np.array(valid, dtype=float)
        out[method]  = {"mean": arr.mean(0), "std": arr.std(0), "n": len(valid)}
    return out


def plot_panel(ax, data, title, marker_every=20):
    """Plot one dataset panel on ax."""
    ax.set_title(title, fontweight="bold")
    handles = []

    for method in ["rlhf", "abc", "rpibc_deep"]:
        if method not in data:
            continue
        mean = data[method]["mean"]
        std  = data[method]["std"]
        n    = data[method]["n"]
        c    = C[method]
        mk   = MK[method]
        x    = np.arange(len(mean))

        ax.plot(x, mean, color=c, linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=c, edgecolor="none")
        ax.plot(x[::marker_every], mean[::marker_every], mk,
                markersize=7, markerfacecolor="white",
                markeredgewidth=2, markeredgecolor=c)
        handles.append(mlines.Line2D([], [], color=c, marker=mk, markersize=7,
                                     markeredgewidth=2, markerfacecolor="white",
                                     linestyle="-", label=f"{LABELS[method]} (n={n})"))

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Reward")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="minor", length=3, width=1)
    ax.tick_params(which="major", length=6, width=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.patch.set_facecolor("none")
    return handles


# ── Main ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Deep RPIBC vs Baselines — All Datasets", fontsize=16, fontweight="bold", y=1.02)

all_handles = []
for ax, ds in zip(axes, DATASETS):
    data = load_dataset_results(ds["glob"], ds["key_fn"], ds["max_steps"])
    if not data:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="gray")
        ax.set_title(ds["title"], fontweight="bold")
        continue
    h = plot_panel(ax, data, ds["title"])
    if len(h) > len(all_handles):
        all_handles = h   # keep panel with most methods for shared legend

# Shared legend below all panels
fig.legend(handles=all_handles, loc="lower center",
           bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=12,
           framealpha=0.9)

plt.tight_layout()

if args.save_fig:
    out = os.path.join(BASE_PATH, "results", "figures", "cross_dataset_deep_rpibc.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"\nSaved → {out}")
else:
    plt.show()
