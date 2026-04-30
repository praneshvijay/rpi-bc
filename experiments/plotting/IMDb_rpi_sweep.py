"""
RPIBC Hyperparameter Sweep plotting — identical style to experiments/plotting/IMDb.py.
Each sweep config (rpibc_T{T}_K{K}_S{S}) is treated as a separate method.
RLHF and ABC baselines are included for comparison.
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

parser = argparse.ArgumentParser()
parser.add_argument("--save_fig", action="store_true")
args = parser.parse_args()

# ── Exact same rcParams as IMDb.py ────────────────────────────────────────────
plt.rcParams["figure.figsize"] = [8, 5]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
plt.rcParams["font.style"] = "normal"
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["figure.titlesize"] = 16
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.4
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["grid.linewidth"] = 2
plt.rcParams["font.family"] = "DejaVu Sans"

graph_colour = "black"
plt.rcParams["axes.edgecolor"] = graph_colour
plt.rcParams["xtick.color"] = graph_colour
plt.rcParams["ytick.color"] = graph_colour
plt.rcParams["axes.labelcolor"] = graph_colour

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NUMERICS_DIR = os.path.join(BASE_PATH, "results", "numerics")

# Baselines use same colours as IMDb.py
BASE_COLOURS = {"rlhf": "#94ba22", "abc": "#1fbdaa"}
BASE_MARKERS = {"rlhf": "s", "abc": "o"}
BASE_NAMES   = {"rlhf": "RLHF", "abc": "ABC"}

# Sweep configs get distinct colors (one per T×K combo)
SWEEP_PALETTE = [
    "#e41a1c", "#ff7f00", "#ffcc00",   # T=5:  K=3,5,10
    "#377eb8", "#1fbdaa", "#0d4a8b",   # T=10: K=3,5,10
    "#984ea3", "#c966d0", "#f781bf",   # T=20: K=3,5,10
]
SWEEP_MARKERS = ["o", "s", "D", "^", "P", "X", "v", "h", "*"]


def load_all():
    res = {}
    pkl_files = glob.glob(os.path.join(NUMERICS_DIR, "IMDb_*.pkl"))
    print(f"\n[LOAD] Found {len(pkl_files)} pkl file(s) in {NUMERICS_DIR}:")
    for f in sorted(pkl_files):
        try:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
                if isinstance(data, dict):
                    print(f"  {os.path.basename(f)}  → {len(data)} key(s): {list(data.keys())}")
                    res.update(data)
        except Exception as e:
            print(f"  ERROR loading {f}: {e}")
    print(f"[LOAD] Total keys loaded: {len(res)}\n")
    return res


def group_runs(res):
    groups = {}
    print("[GROUP] Classifying all keys:")
    for key, val in res.items():
        # Skip test/smoke-test runs
        if "_TEST" in key.upper():
            print(f"  SKIP (test) {key}")
            continue

        m = re.match(r"(rpibc_T\d+_K\d+_S\d+)(?!_L)", key)
        # Deep RPIBC: rpibc_deep_T5_K10_S1_Lall_...
        # Use [a-zA-Z0-9]+ instead of \w+ to avoid matching underscores (timestamps)
        m_deep = re.match(r"(rpibc_deep_T\d+_K\d+_S\d+_L[a-zA-Z0-9]+)", key)
        if m_deep:
            tag = m_deep.group(1)
        elif m:
            tag = m.group(1)
        elif key.startswith("abc_"):
            tag = "abc"
        elif key.startswith("rlhf_"):
            tag = "rlhf"
        else:
            print(f"  SKIP  {key}")
            continue
        print(f"  {tag:<36} ← {key}  (len={len(val)})")
        groups.setdefault(tag, []).append(val)
    print(f"[GROUP] {len(groups)} group(s): {list(groups.keys())}\n")
    return groups


def aggregate(runs, tag="?", max_steps=151):
    target = min(max(len(r) for r in runs), max_steps)
    valid = [r[:target] for r in runs if len(r) >= target]
    print(f"  [AGG] {tag}: {len(runs)} run(s) total, target_len={target}, "
          f"valid (≥target)={len(valid)}/{len(runs)}, "
          f"run lengths={[len(r) for r in runs]}")
    if not valid:
        return None, None, 0
    arr = np.array(valid)
    mean, std = np.mean(arr, axis=0), np.std(arr, axis=0)
    print(f"         mean_final={mean[-1]:.3f}  std_final={std[-1]:.3f}")
    auc        = float(np.trapz(mean)) / len(mean)
    avg_reward = float(np.mean(mean))
    avg_std    = float(np.mean(std))
    pct90      = np.where(mean >= 0.9 * mean.max())[0]
    pct90_step = int(pct90[0]) if len(pct90) else -1
    print(f"         [paper metrics] AUC={auc:.3f}  avg_reward={avg_reward:.3f}  "
          f"avg_seed_std={avg_std:.3f}  steps_to_90pct_peak={pct90_step}")
    return mean, std, len(valid)


def plot_sweep(save=False):
    res = load_all()
    groups = group_runs(res)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    handles = []
    means = {}

    # ── Plot baselines first ──────────────────────────────────
    for tag in ("rlhf", "abc"):
        if tag not in groups:
            print(f"Skipping {tag}: no runs found.")
            continue
        mean, std, n = aggregate(groups[tag], tag=tag)
        if mean is None:
            continue
        means[tag] = mean
        c = BASE_COLOURS[tag]
        marker = BASE_MARKERS[tag]
        ax.plot(mean, color=c)
        ax.fill_between(range(len(mean)), mean - std, mean + std,
                        alpha=0.2, color=c, edgecolor="none")
        ax.plot(range(0, len(mean), 20), mean[::20], marker,
                markersize=8, markerfacecolor="white",
                markeredgewidth=2, markeredgecolor=c)
        handles.append(mlines.Line2D([], [], color=c, marker=marker,
                                     markersize=8, markeredgewidth=2,
                                     markerfacecolor="white", linestyle="-",
                                     label=BASE_NAMES[tag]))

    # ── Plot sweep configs ────────────────────────────────────
    sweep_tags = sorted([t for t in groups if t not in ("abc", "rlhf")])
    for idx, tag in enumerate(sweep_tags):
        mean, std, n = aggregate(groups[tag], tag=tag)
        if mean is None:
            print(f"Skipping {tag}: no complete runs.")
            continue
        means[tag] = mean
        c = SWEEP_PALETTE[idx % len(SWEEP_PALETTE)]
        marker = SWEEP_MARKERS[idx % len(SWEEP_MARKERS)]
        m = re.match(r"rpibc_T(\d+)_K(\d+)_S(\d+)", tag)
        label = f"T={m.group(1)},K={m.group(2)}" if m else tag
        ax.plot(mean, color=c)
        ax.fill_between(range(len(mean)), mean - std, mean + std,
                        alpha=0.2, color=c, edgecolor="none")
        ax.plot(range(0, len(mean), 20), mean[::20], marker,
                markersize=8, markerfacecolor="white",
                markeredgewidth=2, markeredgecolor=c)
        handles.append(mlines.Line2D([], [], color=c, marker=marker,
                                     markersize=8, markeredgewidth=2,
                                     markerfacecolor="white", linestyle="-",
                                     label=label))

    # ── Crossing analysis: same as IMDb.py ───────────────────
    if "rlhf" not in means:
        print("No RLHF baseline; skipping crossing analysis.")
    else:
        rlhf_argmax = int(np.argmax(means["rlhf"]))
        rlhf_max = means["rlhf"][rlhf_argmax]
        print(f"RLHF max: {rlhf_max:.3f} at step {rlhf_argmax}")

        # Check all sweep configs + abc
        check = ["abc"] + sweep_tags
        y_levels = [3.5, 2.5] + [1.5 - 0.5 * i for i in range(len(sweep_tags))]

        for i, m_tag in enumerate(check):
            if m_tag not in means:
                continue
            gt = np.where(means[m_tag] > rlhf_max)[0]
            peak_idx = int(np.argmax(means[m_tag]))
            peak_val = means[m_tag][peak_idx]
            print(f"{m_tag} peak: {peak_val:.3f} at step {peak_idx}")

            if len(gt):
                cross_idx = int(gt[0])
                cross_val = means[m_tag][cross_idx]
                print(f"{m_tag} crosses RLHF max at step {cross_idx}")
            else:
                cross_idx = peak_idx
                cross_val = peak_val
                print(f"{m_tag} NEVER crosses RLHF max.")

            if m_tag == "abc":
                col = BASE_COLOURS["abc"]
                short = "ABC"
            else:
                mm = re.match(r"rpibc_T(\d+)_K(\d+)_S(\d+)", m_tag)
                col = SWEEP_PALETTE[(sweep_tags.index(m_tag)) % len(SWEEP_PALETTE)]
                short = f"T={mm.group(1)},K={mm.group(2)}" if mm else m_tag

            ax.scatter([cross_idx], [cross_val], color=graph_colour,
                       marker="x", s=80, zorder=10.0)
            ax.vlines(cross_idx, ymin=0, ymax=cross_val,
                      color=col, linestyle="--", linewidth=1.5)
            if rlhf_argmax > 0:
                ax.annotate("",
                    xy=(cross_idx, y_levels[i]),
                    xytext=(rlhf_argmax, y_levels[i]),
                    arrowprops=dict(arrowstyle="<->", lw=1.5,
                                   color=col, linestyle="--"))
            ax.text(cross_idx + 2, y_levels[i], short,
                    fontsize=11, color=col, weight="bold")

        # RLHF peak marker
        ax.scatter([rlhf_argmax], [rlhf_max], color=graph_colour,
                   marker="x", s=80, zorder=10.0)
        ax.vlines(rlhf_argmax, ymin=0, ymax=rlhf_max,
                  color=BASE_COLOURS["rlhf"], linestyle="--", linewidth=1.5)

        ax.text(25, 1.0, "Steps to reach RLHF max →",
                fontsize=12, color=graph_colour, weight="bold")

    # ── Axes / ticks / spines (exact IMDb.py) ────────────────
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    n_cols = min(len(handles), 5)
    legend = fig.legend(handles=handles, loc="lower center",
                        bbox_to_anchor=(0.5, -0.07), ncol=n_cols, fontsize=12)
    all_colours = list(BASE_COLOURS.values()) + SWEEP_PALETTE
    for text, color in zip(legend.get_texts(), all_colours):
        text.set_color("white")
        text.set_weight("bold")
        text.set_path_effects([path_effects.withStroke(linewidth=6, foreground=color)])

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.set_xlim(0, 155)
    ax.patch.set_facecolor("none")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="both", which="minor", length=3, width=1, color=graph_colour)
    ax.tick_params(axis="both", which="major", length=6, width=2, color=graph_colour)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout()
    if save:
        out = os.path.join(BASE_PATH, "results", "figures", "IMDb_rpi_sweep.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
        print(f"Saved → {out}")


def plot_summary_chart(means_dict, std_dict, save=False):
    """4-panel bar chart comparing all methods on paper metrics at a glance."""
    # Gather stats per method
    rows = []
    for tag, mean in means_dict.items():
        std  = std_dict.get(tag, np.zeros_like(mean))
        auc  = float(np.trapz(mean)) / len(mean)
        avg  = float(np.mean(mean))
        astd = float(np.mean(std))
        pct  = np.where(mean >= 0.9 * mean.max())[0]
        p90  = int(pct[0]) if len(pct) else len(mean)
        # pretty label
        m = re.match(r"rpibc_T(\d+)_K(\d+)_S(\d+)", tag)
        m_deep = re.match(r"rpibc_deep_T(\d+)_K(\d+)_S(\d+)_L([a-zA-Z0-9]+)", tag)
        
        if m_deep:
            label = f"Deep RPI (T={m_deep.group(1)},K={m_deep.group(2)},L={m_deep.group(4)})"
        elif m:
            label = f"T={m.group(1)},K={m.group(2)},S={m.group(3)}"
        elif tag == "abc":
            label = "ABC (Baseline)"
        elif tag == "rlhf":
            label = "RLHF (Standard)"
        else:
            label = tag.upper()
            
        rows.append({"label": label, "tag": tag, "auc": auc, "avg": avg, "astd": astd, "p90": p90})

    if not rows:
        return

    rows.sort(key=lambda x: -x["auc"])   # sort by AUC desc

    labels = [r["label"] for r in rows]
    colors = []
    for r in rows:
        tag = r["tag"]
        if tag == "abc":
            colors.append(BASE_COLOURS["abc"])
        elif tag == "rlhf":
            colors.append(BASE_COLOURS["rlhf"])
        else:
            sweep_tags = sorted([t for t in means_dict if t not in ("abc", "rlhf")])
            idx = sweep_tags.index(tag) if tag in sweep_tags else 0
            colors.append(SWEEP_PALETTE[idx % len(SWEEP_PALETTE)])

    fig, axes = plt.subplots(1, 4, figsize=(18, max(5, len(rows) * 0.4 + 2)))
    fig.suptitle("RPIBC Hyperparameter Sweep — Summary Metrics (IMDb)", fontsize=16, weight="bold", y=0.98)

    metrics = [
        ("auc",  "AUC (higher = better)",          True),
        ("avg",  "Avg Training Reward (higher)",    True),
        ("astd", "Avg Seed Std (lower = stable)",   False),
        ("p90",  "Steps to 90% of Peak (lower)",   False),
    ]

    for ax, (key, title, higher_better) in zip(axes, metrics):
        vals = [r[key] for r in rows]
        best = max(vals) if higher_better else min(vals)
        bar_colors = ["gold" if v == best else c for v, c in zip(vals, colors)]

        bars = ax.barh(labels, vals, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        # Value annotations on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}" if key != "p90" else str(int(val)),
                    va="center", fontsize=8)

        # Star the best — use the actual position in THIS panel's sorted order
        best_idx = vals.index(best)   # correct: index in current rows
        n = len(vals)
        # y position in axes coords: bars are evenly spaced, inverted
        star_y = (best_idx + 0.5) / n
        ax.text(0.98, 1.0 - star_y, "★",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=13, color="darkgoldenrod", weight="bold")

    plt.tight_layout()
    if save:
        out = os.path.join(BASE_PATH, "results", "figures", "IMDb_rpi_sweep_summary.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
        print(f"Saved summary → {out}")
    else:
        plt.show()


def run_all(save=False):
    res = load_all()
    groups = group_runs(res)

    # Collect means and stds for both plots
    all_means, all_stds = {}, {}
    for tag, runs in groups.items():
        target = min(max(len(r) for r in runs), 151)
        valid = [r[:target] for r in runs if len(r) >= target]
        if not valid:
            continue
        arr = np.array(valid)
        all_means[tag] = np.mean(arr, axis=0)
        all_stds[tag]  = np.std(arr, axis=0)

    plot_sweep(save=save)
    plot_summary_chart(all_means, all_stds, save=save)


run_all(save=args.save_fig)
