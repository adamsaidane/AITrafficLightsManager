"""
visualization.py — Generate training and evaluation plots from CSV logs.
"""

from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def smooth(values, window: int = 20) -> pd.Series:
    return pd.Series(values).rolling(window, min_periods=1).mean()


def plot_training_curves(csv_path: str, out_dir: str = "plots",
                         window: int = 20) -> None:
    """
    Generate a 2×3 panel of training metrics from a CSV log file.
    """
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    metrics = [
        ("reward",   "Episode Reward",           "steelblue"),
        ("waiting",  "Mean Waiting Time (s)",     "tomato"),
        ("arrived",  "Vehicles Arrived",          "mediumseagreen"),
        ("queue",    "Mean Queue Length",         "darkorange"),
        ("epsilon",  "Epsilon",                   "mediumpurple"),
        ("loss",     "Training Loss",             "dimgray"),
    ]

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    for idx, (col, title, color) in enumerate(metrics):
        if col not in df.columns:
            continue
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.plot(df["episode"], df[col], alpha=0.25, color=color)
        ax.plot(df["episode"], smooth(df[col], window),
                color=color, linewidth=2, label=f"MA({window})")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.legend()
        ax.grid(alpha=0.3)

    agent_name = os.path.basename(os.path.dirname(csv_path))
    plt.suptitle(f"Training Curves — {agent_name}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved → {out}")


def plot_phase_distribution(phase_counts: dict, out_dir: str = "plots") -> None:
    """Bar chart of how often each traffic phase was selected."""
    phases = list(phase_counts.keys())
    counts = list(phase_counts.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(phases, counts, color="steelblue", edgecolor="black")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Times Selected")
    ax.set_title("Phase Selection Distribution", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, "phase_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved → {out}")
