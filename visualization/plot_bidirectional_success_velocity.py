#!/usr/bin/env python3
"""Generate bidirectional success-rate vs velocity chart using matplotlib."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    "baseline",
    "long_horizon_planning",
    "curated_tactics",
    "informed_traceback",
    "attck_r",
]

METHOD_LABELS = {
    "baseline": "Baseline (GALA)",
    "long_horizon_planning": "with Long-Horizon Planning",
    "curated_tactics": "with Curated Tactics",
    "informed_traceback": "with Informed Traceback",
    "attck_r": "ATTCK-R",
}

# Softer, visually balanced palette
METHOD_COLORS = {
    "baseline": "#0072B2",
    "long_horizon_planning": "#E69F00",
    "curated_tactics": "#009E73",
    "informed_traceback": "#D55E00",
    "attck_r": "#CC79A7",
}


def read_metric_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model", *METHODS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")

    for m in METHODS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    df = df.dropna(subset=METHODS).copy()
    return df


def build_plot_rows(sr_df: pd.DataFrame, vel_df: pd.DataFrame) -> List[str]:
    common = [m for m in sr_df["model"].tolist() if m in set(vel_df["model"].tolist())]
    if not common:
        raise ValueError("No common models found between success and velocity files.")

    selected = common[:2]

    sr_common = sr_df[sr_df["model"].isin(common)].copy()
    vel_common = vel_df[vel_df["model"].isin(common)].copy()

    sr_avg = {m: sr_common[m].mean() for m in METHODS}
    vel_avg = {m: vel_common[m].mean() for m in METHODS}

    avg_label = f"Average \n(of {len(common)} models)"
    sr_df.loc[len(sr_df)] = {"model": avg_label, **sr_avg}
    vel_df.loc[len(vel_df)] = {"model": avg_label, **vel_avg}

    selected.append(avg_label)
    return selected


def to_model_map(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        out[str(row["model"])] = {m: float(row[m]) for m in METHODS}
    return out


def plot_bidirectional(
    selected_models: List[str],
    sr_map: Dict[str, Dict[str, float]],
    vel_map: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    n = len(selected_models)
    row_step = 0.90
    y = np.arange(n) * row_step

    fig, (ax_l, ax_c, ax_r) = plt.subplots(
        1,
        3,
        figsize=(18, max(5.6, 1.9 * n)),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 0.34, 1], "wspace": 0.02},
    )

    bar_h = 0.14
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * bar_h

    max_vel = max(vel_map[m][k] for m in selected_models for k in METHODS)
    max_sr = max(sr_map[m][k] for m in selected_models for k in METHODS)

    for i, method in enumerate(METHODS):
        ypos = y + offsets[i]
        vel_vals = [vel_map[m][method] for m in selected_models]
        sr_vals = [sr_map[m][method] for m in selected_models]

        ax_l.barh(
            ypos,
            vel_vals,
            height=bar_h,
            color=METHOD_COLORS[method],
            # edgecolor="black",
            linewidth=0.4,
            label=METHOD_LABELS[method],
        )
        ax_r.barh(
            ypos,
            sr_vals,
            height=bar_h,
            color=METHOD_COLORS[method],
            # edgecolor="black",
            linewidth=0.4,
        )

    # Left panel: bars point left
    ax_l.set_xlim(max_vel * 1.15, 0)
    ax_l.set_ylim(y[0] - 0.52, y[-1] + 0.52)
    ax_l.grid(axis="x", linestyle="--", alpha=0.35)
    # Explicit zero/reference line at the center boundary for velocity panel
    ax_l.axvline(0, color="#444444", linewidth=1.6, zorder=5)
    ax_l.set_xlabel("Velocity (success/hour)", fontsize=18, fontweight="bold")
    ax_l.tick_params(axis="x", labelsize=14)
    ax_l.set_yticks([])

    # Right panel
    ax_r.set_xlim(0, max_sr * 1.15)
    ax_r.set_ylim(y[0] - 0.52, y[-1] + 0.52)
    ax_r.grid(axis="x", linestyle="--", alpha=0.35)
    ax_r.set_xlabel("Success Rate", fontsize=18, fontweight="bold")
    ax_r.tick_params(axis="x", labelsize=14)
    ax_r.set_yticks([])

    # Center model labels in dedicated middle axis
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(y[0] - 0.52, y[-1] + 0.52)
    ax_c.axis("off")
    for yi, model in zip(y, selected_models):
        if "Average" in model:
            fontweight = "bold"
        else:
            fontweight = "normal"
        ax_c.text(
            0.5,
            yi,
            model,
            ha="center",
            va="center",
            fontsize=17,
            fontweight=fontweight,
        )

    ax_l.invert_yaxis()
    ax_c.invert_yaxis()
    ax_r.invert_yaxis()

    # Clean look
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)
    ax_l.spines["left"].set_visible(False)

    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)

    # Legend in top-center (single row) to avoid overlap
    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=5,
        fontsize=13,
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--success_csv",
        type=Path,
        default=script_dir / "sample_results" / "success_rate_experiments.csv",
        help="CSV with success rates per model and method",
    )
    parser.add_argument(
        "--velocity_csv",
        type=Path,
        default=script_dir / "sample_results" / "velocity_experiments.csv",
        help="CSV with velocity (success/hour) per model and method",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=script_dir / "bidirectional_success_velocity.pdf",
        help="Output figure path (png/pdf/svg, etc.)",
    )
    args = parser.parse_args()

    sr_df = read_metric_csv(args.success_csv)
    vel_df = read_metric_csv(args.velocity_csv)
    selected = build_plot_rows(sr_df, vel_df)
    plot_bidirectional(selected, to_model_map(sr_df), to_model_map(vel_df), args.output_path)
    print(f"Saved figure to: {args.output_path}")


if __name__ == "__main__":
    main()
