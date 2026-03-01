#!/usr/bin/env python3
"""Create a horizontal grouped bar chart for protected vs unprotected LLM scores using matplotlib."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"model", "protected_score", "unprotected_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["protected_score"] = pd.to_numeric(df["protected_score"], errors="coerce")
    df["unprotected_score"] = pd.to_numeric(df["unprotected_score"], errors="coerce")
    df = df.dropna(subset=["protected_score", "unprotected_score"]).copy()

    avg_row = {
        "model": "Average",
        "protected_score": df["protected_score"].mean(),
        "unprotected_score": df["unprotected_score"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=False)
    return df


def plot_chart(df: pd.DataFrame, output_path: Path) -> None:
    models = df["model"].tolist()
    protected = df["protected_score"].to_numpy()
    unprotected = df["unprotected_score"].to_numpy()

    y = np.arange(len(models)) * 0.84
    h = 0.24

    fig_h = max(5.6, 1.9 * len(models) + 2)
    fig, ax = plt.subplots(figsize=(18, fig_h), constrained_layout=True)

    b1 = ax.barh(
        y - h / 2,
        protected,
        height=h,
        label="Protected Target",
        color="#0072B2",
        # edgecolor="black",
        linewidth=0.4,
    )
    b2 = ax.barh(
        y + h / 2,
        unprotected,
        height=h,
        label="Unprotected Target",
        color="#E69F00",
        # edgecolor="black",
        linewidth=0.4,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=48)
    ax.invert_yaxis()
    ax.set_xlabel("Success Rate", fontsize=44, fontweight="bold")
    ax.set_xlim(0, max(np.max(protected), np.max(unprotected)) * 1.15)
    # Add extra space at the visual top (right below legend), not bottom
    ax.set_ylim(y[0] - 0.52, y[-1] + 0.80)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", labelsize=30, top=False)
    ax.tick_params(axis="y", length=0)

    # Match clean spine style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Top-center legend style to match bidirectional plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        fontsize=38,
        frameon=False,
    )

    for bars in (b1, b2):
        for bar in bars:
            x = bar.get_width()
            y_text = bar.get_y() + bar.get_height() / 2
            ax.text(x + 0.01, y_text, f"{x:.2f}", va="center", fontsize=38)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=script_dir / "sample_results" / "protected_vs_unprotected_sample.csv",
        help="Path to CSV with model/protected_score/unprotected_score",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=script_dir / "protected_vs_unprotected_barh.pdf",
        help="Output figure path (png/pdf/svg, etc.)",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv)
    plot_chart(df, args.output_path)
    print(f"Saved figure to: {args.output_path}")


if __name__ == "__main__":
    main()
