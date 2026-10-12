#!/usr/bin/env python3
"""Plot the ChIP-ISO unfreezing experiment summary written by chipiso_levels.py."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LEVEL_LABELS = {
    1: "1\nzero-shot",
    2: "2\nhead",
    3: "3\n+FFN/LN",
    4: "4\n+transformer",
    5: "5\n+conv[2]",
    6: "6\nall",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="checkpoints/FOXA1_ChIPISO_indep_levels_summary.tsv")
    parser.add_argument("--out", default="doc/chipiso_levels.png")
    args = parser.parse_args()

    df = pd.read_csv(args.summary, sep="\t").sort_values("level")
    x = df["level"]
    labels = [LEVEL_LABELS.get(l, str(l)) for l in x]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("Spearman", [("ft_val_spearman", "ft_val", "tab:blue", "--"),
                      ("ft_test_spearman", "ft_test", "tab:red", "-")]),
        ("Pearson", [("ft_val_pearson", "ft_val", "tab:blue", "--"),
                     ("ft_test_pearson", "ft_test", "tab:red", "-")]),
        ("MSE (log1p signal)", [("ft_val_mse", "ft_val", "tab:blue", "--"),
                                ("ft_test_mse", "ft_test", "tab:red", "-")]),
    ]
    for ax, (title, series) in zip(axes, panels):
        for i, (col, name, color, style) in enumerate(series):
            ax.plot(x, df[col], style, marker="o", color=color, label=name)
            offset = (0, 7) if i == 0 else (0, -13)
            for xi, yi in zip(x, df[col]):
                ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=offset,
                            ha="center", fontsize=8, color=color)
        lo = min(df[c].min() for c, *_ in series)
        hi = max(df[c].max() for c, *_ in series)
        pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="center right")
    axes[0].set_ylabel("metric")
    fig.suptitle("FOXA1 ChIP-ISO fine-tuning by unfreezing level", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
