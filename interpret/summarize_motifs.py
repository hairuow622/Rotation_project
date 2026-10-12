"""Rank motifs by their contribution to the bound prediction, combining TF-MoDISco, TOMTOM and Fi-NeMo results.

Inputs (in --out_dir): patterns_annotation.tsv, finemo/hits_unique.tsv, regions.bed, contrib.npz
Outputs (in --out_dir):
    motif_ranking.tsv           one row per motif, sorted by total hit contribution
    motif_cooccurrence.tsv      observed / expected co-occurrence of the top motifs across sequences
    motif_ranking.png           total contribution and sequence coverage per motif
    motif_positions.png         hit position relative to the peak center for the top motifs
    motif_cooccurrence.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

CATEGORY_COLORS = {
    "known_forkhead": "tab:red", "cofactor": "tab:blue", "other_tf": "tab:green", "composite": "tab:purple",
    "low_complexity": "tab:gray", "unmatched": "tab:olive",
}


def label(row):
    if isinstance(row["match0"], str) and row["match0"]:
        return f"{row['families'].split(',')[0]} | {row['match0']} | {row['consensus']}"
    return f"no match | {row['consensus']}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/interpret/FOXA1_A549_win240")
    p.add_argument("--top_n", type=int, default=8, help="motifs shown in the position and co-occurrence plots")
    args = p.parse_args()
    d = args.out_dir

    ann = pd.read_table(f"{d}/patterns_annotation.tsv")
    hits = pd.read_table(f"{d}/finemo/hits_unique.tsv")
    regions = pd.read_table(f"{d}/regions.bed", header=None, usecols=[0, 1, 3], names=["chrom", "start", "name"])
    contrib = np.load(f"{d}/contrib.npz")["arr_0"].sum(1)
    n_seqs = len(regions)

    center = dict(zip(regions["name"], regions["start"] + 120))
    hits["offset"] = (hits["start"] + hits["end"]) / 2 - hits["peak_name"].map(center)

    per_motif = hits.groupby("motif_name").agg(
        n_hits=("hit_importance", "size"),
        n_seqs_with_hit=("peak_id", "nunique"),
        hit_importance_mean=("hit_importance", "mean"),
        hit_importance_total=("hit_importance", "sum"),
        hit_coefficient_mean=("hit_coefficient", "mean"),
        offset_median_abs=("offset", lambda x: np.median(np.abs(x))),
    ).reset_index().rename(columns={"motif_name": "pattern"})
    per_motif["frac_seqs_with_hit"] = per_motif["n_seqs_with_hit"] / n_seqs
    per_motif["frac_of_total_positive_contrib"] = per_motif["hit_importance_total"] / contrib[contrib > 0].sum()

    ranking = ann.merge(per_motif, on="pattern", how="left")
    ranking[["n_hits", "n_seqs_with_hit"]] = ranking[["n_hits", "n_seqs_with_hit"]].fillna(0).astype(int)
    ranking = ranking.sort_values("hit_importance_total", ascending=False, na_position="last").reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    cols = ["rank", "pattern", "category", "match0", "qval0", "match1", "match2", "consensus", "n_seqlets",
            "n_hits", "n_seqs_with_hit", "frac_seqs_with_hit", "hit_importance_mean", "hit_importance_total",
            "frac_of_total_positive_contrib", "hit_coefficient_mean", "offset_median_abs"]
    ranking[cols].to_csv(f"{d}/motif_ranking.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3g}".format):
        print(ranking[["rank", "pattern", "category", "match0", "consensus", "n_seqlets", "frac_seqs_with_hit",
                       "hit_importance_mean", "hit_importance_total", "frac_of_total_positive_contrib"]])

    pos = ranking[ranking["pattern"].str.startswith("pos_patterns") & (ranking["n_hits"] > 0)]
    labels = [label(r) for _, r in pos.iterrows()]
    colors = [CATEGORY_COLORS.get(c, "k") for c in pos["category"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.45 * len(pos) + 1.5), sharey=True)
    y = np.arange(len(pos))[::-1]
    axes[0].barh(y, pos["hit_importance_total"], color=colors)
    axes[0].set(yticks=y, yticklabels=labels, xlabel="total hit contribution (sum of DeepLiftShap scores)")
    axes[1].barh(y, pos["frac_seqs_with_hit"], color=colors)
    axes[1].set(xlabel=f"fraction of {n_seqs} bound sequences with a hit", xlim=(0, 1))
    axes[1].legend(handles=[Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()
                            if cat in set(pos["category"])], loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{d}/motif_ranking.png", dpi=150)

    top = pos["pattern"].iloc[:args.top_n].tolist()
    ncol = 4
    nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, pat in zip(axes.flat, top):
        ax.hist(hits.loc[hits["motif_name"] == pat, "offset"], bins=np.arange(-120, 121, 10), color="tab:blue")
        ax.set(title=label(pos[pos["pattern"] == pat].iloc[0]), xlabel="offset from peak center (bp)")
        ax.title.set_fontsize(8)
    for ax in axes.flat[len(top):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{d}/motif_positions.png", dpi=150)

    presence = (hits[hits["motif_name"].isin(top)].groupby(["peak_id", "motif_name"]).size().unstack(fill_value=0) > 0)
    presence = presence.reindex(index=range(n_seqs), columns=top, fill_value=False).astype(float)
    observed = presence.T @ presence / n_seqs
    freq = presence.mean()
    ratio = observed / np.outer(freq, freq)
    np.fill_diagonal(ratio.values, np.nan)
    ratio.index = ratio.columns = [label(pos[pos["pattern"] == t].iloc[0]) for t in top]
    ratio.to_csv(f"{d}/motif_cooccurrence.tsv", sep="\t")

    fig, ax = plt.subplots(figsize=(4 + 0.9 * len(top), 1.5 + 0.8 * len(top)))
    log_ratio = np.log2(ratio.values)
    im = ax.imshow(log_ratio, cmap="RdBu_r", vmin=-2, vmax=2)
    for (i, j), v in np.ndenumerate(log_ratio):
        if not np.isnan(v):
            ax.text(j, i, f"{ratio.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set(xticks=range(len(top)), yticks=range(len(top)), yticklabels=ratio.index,
           title="observed / expected co-occurrence across bound sequences")
    ax.set_xticklabels(ratio.columns, rotation=30, ha="right")
    fig.colorbar(im, ax=ax, label="log2(observed / expected)")
    fig.tight_layout()
    fig.savefig(f"{d}/motif_cooccurrence.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
