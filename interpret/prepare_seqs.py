"""One-hot encode the 240bp peak-centered test windows and keep the ones the classifier calls bound.

Outputs (in --out_dir):
    ohe.npz      (N, 4, 240) one-hot, ACGT order, key arr_0 (tfmodisco-lite format)
    regions.bed  narrowPeak with summit offset 120, for Fi-NeMo
    preds.tsv    all candidate regions with logit, prob and whether they were kept
"""

import argparse
import os

import numpy as np
import pandas as pd
import pyfaidx
import torch
from seqchromloader.utils import dna2OneHot

from attr_model import DEFAULT_CKPT, build_original_model, predict_logits

DATA_DIR = "ChIP-ISO_Lucy_lab/FoxA1_ChIP_Replicate1_Data"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--peaks", default=f"{DATA_DIR}/FoxA1_ChIP_A549_MACS2_callpeak_narrowPeak_240bp_around_center.bed")
    p.add_argument("--genome", default="/home/imv5103/group/lab/ishika/genomes/hg38/hg38.fa")
    p.add_argument("--chroms", nargs="+", default=["chr2"])
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--prob_threshold", type=float, default=0.5)
    p.add_argument("--out_dir", default="results/interpret/FOXA1_A549_win240")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    peaks = pd.read_table(args.peaks, header=None, usecols=range(3), names=["chrom", "start", "end"])
    peaks = peaks[peaks["chrom"].isin(args.chroms)].reset_index(drop=True)
    if not (peaks["end"] - peaks["start"] == 240).all():
        raise ValueError("expected 240bp windows")

    genome = pyfaidx.Fasta(args.genome)
    seqs = [genome[r.chrom][r.start:r.end].seq for r in peaks.itertuples()]
    peaks["has_N"] = [("N" in s.upper()) for s in seqs]
    X = torch.from_numpy(np.stack([dna2OneHot(s) for s in seqs]))

    peaks["logit"] = predict_logits(build_original_model(args.ckpt), X, device).numpy()
    peaks["prob"] = 1 / (1 + np.exp(-peaks["logit"]))
    peaks["kept"] = (peaks["prob"] >= args.prob_threshold) & ~peaks["has_N"]
    peaks.to_csv(f"{args.out_dir}/preds.tsv", sep="\t", index=False)

    kept = peaks[peaks["kept"]].reset_index(drop=True)
    np.savez_compressed(f"{args.out_dir}/ohe.npz", X[torch.from_numpy(peaks["kept"].to_numpy())].numpy())
    pd.DataFrame({
        "chrom": kept["chrom"], "start": kept["start"], "end": kept["end"],
        "name": [f"{r.chrom}:{r.start}-{r.end}" for r in kept.itertuples()],
        "score": 0, "strand": ".", "signal": kept["prob"], "p": -1, "q": -1, "summit": 120,
    }).to_csv(f"{args.out_dir}/regions.bed", sep="\t", header=False, index=False)

    print(f"{len(peaks)} peaks on {','.join(args.chroms)}; {peaks['has_N'].sum()} contain N; "
          f"{(peaks['prob'] >= args.prob_threshold).sum()} predicted bound (prob >= {args.prob_threshold}); "
          f"kept {len(kept)}")


if __name__ == "__main__":
    main()
