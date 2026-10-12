#!/usr/bin/env python3
"""
Summarize the ChIP-ISO unfreezing experiment (see doc/chip-iso_fine-tuning.md):
evaluate every level on ft_val and ft_test, and pick the best level on ft_val.

Machine-readable results are printed as KEY=VALUE lines on stdout; everything else goes to stderr.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from datamodule import ChIPISODataModule
from models import ConvTowerDomain_v6_FineTune


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def best_ckpt(run_dir):
    ckpts = sorted(glob.glob(os.path.join(run_dir, "best-*.ckpt")), key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError(f"no best-*.ckpt in {run_dir}")
    return ckpts[-1]


def subset(dm, split):
    manifest = pd.read_csv(dm.split_file, sep="\t")
    return dm._build(manifest[manifest["split"] == split])


@torch.no_grad()
def predict(model, ds, device, batch_size=256):
    model.eval()
    preds = []
    for i in range(0, len(ds.seqs), batch_size):
        seq = ds.seqs[i:i + batch_size].to(device)
        preds.append(model(seq, None).float().flatten().cpu())
    return torch.cat(preds).numpy()


def metrics(pred, ds):
    y = ds.targets.flatten().numpy()
    return {"spearman": spearmanr(pred, y)[0], "pearson": pearsonr(pred, y)[0], "mse": float(np.mean((pred - y) ** 2))}


def zero_shot_model(args, train_ds, device):
    model = ConvTowerDomain_v6_FineTune(pretrained_ckpt=args.pretrained_ckpt,
                                        classification=False, input_len=240, conv1d_filter=128,
                                        conv_tower_depth=4, conv_tower_kernel=5, attn_num_layers=1,
                                        attn_num_heads=8, attn_dim_feedforward=1024, seqonly=True).to(device)
    pred = predict(model, train_ds, device)
    a, b = np.polyfit(pred, train_ds.targets.flatten().numpy(), 1)
    with torch.no_grad():
        model.main_pred[0].weight.mul_(float(a))
        model.main_pred[0].bias.mul_(float(a)).add_(float(b))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta", default="ChIP-ISO_Lucy_lab/ChIP-ISO_signal/A549_FoxA1_lib1_plus_2_all.fasta")
    parser.add_argument("--split-file", default="ChIP-ISO_Lucy_lab/ChIP-ISO_signal/chipiso_split.tsv")
    parser.add_argument("--pretrained-ckpt", default="checkpoints/FOXA1_A549_rep1_center_win240_wd1e-4/best-epoch=40-val_loss=0.344953.ckpt")
    parser.add_argument("--runs", nargs="+", required=True, help="LEVEL=RUN_DIR, or LEVEL=zeroshot for level 1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dm = ChIPISODataModule(args.fasta, args.split_file, num_workers=0)
    dm.setup("fit")
    datasets = {"ft_val": dm.val_ds, "ft_test": subset(dm, "ft_test")}

    rows = []
    for spec in args.runs:
        level, run_dir = spec.split("=", 1)
        if run_dir == "zeroshot":
            model, ckpt = zero_shot_model(args, dm.train_ds, device), ""
        else:
            ckpt = best_ckpt(run_dir)
            model = ConvTowerDomain_v6_FineTune.load_from_checkpoint(ckpt, map_location="cpu").to(device)
        row = {"level": int(level), "unfreeze_level": model.unfreeze_level, "ckpt": ckpt}
        for split, ds in datasets.items():
            for k, v in metrics(predict(model, ds, device), ds).items():
                row[f"{split}_{k}"] = v
        rows.append(row)
        log(f"level {level}: val rho={row['ft_val_spearman']:.3f}  test rho={row['ft_test_spearman']:.3f}")

    df = pd.DataFrame(rows).sort_values("level")
    df.to_csv(args.out, sep="\t", index=False, float_format="%.4f")
    log("\n" + df.drop(columns="ckpt").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    log(f"Wrote {args.out}")

    best = df.loc[df["ft_val_spearman"].idxmax()]
    log(f"Best level on ft_val: {int(best['level'])} (val Spearman {best['ft_val_spearman']:.3f})")
    print(f"BEST_LEVEL={int(best['level'])}")
    print(f"BEST_CKPT={best['ckpt']}")


if __name__ == "__main__":
    main()
