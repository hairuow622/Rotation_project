"""Check whether DeepLiftShap is trustworthy on this model.

On a random subset of sequences, for the original and the attributable model:
  1. conservation: per (sequence, reference) pair, relative error between sum(mult * (x - r)) and f(x) - f(r)
  2. agreement with expected gradients (integrated gradients averaged over the same references)

Outputs (in --out_dir): diagnostics.tsv (per sequence), diagnostics_summary.tsv, diagnostics.png
"""

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tangermeme.deep_lift_shap import deep_lift_shap

from attr_model import DEFAULT_CKPT, build_attr_model, build_original_model, predict_logits


def to_hypothetical(mult, refs):
    """(N, R, 4, L) multipliers and references -> (N, 4, L) hypothetical contributions averaged over references."""
    return (mult - (refs * mult).sum(2, keepdim=True)).mean(1)


def deeplift_raw(model, X, n_shuffles, seed, device):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = deep_lift_shap(model, X, raw_outputs=True, return_references=True, n_shuffles=n_shuffles,
                             random_state=seed, warning_threshold=np.inf, device=device)
    return res.attributions, res.references


def conservation(model, X, mult, refs, device, min_diff):
    n, r = refs.shape[:2]
    fx = predict_logits(model, X, device)
    fr = predict_logits(model, refs.reshape(n * r, *X.shape[1:]), device).reshape(n, r)
    diff = fx[:, None] - fr
    total = ((X[:, None] - refs) * mult).sum((2, 3))
    rel = (total - diff).abs() / diff.abs()
    rel[diff.abs() < min_diff] = float("nan")
    return rel


def expected_gradients(model, X, refs, n_steps, device):
    """Integrated gradients from each reference (midpoint rule), projected to hypothetical scores, averaged.

    Returns the (N, 4, L) hypothetical scores and the (N, R) per-reference attribution totals.
    """
    model.to(device)
    alphas = ((torch.arange(n_steps) + 0.5) / n_steps).to(device)
    hyp = torch.zeros_like(X)
    total = torch.zeros(refs.shape[:2])
    for i in range(X.shape[0]):
        x, r = X[i].to(device), refs[i].to(device)                            # (4, L), (R, 4, L)
        path = r[:, None] + alphas[None, :, None, None] * (x - r)[:, None]    # (R, S, 4, L)
        path = path.reshape(-1, *x.shape).requires_grad_(True)
        g = torch.autograd.grad(model(path).sum(), path)[0].reshape(r.shape[0], n_steps, *x.shape).mean(1)
        hyp[i] = (g - (r * g).sum(1, keepdim=True)).mean(0).cpu()
        total[i] = ((x - r) * g).sum((1, 2)).cpu()
    return hyp, total


def per_seq_agreement(a, b, X, top_k):
    ca, cb = (a * X).sum(1).numpy(), (b * X).sum(1).numpy()
    pearson = np.array([np.corrcoef(u, v)[0, 1] for u, v in zip(ca, cb)])
    top = lambda c: np.argsort(-np.abs(c), axis=1)[:, :top_k]
    jaccard = np.array([len(set(u) & set(v)) / len(set(u) | set(v)) for u, v in zip(top(ca), top(cb))])
    return pearson, jaccard


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/interpret/FOXA1_A549_win240")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--n_seqs", type=int, default=300)
    p.add_argument("--n_shuffles", type=int, default=20)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--min_diff", type=float, default=0.1, help="skip pairs with |f(x) - f(r)| below this (logit)")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_all = torch.from_numpy(np.load(f"{args.out_dir}/ohe.npz")["arr_0"]).float()
    idx = np.sort(np.random.default_rng(args.seed).choice(len(X_all), min(args.n_seqs, len(X_all)), replace=False))
    X = X_all[idx]

    models = {"original": build_original_model(args.ckpt), "attributable": build_attr_model(args.ckpt)}
    rel, hyp = {}, {}
    refs = None
    for name, model in models.items():
        mult, r = deeplift_raw(model, X, args.n_shuffles, args.seed, device)
        if refs is None:
            refs = r
        elif not torch.equal(refs, r):
            raise RuntimeError("references differ between models")
        rel[name] = conservation(model, X, mult, r, device, args.min_diff)
        hyp[name] = to_hypothetical(mult, r)

    eg_hyp, eg_total = expected_gradients(models["original"], X, refs, args.n_steps, device)
    fx = predict_logits(models["original"], X, device)
    fr = predict_logits(models["original"], refs.reshape(-1, *X.shape[1:]), device).reshape(refs.shape[:2])
    eg_rel = ((eg_total - (fx[:, None] - fr)).abs() / (fx[:, None] - fr).abs()).nanmean().item()

    df = pd.DataFrame({"seq_index": idx})
    for name in models:
        df[f"rel_err_median_{name}"] = np.nanmedian(rel[name].numpy(), axis=1)
        df[f"pearson_vs_eg_{name}"], df[f"jaccard_top{args.top_k}_vs_eg_{name}"] = \
            per_seq_agreement(hyp[name], eg_hyp, X, args.top_k)
    df.to_csv(f"{args.out_dir}/diagnostics.tsv", sep="\t", index=False)

    rows = []
    for name in models:
        flat = rel[name].numpy().ravel()
        flat = flat[~np.isnan(flat)]
        rows.append({
            "model": name,
            "rel_err_median": np.median(flat), "rel_err_q90": np.quantile(flat, 0.9),
            "pearson_vs_eg_median": df[f"pearson_vs_eg_{name}"].median(),
            f"jaccard_top{args.top_k}_vs_eg_median": df[f"jaccard_top{args.top_k}_vs_eg_{name}"].median(),
            "usable": np.median(flat) < 0.05,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{args.out_dir}/diagnostics_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"expected gradients mean relative conservation error ({args.n_steps} steps): {eg_rel:.3g}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name in models:
        flat = rel[name].numpy().ravel()
        flat = flat[~np.isnan(flat)]
        axes[0].hist(np.log10(flat + 1e-12), bins=60, alpha=0.6, label=name)
        axes[1].hist(df[f"pearson_vs_eg_{name}"], bins=40, range=(-1, 1), alpha=0.6, label=name)
        axes[2].hist(df[f"jaccard_top{args.top_k}_vs_eg_{name}"], bins=20, range=(0, 1), alpha=0.6, label=name)
    axes[0].axvline(np.log10(0.05), color="k", ls="--", lw=1)
    axes[0].set(xlabel="log10 relative conservation error", ylabel="(sequence, reference) pairs",
                title="DeepLIFT conservation")
    axes[1].set(xlabel="Pearson r vs expected gradients", ylabel="sequences", title="Agreement (per sequence)")
    axes[2].set(xlabel=f"Jaccard of top-{args.top_k} positions", ylabel="sequences", title="Top position overlap")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/diagnostics.png", dpi=150)


if __name__ == "__main__":
    main()
