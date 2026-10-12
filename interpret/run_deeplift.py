"""DeepLiftShap hypothetical contributions on the attributable model, averaged over both strands.

Outputs (in --out_dir):
    hyp.npz      (N, 4, 240) hypothetical contributions, key arr_0 (tfmodisco-lite format)
    contrib.npz  (N, 4, 240) hyp * one-hot, contribution of the observed base
"""

import argparse
import warnings

import numpy as np
import torch
from tangermeme.deep_lift_shap import deep_lift_shap

from attr_model import DEFAULT_CKPT, build_attr_model, build_original_model, check_equivalence


def attribute(model, X, args, device):
    return deep_lift_shap(model, X, hypothetical=True, n_shuffles=args.n_shuffles, batch_size=args.batch_size,
                          random_state=args.seed, warning_threshold=args.warning_threshold, device=device,
                          verbose=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/interpret/FOXA1_A549_win240")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--n_shuffles", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--warning_threshold", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.from_numpy(np.load(f"{args.out_dir}/ohe.npz")["arr_0"]).float()
    model = build_attr_model(args.ckpt)
    check_equivalence(build_original_model(args.ckpt), model, X[:512], device)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        hyp_fwd = attribute(model, X, args, device)
        hyp_rc = attribute(model, X.flip(1, 2), args, device).flip(1, 2)
    n_warn = sum("Convergence deltas too high" in str(w.message) for w in caught)
    print(f"batches with convergence delta > {args.warning_threshold}: {n_warn}")

    hyp = ((hyp_fwd + hyp_rc) / 2).numpy().astype(np.float32)
    np.savez_compressed(f"{args.out_dir}/hyp.npz", hyp)
    np.savez_compressed(f"{args.out_dir}/contrib.npz", hyp * X.numpy())
    fwd_rc_r = np.corrcoef((hyp_fwd * X).sum(1).flatten(), (hyp_rc * X).sum(1).flatten())[0, 1]
    print(f"saved hyp.npz {hyp.shape}; forward vs reverse-complement contribution Pearson r = {fwd_rc_r:.3f}")


if __name__ == "__main__":
    main()
