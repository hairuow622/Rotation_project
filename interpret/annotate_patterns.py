"""Annotate TF-MoDISco patterns with TOMTOM (JASPAR) and sort them into categories.

Categories:
    known_forkhead   best significant match is a FOX motif
    cofactor         best significant match is a candidate FOXA1 co-binding family (AP-1, CEBP, GRHL, NFI, ...)
    other_tf         significant match to some other TF
    composite        significant matches from >= 2 TF families on a long pattern (likely two motifs side by side)
    low_complexity   no significant match and strongly GC/AT-biased or repetitive consensus
    unmatched        no significant match (best_any / best_any_qval still report the closest motif)

Output: patterns_annotation.tsv in --out_dir
"""

import argparse
import re

import h5py
import numpy as np
import pandas as pd
from modiscolite.report import fetch_tomtom_matches

COFACTOR_FAMILIES = {
    "AP-1": r"^(FOS|JUN|ATF[2-7]|BATF|MAF)",
    "CEBP": r"^CEBP",
    "GRHL": r"^(GRHL|TFCP2)",
    "NFI": r"^NFI[ABCX]",
    "HNF1": r"^HNF1",
    "HNF4": r"^HNF4",
    "TEAD": r"^TEAD",
    "NR": r"^(NR[0-9]|ESR|AR$|GR$|NR3C|RXR|RAR|PPAR)",
    "CTCF": r"^CTCF",
    "ETS": r"^(ETS|ELF|ELK|ETV|ERG|FLI1|GABPA|SPI)",
    "KLF/SP": r"^(KLF|SP[1-9])",
    "NFKB": r"^(NFKB|REL)",
    "TP53": r"^TP(53|63|73)",
}


def read_meme_names(path):
    names = {}
    with open(path) as f:
        for line in f:
            if line.startswith("MOTIF"):
                fields = line.split()
                names[fields[1]] = fields[2] if len(fields) > 2 else fields[1]
    return names


def family(tf_name):
    name = tf_name.upper().split("::")[0]
    if name.startswith("FOX"):
        return "Forkhead"
    for fam, pattern in COFACTOR_FAMILIES.items():
        if re.match(pattern, name):
            return fam
    return name


def trim(ppm, cwm, threshold=0.3):
    score = np.abs(cwm).sum(1)
    keep = np.where(score >= score.max() * threshold)[0]
    return ppm[keep.min():keep.max() + 1]


def consensus_stats(ppm):
    consensus = "".join("ACGT"[i] for i in ppm.argmax(1))
    gc = ppm[:, [1, 2]].sum() / ppm.sum()
    ic = (2 + np.sum(ppm * np.log2(ppm + 1e-6), axis=1)).mean()
    period1 = max(consensus.count(b) for b in "ACGT") / len(consensus)
    period2 = np.mean([consensus[i] == consensus[i + 2] for i in range(len(consensus) - 2)]) if len(consensus) > 2 else 0
    return consensus, gc, ic, period1, period2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/interpret/FOXA1_A549_win240")
    p.add_argument("--meme_db", default="resources/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt")
    p.add_argument("--qval", type=float, default=0.05)
    p.add_argument("--n_matches", type=int, default=3)
    p.add_argument("--composite_min_len", type=int, default=16)
    args = p.parse_args()

    names = read_meme_names(args.meme_db)
    rows = []
    with h5py.File(f"{args.out_dir}/modisco_results.h5", "r") as h5:
        for group in ("pos_patterns", "neg_patterns"):
            if group not in h5:
                continue
            for pname in sorted(h5[group], key=lambda x: int(x.split("_")[-1])):
                pattern = h5[group][pname]
                ppm, cwm = pattern["sequence"][:], pattern["contrib_scores"][:]
                trimmed = trim(ppm, cwm)
                consensus, gc, ic, period1, period2 = consensus_stats(trimmed)

                all_matches = fetch_tomtom_matches(ppm, cwm, motifs_db=args.meme_db)
                matches = all_matches[all_matches["q-value"] < args.qval].iloc[:args.n_matches]
                match_names = [names.get(t, t) for t in matches["Target_ID"]]
                families = list(dict.fromkeys(family(n) for n in match_names))

                if len(families) >= 2 and len(consensus) >= args.composite_min_len:
                    category = "composite"
                elif families and families[0] == "Forkhead":
                    category = "known_forkhead"
                elif families and families[0] in COFACTOR_FAMILIES:
                    category = "cofactor"
                elif families:
                    category = "other_tf"
                elif gc > 0.8 or gc < 0.2 or period1 > 0.6 or period2 > 0.8:
                    category = "low_complexity"
                else:
                    category = "unmatched"

                row = {
                    "pattern": f"{group}.{pname}",
                    "n_seqlets": int(pattern["seqlets"]["n_seqlets"][:][0]),
                    "contrib_sum": float(cwm.sum()),
                    "consensus": consensus, "length": len(consensus),
                    "gc": round(float(gc), 3), "mean_ic": round(float(ic), 3),
                    "category": category, "families": ",".join(families),
                }
                for i in range(args.n_matches):
                    row[f"match{i}"] = match_names[i] if i < len(match_names) else ""
                    row[f"match{i}_id"] = matches["Target_ID"].iloc[i] if i < len(matches) else ""
                    row[f"qval{i}"] = matches["q-value"].iloc[i] if i < len(matches) else np.nan
                if len(all_matches):
                    row["best_any"] = names.get(all_matches["Target_ID"].iloc[0], all_matches["Target_ID"].iloc[0])
                    row["best_any_qval"] = all_matches["q-value"].iloc[0]
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/patterns_annotation.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[["pattern", "n_seqlets", "consensus", "category", "match0", "qval0", "match1", "match2",
                  "best_any", "best_any_qval"]])


if __name__ == "__main__":
    main()
