#!/usr/bin/env python3
"""
Split ChIP-ISO sequences with a measured signal into a fine-tuning part
(ft_train / ft_val / ft_test) and a held-out active learning pool (al_pool).

Sequences derived from the same parent (WT, mutants, reverse complements,
designed variants sharing a backbone) are kept on the same side of every split.
"""

import argparse
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

COORD_RE = re.compile(r"chr[0-9XYM]+:\d+-\d+")
SEQ_ID_RE = re.compile(r"^(seq\d+)(?:_|$)")


def read_fasta(path):
    seqs = {}
    name = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:]
                seqs[name] = []
            else:
                seqs[name].append(line.upper())
    return {k: "".join(v) for k, v in seqs.items()}


def read_signal(path):
    """Return {name: float}, dropping '#N/A' and empty entries."""
    signal = {}
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            try:
                signal[fields[0]] = float(fields[1])
            except ValueError:
                continue
    return signal


def revcomp(s):
    return s[::-1].translate(str.maketrans("ACGTN", "TGCAN"))


def name_key(name):
    """Parent key implied by the sequence name alone."""
    m = COORD_RE.search(name)
    if m:
        return m.group(0)
    m = SEQ_ID_RE.match(name)
    if m:
        return m.group(1)
    return None


def family(name):
    """Coarse design family, only used for reporting."""
    if COORD_RE.search(name):
        return name.split("_")[0] if name.startswith("FOXA1") else "seq_genomic"
    if SEQ_ID_RE.match(name):
        return "seq"
    if name.endswith(("_scrambled", "_polyT", "_polyA")):
        return "scan"
    if re.match(r"^(ori|mut|rev|con)_", name):
        return "ori_mut_rev_con"
    return re.sub(r"\d+", "#", name.split("_")[0])


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_groups(names, seqs, kmer=30, adapter_trim=25):
    """Union sequences that share a name-derived parent or any k-mer (either strand)."""
    uf = UnionFind(names)

    by_key = defaultdict(list)
    for n in names:
        k = name_key(n)
        if k is not None:
            by_key[k].append(n)
    for members in by_key.values():
        for m in members[1:]:
            uf.union(members[0], m)

    # Constant adapters flank every insert, so they are trimmed before k-mer matching.
    kmer_owner = {}
    for n in names:
        insert = seqs[n][adapter_trim:len(seqs[n]) - adapter_trim]
        for s in (insert, revcomp(insert)):
            for i in range(len(s) - kmer + 1):
                km = s[i:i + kmer]
                if "N" in km:
                    continue
                if km in kmer_owner:
                    uf.union(n, kmer_owner[km])
                else:
                    kmer_owner[km] = n

    groups = {n: uf.find(n) for n in names}
    # Readable, deterministic group ids.
    roots = sorted(set(groups.values()), key=lambda r: (-sum(1 for v in groups.values() if v == r), r))
    root_id = {r: f"g{i:04d}" for i, r in enumerate(roots)}
    return {n: root_id[r] for n, r in groups.items()}


def assign_groups(group_sizes, targets, rng):
    """
    Assign whole groups to splits so that sample counts approach `targets` (fractions).

    Groups too large to fit the smallest split are placed first, largest first, into the
    split with the largest absolute deficit. All other groups are visited in random order
    and go to the split with the largest relative deficit, which keeps the composition of
    each split close to random sampling.
    """
    total = sum(group_sizes.values())
    goal = {s: frac * total for s, frac in targets.items()}
    filled = {s: 0 for s in targets}
    oversize_cutoff = 0.5 * min(goal.values())

    gids = list(group_sizes)
    rng.shuffle(gids)
    oversized = sorted((g for g in gids if group_sizes[g] > oversize_cutoff), key=lambda g: -group_sizes[g])
    regular = [g for g in gids if group_sizes[g] <= oversize_cutoff]

    assignment = {}
    for g in oversized:
        split = max(targets, key=lambda s: goal[s] - filled[s])
        assignment[g] = split
        filled[split] += group_sizes[g]
    for g in regular:
        split = max(targets, key=lambda s: (goal[s] - filled[s]) / goal[s])
        assignment[g] = split
        filled[split] += group_sizes[g]
    return assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta", default="ChIP-ISO_Lucy_lab/ChIP-ISO_signal/A549_FoxA1_lib1_plus_2_all.fasta")
    parser.add_argument("--signal", default="ChIP-ISO_Lucy_lab/ChIP-ISO_signal/ChIP_ISO_signal_NM0_1_all.txt")
    parser.add_argument("--out", default="ChIP-ISO_Lucy_lab/ChIP-ISO_signal/chipiso_split.tsv")
    parser.add_argument("--al-frac", type=float, default=0.3)
    parser.add_argument("--val-frac", type=float, default=0.1, help="fraction of the fine-tuning part")
    parser.add_argument("--test-frac", type=float, default=0.1, help="fraction of the fine-tuning part")
    parser.add_argument("--kmer", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seqs = read_fasta(args.fasta)
    signal = read_signal(args.signal)
    names = sorted(n for n in signal if n in seqs)
    print(f"FASTA sequences: {len(seqs)}; numeric signal: {len(signal)}; usable: {len(names)}")

    groups = build_groups(names, seqs, kmer=args.kmer)
    df = pd.DataFrame({
        "name": names,
        "group": [groups[n] for n in names],
        "family": [family(n) for n in names],
        "signal": [signal[n] for n in names],
    })
    group_sizes = df.groupby("group").size().to_dict()

    rng = np.random.default_rng(args.seed)
    top = assign_groups(group_sizes, {"ft": 1 - args.al_frac, "al_pool": args.al_frac}, rng)
    ft_groups = {g: n for g, n in group_sizes.items() if top[g] == "ft"}
    train_frac = 1 - args.val_frac - args.test_frac
    inner = assign_groups(ft_groups, {"ft_train": train_frac, "ft_val": args.val_frac, "ft_test": args.test_frac}, rng)
    df["split"] = [inner.get(g, top[g]) for g in df["group"]]

    df[["name", "group", "family", "signal", "split"]].to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}\n")

    sizes = sorted(group_sizes.values(), reverse=True)
    print(f"Groups: {len(sizes)}; largest: {sizes[:5]}; singletons: {sizes.count(1)}")
    largest = df[df["group"] == "g0000"]
    print(f"Largest group g0000 -> {largest['split'].iloc[0]}; families: {dict(Counter(largest['family']))}\n")

    order = ["ft_train", "ft_val", "ft_test", "al_pool"]
    summary = df.groupby("split").agg(
        n=("name", "size"),
        groups=("group", "nunique"),
        signal_q25=("signal", lambda x: x.quantile(0.25)),
        signal_median=("signal", "median"),
        signal_q75=("signal", lambda x: x.quantile(0.75)),
        signal_q90=("signal", lambda x: x.quantile(0.9)),
    ).reindex(order)
    summary["frac"] = summary["n"] / len(df)
    print(summary.round(3).to_string(), "\n")

    print("Family counts per split:")
    print(pd.crosstab(df["family"], df["split"]).reindex(columns=order, fill_value=0).to_string())

    for a in order:
        for b in order:
            if a < b:
                shared = set(df.loc[df["split"] == a, "group"]) & set(df.loc[df["split"] == b, "group"])
                assert not shared, f"groups shared between {a} and {b}"


if __name__ == "__main__":
    main()
