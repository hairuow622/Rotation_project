#!/usr/bin/env python

"""
dataset_v2.py

Compared to original dataset.py, add gc matched and motif matched regions into training set

read count predictor training set:
    - bound shift (1k)
    - unbound gc match (1k)
    - unbound flanking (2k)
    - unbound motif match (1k)
    - unbound random (1k)
domain discriminator training set:
    - accessible region (1k)
    - motif match region (1k)

* k represents the base number of samples
"""

import argparse
import logging
import os
from functools import partial

import numpy as np
import pandas as pd
import pysam
import seqchromloader
import yaml
from Bio import motifs
from pybedtools import BedTool, Interval
import pybedtools
from seqchromloader import dump_data_webdataset, utils
pybedtools.set_tempdir('/home/hvw5476/tmp')
DEFAULT_TRANSFORM_STR = (
    "{'target': lambda t: np.log(t+1), 'label': lambda l: l.astype(np.float32)}"
)

SEED = 223344
RNG = np.random.default_rng(SEED)

logging.basicConfig(level=logging.DEBUG)


def mark_region(regions, bdt, win=256):
    "mark regions whose center win regions intersect with bdt"
    half_win = int(win / 2)
    regions_win = regions.assign(
        midpoint=lambda x: ((x["start"] + x["end"]) / 2).astype(int)
    ).assign(
        start=lambda x: x["midpoint"] - half_win, end=lambda x: x["midpoint"] + half_win
    )[
        ["chrom", "start", "end"]
    ]
    regions_win.loc[regions_win.start < 0, "start"] = (
        0  # set negative start coordinates as 0
    )
    regions_win.loc[regions_win.end < 0, "end"] = 0  # set negative end coordinates as 0
    regions_win_bdt = BedTool.from_dataframe(regions_win)

    intersect = regions_win_bdt.intersect(bdt, c=True).to_dataframe()["name"] > 0

    return intersect


def remove_ambiguous_region(regions, chip_coords_bdt, center_win=256, train_win=1024):
    "Remove ambiguous regions that center regions don't intersects with peak but training regions do"

    intersect_center = mark_region(regions, chip_coords_bdt, win=center_win)
    intersect_train = mark_region(regions, chip_coords_bdt, win=train_win)

    return regions[~((~intersect_center) & (intersect_train))]


def define_random_coordinates(
    chip_coords: pd.DataFrame,
    genome_size_file: str,
    curr_genome_bdt: BedTool,
    blacklist_bdt: BedTool,
    l: int,
    input_window_length: int,
    n: int,
):

    chip_coords_bdt = BedTool.from_dataframe(chip_coords)

    random_regions = utils.random_coords(
        gs=genome_size_file,
        l=l,
        n=n,
        incl=curr_genome_bdt,
        excl=blacklist_bdt,
        seed=SEED,
    )

    random_regions = remove_ambiguous_region(
        random_regions,
        chip_coords_bdt,
        center_win=l,
        train_win=input_window_length,
    )

    random_regions = (
        BedTool.from_dataframe(random_regions)
        .intersect(chip_coords_bdt, c=True)
        .to_dataframe()
        .rename({"name": "label"}, axis=1)
        .assign(type="random_genome")
    )
    random_regions["label"] = (random_regions["label"] > 0) * 1

    return random_regions


def define_training_coordinates(
    chip_coords: pd.DataFrame,
    genome_fasta_file: str,
    genome_sizes_file: str,
    curr_genome_bdt: BedTool,
    blacklist_bdt: BedTool,
    motif: motifs.Motif | None,
    L,
    input_window_length: int,
    bound_shift_factor: float,
    unbound_random_factor: float,
    unbound_gc_factor: float,
    unbound_motif_factor: float,
    unbound_flank_factor: float,
    split=False,
):
    """
    Use the chip-seq peak file and the blacklist files to define a bound
    set and an unbound set of sites.
    The unbound/negative set is chosen randomly from the genome
    Label marks where the sample is from 0(source)/1(target)
    """

    chip_coords_num = len(chip_coords)
    chip_coords_bdt = BedTool.from_dataframe(chip_coords)

    # POS. SAMPLES
    # Take a sample from the chip_coords file,
    # Then apply a random shift.
    # Create a BedTool object for further use.
    bound_sample_shift = (
        chip_coords.sample(
            n=int(bound_shift_factor * chip_coords_num), replace=True, random_state=RNG
        )
        .pipe(utils.make_random_shift, L, buffer=25, rng=RNG)
        .query("start >= 0")
    )
    bound_sample_bdt_obj = BedTool.from_dataframe(bound_sample_shift).intersect(
        blacklist_bdt, v=True
    )
    bound_sample_shift = bound_sample_bdt_obj.to_dataframe().assign(type="pos_shift")

    logging.debug(f"Bound samples in total: {bound_sample_bdt_obj.count()}")

    # NEG. SAMPLES: RANDOM ACROSS WHOLE GENOME
    unbound_genome_df = utils.random_coords(
        gs=genome_sizes_file,
        l=L,
        n=int(unbound_random_factor * chip_coords_num),
        incl=curr_genome_bdt,
        excl=chip_coords_bdt.cat(blacklist_bdt),
        seed=SEED,
    ).assign(type="neg_random")

    # NEG. SAMPLES: GC MATCH
    unbound_gc_df = seqchromloader.make_gc_match(
        chip_coords_bdt.to_dataframe(),
        genome_fa=genome_fasta_file,
        l=L,
        n=int(unbound_gc_factor * chip_coords_num),
        seed=SEED,
        gc_diff_max=0.10,
        incl=curr_genome_bdt,
        excl=chip_coords_bdt.cat(blacklist_bdt),
    ).assign(type="neg_gc")

    if motif is not None:
        unbound_motif_df = seqchromloader.make_motif_match(
            motif,
            genome_fa=genome_fasta_file,
            l=L,
            n=int(unbound_motif_factor * chip_coords_num),
            seed=SEED,
            incl=curr_genome_bdt,
            excl=chip_coords_bdt.cat(blacklist_bdt),
        ).assign(type="neg_motif")
    else:
        unbound_motif_df = None

    unbound_flank_df = pd.concat(
        [
            seqchromloader.make_flank(
                chip_coords.sample(
                    n=int(unbound_flank_factor * chip_coords_num / 4),
                    replace=True,
                    random_state=RNG,
                )
                .pipe(utils.make_random_shift, L, rng=RNG)
                .query("start >= 0"),
                L=L,
                d=2 * L,
            ).assign(type="neg_flank"),
            seqchromloader.make_flank(
                chip_coords.sample(
                    n=int(unbound_flank_factor * chip_coords_num / 4),
                    replace=True,
                    random_state=RNG,
                )
                .pipe(utils.make_random_shift, L, rng=RNG)
                .query("start >= 0"),
                L=L,
                d=-2 * L,
            ).assign(type="neg_flank"),
            seqchromloader.make_flank(
                chip_coords.sample(
                    n=int(unbound_flank_factor * chip_coords_num / 4),
                    replace=True,
                    random_state=RNG,
                )
                .pipe(utils.make_random_shift, L, rng=RNG)
                .query("start >= 0"),
                L=L,
                d=4 * L,
            ).assign(type="neg_flank"),
            seqchromloader.make_flank(
                chip_coords.sample(
                    n=int(unbound_flank_factor * chip_coords_num / 4),
                    replace=True,
                    random_state=RNG,
                )
                .pipe(utils.make_random_shift, L, rng=RNG)
                .query("start >= 0"),
                L=L,
                d=-4 * L,
            ).assign(type="neg_flank"),
        ]
    )

    # Merge training set
    ## merge unbound sets
    unbound = (
        pd.concat(
            [unbound_genome_df, unbound_gc_df, unbound_motif_df, unbound_flank_df],
            ignore_index=True,
        )
        .query("start >= 0")
        .reset_index(drop=True)
    )
    ## remove unbound regions whose model input windows intersect with chip peaks
    unbound = unbound[~mark_region(unbound, chip_coords_bdt, win=input_window_length)]
    ## merge all sets
    training_coords_bichrom = pd.concat(
        [bound_sample_shift, unbound], ignore_index=True
    )

    # intersect with chip coords to get label
    training_coords_bichrom_intersect = (
        BedTool.from_dataframe(training_coords_bichrom[["chrom", "start", "end"]])
        .intersect(chip_coords_bdt, c=True)
        .to_dataframe()
    )
    training_coords_bichrom["label"] = (
        training_coords_bichrom_intersect["name"] > 0
    ) * 1

    # sanity check
    num_false_neg_labels = len(
        training_coords_bichrom[
            (training_coords_bichrom.type.str.contains("pos"))
            & (training_coords_bichrom.label == 0)
        ]
    )
    num_false_pos_labels = len(
        training_coords_bichrom[
            (training_coords_bichrom.type.str.contains("neg"))
            & (training_coords_bichrom.label == 1)
        ]
    )
    logging.debug(f"There are {num_false_pos_labels} false positive labels!")
    logging.debug(f"There are {num_false_neg_labels} false negative labels!")

    # logging summary
    logging.debug(training_coords_bichrom.groupby(["label", "type"]).size())

    # shuffle
    training_coords_bichrom.sample(frac=1, random_state=RNG)

    if split:
        return (
            training_coords_bichrom.loc[training_coords_bichrom.label == 1],
            training_coords_bichrom.loc[training_coords_bichrom.label == 0],
        )
    else:
        return training_coords_bichrom  # randomly shuffle the dataFrame


def define_domain_task_coordinates_new(
    genome_sizes_file: str,
    genome_fasta_file: str,
    chip_coords: pd.DataFrame,
    acc_bdt: BedTool,
    curr_genome_bdt: BedTool,
    blacklist_bdt: BedTool,
    motif,
    L: int,
    input_window_length: int,
    n: int,
):

    n_quota = n // 3 if motif is not None else n // 2
    chip_coords_bdt = BedTool.from_dataframe(chip_coords)

    # ~~~ CHANGED: wrap both acc_bdt usages in if/else ~~~
    if acc_bdt is not None:
        random_accessible_coords_df = utils.random_coords(
            gs=genome_sizes_file,
            l=L,
            n=n_quota,
            incl=acc_bdt.intersect(curr_genome_bdt),  # unchanged
            excl=blacklist_bdt,
            seed=SEED,
        ).assign(type="accessible_for_domain_task")

        random_inaccessible_coords_df = utils.random_coords(
            gs=genome_sizes_file,
            l=L,
            n=n_quota,
            incl=curr_genome_bdt,
            excl=acc_bdt.cat(blacklist_bdt),           # unchanged
            seed=SEED,
        ).assign(type="inaccessible_for_domain_task")
    else:
        # ~~~ ADDED: fallback when no accessibility data provided ~~~
        random_accessible_coords_df = utils.random_coords(
            gs=genome_sizes_file,
            l=L,
            n=n_quota,
            incl=curr_genome_bdt,
            excl=blacklist_bdt,
            seed=SEED,
        ).assign(type="random_for_domain_task")

        random_inaccessible_coords_df = None  # ~~~ ADDED ~~~
    # ~~~ END CHANGED ~~~


    if motif is not None:
        motif_df = seqchromloader.make_motif_match(
            motif,
            genome_fa=genome_fasta_file,
            l=L,
            n=n_quota,
            seed=SEED,
            incl=curr_genome_bdt,
            excl=blacklist_bdt,
            threshold=0,
        ).assign(type="motif_match_for_domain_task")
    else:
        motif_df = None

    domain_coords_df = pd.concat(
        [df for df in [random_accessible_coords_df, random_inaccessible_coords_df, motif_df]
         if df is not None],  # ~~~ ADDED: None filter ~~~
        ignore_index=True,
    ).sample(frac=1, random_state=RNG, ignore_index=True)

    # remove ambiguous regions
    domain_coords_df = remove_ambiguous_region(
        domain_coords_df,
        chip_coords_bdt,
        center_win=L,
        train_win=input_window_length,
    )
    # assign labels based on if intervals overlap with chip-seq peak or not
    domain_coords_df = (
        BedTool.from_dataframe(domain_coords_df)
        .intersect(chip_coords_bdt, c=True)
        .to_dataframe()
        .rename(columns={"name": "type", "score": "label"})
    )
    domain_coords_df["label"] = (domain_coords_df["label"] > 0) * 1

    # logging summary
    logging.debug(domain_coords_df.groupby(["label", "type"]).size())

    return domain_coords_df


def genome_size_to_bdt(genome_size_df):
    genome_bed_data = []
    for item in genome_size_df.itertuples():
        genome_bed_data.append(Interval(item.chrom, 0, item.length))
    genome_bed_data = BedTool(genome_bed_data)

    return genome_bed_data


def define_coordinates_in_one_cell(
    chip_peak_file: str,
    dnase_peak_file: str,
    genome_fasta_file: str,
    genome_size_file: str,
    blacklist_file: str,
    motif_file: str,
    augment_goal: int,
    window_length: int,
    input_window_length: int,
    val_chrom: list,
    test_chrom: list,
    generate_domain_data: bool,
    out_prefix: str,
    out_dir: str,
):

    chip_seq_coordinates = pd.read_table(
        chip_peak_file, header=None, usecols=range(3), names=["chrom", "start", "end"]
    )

    acc_bdt = (
        BedTool(dnase_peak_file)
        if generate_domain_data and dnase_peak_file is not None
        else None
    )
    blacklist_bdt = BedTool(blacklist_file)
    motif = (
        motifs.parse(open(motif_file, "r"), "jaspar")[0]
        if motif_file is not None
        else None
    )
    val_chrom = val_chrom or [] # handle None case for val_chrom

    train_genome_bdt = genome_size_to_bdt(
        utils.get_genome_sizes(
            genome_size_file, to_keep=None, to_filter=val_chrom + test_chrom
        )
    )
    val_genome_bdt = (
        genome_size_to_bdt(
            utils.get_genome_sizes(genome_size_file, to_keep=val_chrom, to_filter=None)
        )
        if val_chrom
        else None
    )  # handle None case for val_chrom
    
    test_genome_bdt = genome_size_to_bdt(
        utils.get_genome_sizes(genome_size_file, to_keep=test_chrom, to_filter=None)
    )

    # Define training coordinates
    def filter_and_define_coordinates(
        chip_coords,
        curr_genome_bdt,
        to_keep=None,
        to_filter=None,
        bound_shift_factor=1.0,
        unbound_random_factor=1.0,
        unbound_gc_factor=1.0,
        unbound_motif_factor=1.0,
        unbound_flank_factor=1.0,
        split=False,
    ):
        chip_coords_filter = utils.filter_chromosomes(
            chip_coords, to_filter=to_filter, to_keep=to_keep
        )
        return define_training_coordinates(
            chip_coords_filter,
            genome_fasta_file,
            genome_size_file,
            curr_genome_bdt,
            blacklist_bdt,
            motif=motif,
            L=window_length,
            input_window_length=input_window_length,
            bound_shift_factor=bound_shift_factor,
            unbound_random_factor=unbound_random_factor,
            unbound_gc_factor=unbound_gc_factor,
            unbound_motif_factor=unbound_motif_factor,
            unbound_flank_factor=unbound_flank_factor,
            split=split,
        )

    # compute a global augment factor using # chip-seq peak regions
    bound_shift_factor = float(augment_goal) / len(chip_seq_coordinates)
    augment_factors = {
        "bound_shift_factor": bound_shift_factor,
        "unbound_random_factor": bound_shift_factor,
        "unbound_gc_factor": bound_shift_factor,
        "unbound_motif_factor": bound_shift_factor,
        "unbound_flank_factor": bound_shift_factor,
    }

    training_coords_readcount_bound, training_coords_readcount_unbound = (
        filter_and_define_coordinates(
            chip_seq_coordinates,
            train_genome_bdt,
            to_filter=val_chrom + test_chrom,
            to_keep=None,
            split=True,
            **augment_factors,
        )
    )

    validation_coords_readcount = (
        filter_and_define_coordinates(
            chip_seq_coordinates,
            val_genome_bdt,
            to_keep=val_chrom,
            to_filter=None,
            split=False,
            **augment_factors,
        )
        if val_chrom
        else None
    )  # handle None case for val_chrom

    test_coords_readcount = filter_and_define_coordinates(
        chip_seq_coordinates,
        test_genome_bdt,
        to_keep=test_chrom,
        to_filter=None,
        split=False,
        **augment_factors,
    )

    test_coords_random = define_random_coordinates(
        chip_seq_coordinates,
        genome_size_file,
        test_genome_bdt,
        blacklist_bdt,
        l=window_length,
        input_window_length=input_window_length,
        n=100000,
    )

    training_coords_domain = None
    if generate_domain_data:
        domain_sample_size = max(
            len(training_coords_readcount_bound),
            len(training_coords_readcount_unbound),
        )
        training_coords_domain = define_domain_task_coordinates_new(
            chip_coords=chip_seq_coordinates,
            genome_sizes_file=genome_size_file,
            genome_fasta_file=genome_fasta_file,
            acc_bdt=acc_bdt,
            curr_genome_bdt=train_genome_bdt,
            blacklist_bdt=blacklist_bdt,
            motif=motif,
            L=window_length,
            input_window_length=input_window_length,
            n=domain_sample_size,
        )

    # save filenames into YAML
    df_config = {
        f"{out_prefix}_train_readcount_bound": training_coords_readcount_bound,
        f"{out_prefix}_train_readcount_unbound": training_coords_readcount_unbound,
        f"{out_prefix}_val_readcount": validation_coords_readcount,
        f"{out_prefix}_test_readcount": test_coords_readcount,
        f"{out_prefix}_test_random": test_coords_random,
    }
    if training_coords_domain is not None:
        df_config[f"{out_prefix}_train_domain"] = training_coords_domain

    # Assign strand information to the coordinates
    def forwardReverse(df):
        return pd.concat(
            [df.assign(strand="+"), df.assign(strand="-")], ignore_index=True
        ).reset_index(inplace=False, drop=True)

    bed_config = {}
    for key, df in df_config.items():
        if df is not None:
            df_config[key] = {}
            bed_config[key] = {}
            for t in df.type.unique():
                df_config[key][t] = forwardReverse(df.loc[df.type == t])
                bed_config[key][t] = os.path.join(
                    out_dir, f"{key}_{t}.bed"
                )  # create the bed file name dict
                df_config[key][t].to_csv(
                    bed_config[key][t], header=False, sep="\t", index=False
                )  # save the augmented dataframe into bed file
        else:
            bed_config[key] = None

    return bed_config, df_config


def writeWDS(
    df,
    genome_fasta,
    bigwigs,
    chip_bam,
    chip_bw,
    patch_left,
    patch_right,
    out_dir,
    key,
    transforms=None,
    processors=10,
    batch_size=None,
):
    """
    Given dataframe and bigwig and bam files, write webdataset
    """
    if chip_bam is not None and chip_bw is not None:
        raise Warning(
            "Only chip_bam will be used given both chip_bam and chip_bw as target!"
        )
    wds_files = dump_data_webdataset(
        df,
        genome_fasta,
        bigwig_filelist=bigwigs,
        target_bam=chip_bam,
        target_bw=chip_bw,
        patch_left=patch_left,
        patch_right=patch_right,
        outdir=out_dir,
        outprefix=key,
        compress=False,
        numProcessors=processors,
        transforms=transforms,
        braceexpand=True,
        samples_per_tar=int(5e3),
        batch_size=batch_size,
    )
    return wds_files


def default_chroms_transform(c, mean: list, std: list):
    return (c - np.array(mean, dtype=np.float32)[:, np.newaxis]) / np.array(
        std, dtype=np.float32
    )[:, np.newaxis]


def default_target_bam_transform(t, scale=1.0, vlog=True):
    return np.log(t / scale + 1) if vlog else t / scale


def default_target_bw_transform(t):
    print(t)
    return np.sum(t)[np.newaxis]


def default_label_transform(l):
    return l.astype(np.float32)


def get_mean_and_std(bws):
    ms = []
    stds = []
    for bwf in bws:
        mean, std = utils.compute_mean_std_bigwig(bwf)
        ms.append(mean)
        stds.append(std)
        print(f"{bwf} Mean: {mean}, Standard Deviation {std} from {bwf}")
    return ms, stds


def standardize_transform(
    bigwigs, chip_bam, chip_bw, standardizeChrom=True, normalizeBAM=True
):
    "return tranform library according to the standardization and normalization booleans"

    transforms = {"label": default_label_transform}
    if standardizeChrom and bigwigs:
        bws_mean, bws_std = get_mean_and_std(bigwigs)
        transforms["chrom"] = partial(
            default_chroms_transform, mean=bws_mean, std=bws_std
        )

    if normalizeBAM and chip_bam is not None:
        # compute scaling factor
        scale = float(pysam.AlignmentFile(chip_bam).mapped) / float(1e6)
        print(f"Scaling factor for bam: {scale}")
        transforms["target"] = partial(
            default_target_bam_transform, scale=scale, vlog=True
        )
    elif chip_bw is not None:
        transforms["target"] = default_target_bw_transform

    return transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="YAML config file on input files")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument(
        "--wds",
        action="store_true",
        default=False,
        help="If save into webdataset, pls provide bigwig and bam files if set True",
    )
    parser.add_argument(
        "-p", type=int, default=10, help="Number of processors for writing wds files"
    )
    parser.add_argument(
        "--normalizeBAM",
        action="store_true",
        default=False,
        help="Normalize target bam by RPM",
    )
    args = parser.parse_args()

    # make output directories
    print(f"Create directory {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    # load config file
    config = yaml.safe_load(open(args.config, "r"))
    print(f"standardizeChrom: {config.get('standardizeChrom', True)}")

    def transform_and_write_webdataset(
        coords,
        bigwigs,
        chip_bam,
        chip_bw,
        genome_fasta_file,
        key,
        standardizeChrom: bool,
        normalizeBAM: bool,
        out_dir: str,
        p: int,
    ):

        transforms = standardize_transform(
            bigwigs, chip_bam, chip_bw, standardizeChrom, normalizeBAM
        )

        patch = int(
            (config["input_window_length"] - config["target_window_length"]) / 2
        )
        wds_files = writeWDS(
            df=coords,
            genome_fasta=genome_fasta_file,
            bigwigs=bigwigs,
            chip_bam=chip_bam,
            chip_bw=chip_bw,
            patch_left=patch,
            patch_right=patch,
            out_dir=out_dir,
            key=key,
            transforms=transforms,
            processors=p,
            batch_size=128,
        )

        return wds_files

    # construct dataset in single cell type
    coords_bed_dict, coords_df_dict = define_coordinates_in_one_cell(
        chip_peak_file=config["post_chip_peak"],
        dnase_peak_file=config.get("pre_acc_peak"),
        genome_fasta_file=config["genome_fasta_file"],
        genome_size_file=config["genome_size_file"],
        blacklist_file=config["blacklist_file"],
        motif_file=config.get("motif_file"),
        augment_goal=config["augment_goal"],
        window_length=config["target_window_length"],
        input_window_length=config["input_window_length"],
        val_chrom=config.get("val_chrom"),
        test_chrom=config["test_chrom"],
        generate_domain_data=config.get("generate_domain_data", False),
        out_prefix="single",
        out_dir=args.output,
    )

    config["bed"] = coords_bed_dict

    if args.wds:
        genome_fasta_file = config["genome_fasta_file"]
        bigwigs = config.get("pre_bws", [])
        post_chip_bam_file = None
        post_chip_bw_file = None
        if config.get("post_chip_bam") is not None:
            post_chip_bam_file = config["post_chip_bam"]
        elif config.get("post_chip_bw") is not None:
            post_chip_bw_file = config["post_chip_bw"]

        config["webdataset"] = {}

        for key, df in coords_df_dict.items():
            config["webdataset"][key] = {}
            for t, df in coords_df_dict[key].items():
                wds_files = transform_and_write_webdataset(
                    coords=df,
                    bigwigs=bigwigs,
                    chip_bam=post_chip_bam_file,
                    chip_bw=post_chip_bw_file,
                    genome_fasta_file=genome_fasta_file,
                    key=f"{key}_{t}",
                    standardizeChrom=config.get("standardizeChrom", True),
                    normalizeBAM=args.normalizeBAM,
                    out_dir=args.output,
                    p=args.p,
                )
                config["webdataset"][key][t] = wds_files

        post_run_path = os.path.join(args.output, "post_run.yaml")
        with open(post_run_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Wrote {post_run_path}")