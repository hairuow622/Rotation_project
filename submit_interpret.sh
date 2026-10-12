#!/bin/bash
#SBATCH --job-name=FOXA1_interpret
#SBATCH --partition=mahony
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/FOXA1_interpret_%j.out
#SBATCH --error=logs/FOXA1_interpret_%j.err

# Motif discovery on the win240 FOXA1 classifier, see doc/foxa1_motif_attribution.md.
# Outputs go to results/interpret/FOXA1_A549_win240/.
#
# Submit everything:
#   sbatch submit_interpret.sh
# Or only some stages (any of: prep attr diag modisco annotate finemo summary):
#   STAGES="modisco annotate finemo summary" sbatch submit_interpret.sh

set -eo pipefail

cd /home/hvw5476/group/lab/hairuow/rotation_project
mkdir -p logs

source /home/hvw5476/miniforge3/etc/profile.d/conda.sh
conda activate rotation

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"

OUT="${OUT:-results/interpret/FOXA1_A549_win240}"
JASPAR="resources/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
STAGES="${STAGES:-prep attr diag modisco annotate finemo summary}"
mkdir -p "${OUT}"

run() { [[ " ${STAGES} " == *" $1 "* ]]; }

echo "OUT=${OUT}"
echo "STAGES=${STAGES}"

if run prep; then
  python interpret/prepare_seqs.py --out_dir "${OUT}"
fi

if run attr; then
  python interpret/run_deeplift.py --out_dir "${OUT}"
fi

if run diag; then
  python interpret/diagnose_attr.py --out_dir "${OUT}"
fi

if run modisco; then
  modisco motifs -s "${OUT}/ohe.npz" -a "${OUT}/hyp.npz" -n 20000 -w 240 -o "${OUT}/modisco_results.h5" -v
  modisco report -i "${OUT}/modisco_results.h5" -o "${OUT}/modisco_report/" -s "${OUT}/modisco_report/" \
    -m "${JASPAR}" -n 3
fi

if run annotate; then
  python interpret/annotate_patterns.py --out_dir "${OUT}" --meme_db "${JASPAR}"
fi

if run finemo; then
  python interpret/finemo_compat.py extract-regions-modisco-fmt -s "${OUT}/ohe.npz" -a "${OUT}/hyp.npz" -p "${OUT}/regions.bed" \
    -w 240 -o "${OUT}/finemo_regions.npz"
  python interpret/finemo_compat.py call-hits -r "${OUT}/finemo_regions.npz" -m "${OUT}/modisco_results.h5" \
    -o "${OUT}/finemo/"
  python interpret/finemo_compat.py report -r "${OUT}/finemo_regions.npz" -H "${OUT}/finemo/" -p "${OUT}/regions.bed" \
    -m "${OUT}/modisco_results.h5" -W 240 -o "${OUT}/finemo_report/"
fi

if run summary; then
  python interpret/summarize_motifs.py --out_dir "${OUT}"
fi
