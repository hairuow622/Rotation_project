#!/bin/bash
#SBATCH --job-name=FOXA1_ChIPISO_ft
#SBATCH --partition=mahony
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/FOXA1_ChIPISO_ft_%j.out
#SBATCH --error=logs/FOXA1_ChIPISO_ft_%j.err

# Unfreezing experiment from doc/chip-iso_fine-tuning.md.
# Usage: sbatch --export=ALL,PREFIX=<name> submit_finetune.sh [steps...]
#   steps: level1 level2 level3 level4 level5 level6 summary
#   (default: all, in this order). level1..level6 are independent and can run as separate jobs;
#   summary needs all of them.

set -eo pipefail

cd /home/hvw5476/group/lab/hairuow/rotation_project

source /home/hvw5476/miniforge3/etc/profile.d/conda.sh
conda activate rotation

export PYTHONUNBUFFERED=1

CONFIG="finetune_chipiso_config.yaml"
PREFIX="${PREFIX:-FOXA1_ChIPISO_levels}"
SUMMARY_TSV="checkpoints/${PREFIX}_levels_summary.tsv"
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(level1 level2 level3 level4 level5 level6 summary)

if [ ! -f ChIP-ISO_Lucy_lab/ChIP-ISO_signal/chipiso_split.tsv ]; then
  python split_chipiso.py
fi

best_ckpt() {
  ls -t "checkpoints/$1"/best-*.ckpt | head -n 1
}

# --config must come first: later arguments override values from the config file.
# EXTRA_ARGS (optional) is appended to every call, e.g. "--trainer.max_epochs 1" for a quick check.
run() {
  local run_name=$1 subcommand=$2; shift 2
  echo "===== ${run_name}: python models.py ${subcommand} $* ${EXTRA_ARGS} ====="
  # shellcheck disable=SC2086
  RUN_NAME="${run_name}" python models.py "${subcommand}" \
    --config "${CONFIG}" \
    --trainer.default_root_dir "checkpoints/${run_name}" \
    --trainer.logger.init_args.name "${run_name}" \
    --trainer.logger.init_args.group "${PREFIX}" \
    --trainer.logger.init_args.job_type "${subcommand}" \
    "$@" ${EXTRA_ARGS}
}

# fit one level, then test its best checkpoint on ft_test
fit_and_test() {
  local level=$1; shift
  local name="${PREFIX}_level${level}"
  run "${name}" fit "$@"
  run "${name}_test" test "$@" --ckpt_path "$(best_ckpt "${name}")"
}

for step in "${STEPS[@]}"; do
  case "${step}" in
    level1)
      run "${PREFIX}_level1_test" test --model.init_args.calibrate_head true
      ;;
    level2|level3|level4|level5|level6)
      level=${step#level}
      fit_and_test "${level}" \
        --model.init_args.calibrate_head true \
        --model.init_args.unfreeze_level $((level - 2))
      ;;
    summary)
      python chipiso_levels.py --out "${SUMMARY_TSV}" --runs \
        1=zeroshot \
        2="checkpoints/${PREFIX}_level2" \
        3="checkpoints/${PREFIX}_level3" \
        4="checkpoints/${PREFIX}_level4" \
        5="checkpoints/${PREFIX}_level5" \
        6="checkpoints/${PREFIX}_level6"
      ;;
    *)
      echo "Unknown step: ${step}" >&2
      exit 1
      ;;
  esac
done
