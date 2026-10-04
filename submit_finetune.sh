#!/bin/bash
#SBATCH --job-name=FOXA1_ChIPISO_ft
#SBATCH --partition=mahony
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/FOXA1_ChIPISO_ft_%j.out
#SBATCH --error=logs/FOXA1_ChIPISO_ft_%j.err

# Usage: sbatch submit_finetune.sh [steps...]
#   steps: zeroshot stage1 stage2 alpool (default: all, in this order)

set -eo pipefail

cd /home/hvw5476/group/lab/hairuow/rotation_project

source /home/hvw5476/miniforge3/etc/profile.d/conda.sh
conda activate rotation

export PYTHONUNBUFFERED=1

CONFIG="finetune_chipiso_config.yaml"
PREFIX="${PREFIX:-FOXA1_ChIPISO}"
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(zeroshot stage1 stage2 alpool)

if [ ! -f ChIP-ISO_Lucy_lab/ChIP-ISO_signal/chipiso_split.tsv ]; then
  python split_chipiso.py
fi

best_ckpt() {
  ls -t "checkpoints/$1"/best-*.ckpt | head -n 1
}

# --config must come first: later arguments override values from the config file
run() {
  local run_name=$1 subcommand=$2; shift 2
  echo "===== ${run_name}: python models.py ${subcommand} $* ====="
  RUN_NAME="${run_name}" python models.py "${subcommand}" \
    --config "${CONFIG}" \
    --trainer.default_root_dir "checkpoints/${run_name}" \
    --trainer.logger.init_args.name "${run_name}" \
    "$@"
}

for step in "${STEPS[@]}"; do
  case "${step}" in
    zeroshot)
      run "${PREFIX}_zeroshot" test
      ;;
    stage1)
      run "${PREFIX}_stage1" fit
      run "${PREFIX}_stage1_test" test --ckpt_path "$(best_ckpt "${PREFIX}_stage1")"
      ;;
    stage2)
      STAGE1_CKPT=$(best_ckpt "${PREFIX}_stage1")
      run "${PREFIX}_stage2" fit \
        --model.init_args.pretrained_ckpt "${STAGE1_CKPT}" \
        --model.init_args.freeze_backbone false \
        --model.init_args.calibrate_head false
      run "${PREFIX}_stage2_test" test \
        --model.init_args.freeze_backbone false \
        --ckpt_path "$(best_ckpt "${PREFIX}_stage2")"
      ;;
    alpool)
      run "${PREFIX}_alpool_pred" test \
        --model.init_args.freeze_backbone false \
        --data.init_args.test_split al_pool \
        --ckpt_path "$(best_ckpt "${PREFIX}_stage2")"
      ;;
    *)
      echo "Unknown step: ${step}" >&2
      exit 1
      ;;
  esac
done
