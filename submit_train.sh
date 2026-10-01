#!/bin/bash
#SBATCH --job-name=FOXA1_v6
#SBATCH --partition=mahony
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/FOXA1_train_%j.out
#SBATCH --error=logs/FOXA1_train_%j.err

set -eo pipefail

cd /home/hvw5476/group/lab/hairuow/rotation_project

source /home/hvw5476/miniforge3/etc/profile.d/conda.sh
conda activate rotation

export PYTHONUNBUFFERED=1

RUN_NAME="FOXA1_A549_rep1_center_win240_run1"
DATA_CONFIG="ChIP-ISO_Lucy_lab/FoxA1_ChIP_Replicate1_Data/FOXA1_A549_rep1_center_win240/post_run.yaml"

echo "RUN_NAME=${RUN_NAME}"

python models.py fit \
  --config model_config.yaml \
  --data.config_file "${DATA_CONFIG}" \
  --trainer.default_root_dir "checkpoints/${RUN_NAME}" \
  --trainer.logger.init_args.name "${RUN_NAME}"