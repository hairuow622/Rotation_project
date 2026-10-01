#!/bin/bash
#SBATCH --job-name=FOXA1_v6
#SBATCH --partition=mahony
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/FOXA1_train_%j.out
#SBATCH --error=logs/FOXA1_train_%j.err

set -euo pipefail

mkdir -p logs

cd /home/hvw5476/group/lab/hairuow/rotation_project

source /home/hvw5476/miniforge3/etc/profile.d/conda.sh
conda activate rotation

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

export RUN_NAME="${RUN_NAME:-FOXA1_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "checkpoints/${RUN_NAME}"
echo "RUN_NAME=${RUN_NAME}"
echo "W&B project: Rotation_project (entity: hairuow-carnegie-mellon-university)"

python models.py fit \
  --config model_config.yaml \
  --data.config_file data_config.post_run.yaml \
  --trainer.logger.init_args.name "${RUN_NAME}"
