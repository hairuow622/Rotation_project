#!/bin/bash
#SBATCH --job-name=FOXA1_test
#SBATCH --partition=mahony
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/FOXA1_test_%j.out
#SBATCH --error=logs/FOXA1_test_%j.err

# Test the win240 best checkpoint on FOXA1_A549_rep1_center_win240.
# Metrics and ROC/PR figures go to TensorBoard under tb_logs/${RUN_NAME}/version_*.
#
# Submit:
#   sbatch submit_test.sh
# View:
#   export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
#   tensorboard --logdir tb_logs/FOXA1_A549_rep1_center_win240_wd1e-4_test --port 6006 --host 127.0.0.1

set -eo pipefail

cd /home/hvw5476/group/lab/hairuow/rotation_project
mkdir -p logs

source /home/hvw5476/miniforge3/etc/profile.d/conda.sh
conda activate rotation

export PYTHONUNBUFFERED=1
# TensorBoard 2.11 and this env's protobuf are incompatible unless the pure-Python implementation is used.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

MODEL_CONFIG="model_config.yaml"
DATA_CONFIG="ChIP-ISO_Lucy_lab/FoxA1_ChIP_Replicate1_Data/FOXA1_A549_rep1_center_win240/post_run.yaml"
CKPT="checkpoints/FOXA1_A549_rep1_center_win240_wd1e-4/best-epoch=40-val_loss=0.344953.ckpt"
export RUN_NAME="${RUN_NAME:-FOXA1_A549_rep1_center_win240_wd1e-4_test}"

echo "RUN_NAME=${RUN_NAME}"
echo "CKPT=${CKPT}"
echo "DATA_CONFIG=${DATA_CONFIG}"

python models.py test \
  --config "${MODEL_CONFIG}" \
  --data.config_file "${DATA_CONFIG}" \
  --ckpt_path "${CKPT}" \
  --trainer.devices 1 \
  --trainer.strategy auto \
  --trainer.logger.class_path pytorch_lightning.loggers.TensorBoardLogger \
  --trainer.logger.init_args.save_dir tb_logs \
  --trainer.logger.init_args.name "${RUN_NAME}" \
  --trainer.logger.init_args.default_hp_metric false
