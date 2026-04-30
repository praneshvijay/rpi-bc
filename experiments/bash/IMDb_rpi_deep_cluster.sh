#!/bin/bash
#SBATCH --job-name=rpi_deep
#SBATCH --output=/home/bt2/22CS10013/attention-based-credit/rpi_deep_out_%A_%a.txt
#SBATCH --error=/home/bt2/22CS10013/attention-based-credit/rpi_deep_err_%A_%a.txt
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_h100
#SBATCH --array=0-4   # 5 seeds

REPO=/home/bt2/22CS10013/attention-based-credit
SEED=$((SLURM_ARRAY_TASK_ID + 1))

echo "Running deep RPIBC seed=${SEED} on node=$(hostname)"

source $REPO/venv/bin/activate
export PYTHONPATH=$REPO
export WANDB_MODE=offline

python3 $REPO/experiments/scripts/rlhf_imdb_rpi_deep.py \
    --seed ${SEED} \
    --rpi_time_steps 5 \
    --rpi_num_interpolation 10 \
    --rpi_num_samples 1 \
    --max_epochs 150 \
    --beta 0.8 \
    --batch_size 16 \
    --project_name rlhf_imdb_rpi_deep

echo "Finished seed=${SEED}"
