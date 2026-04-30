#!/bin/bash
#SBATCH --job-name=tldr_deep
#SBATCH --output=/home/bt2/22CS10013/attention-based-credit/tldr_rpi_deep_out_%A_%a.txt
#SBATCH --error=/home/bt2/22CS10013/attention-based-credit/tldr_rpi_deep_err_%A_%a.txt
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_h100
#SBATCH --array=0-4   # 5 seeds — resubmit if 12h isn't enough

set -e


REPO=/home/bt2/22CS10013/attention-based-credit
SEED=$((SLURM_ARRAY_TASK_ID + 1))

echo "Running Deep RPIBC TL;DR seed=${SEED} on node=$(hostname)"

source $REPO/venv/bin/activate
export PYTHONPATH=$REPO
export WANDB_MODE=offline

python3 $REPO/experiments/scripts/rlhf_tldr_rpi_deep.py \
    --seed ${SEED} \
    --rpi_time_steps 5 \
    --rpi_num_interpolation 5 \
    --rpi_num_samples 1 \
    --max_epochs 200 \
    --beta 0.8 \
    --batch_size 4 \
    --l_rate 1.41e-6 \
    --min_generation 8 \
    --max_generation 48 \
    --project_name rlhf_tldr_rpi_deep
    
echo "Finished seed=${SEED}"
