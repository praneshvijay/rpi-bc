#!/bin/bash
#SBATCH --job-name=llama_deep
#SBATCH --output=/home/bt2/22CS10013/attention-based-credit/llama_rpi_deep_out_%A_%a.txt
#SBATCH --error=/home/bt2/22CS10013/attention-based-credit/llama_rpi_deep_err_%A_%a.txt
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_h100
#SBATCH --array=0-4   # 5 seeds — resubmit if 12h isn't enough

set -e


REPO=/home/bt2/22CS10013/attention-based-credit
SEED=$((SLURM_ARRAY_TASK_ID + 1))

echo "Running Deep RPIBC OpenLLaMA seed=${SEED} on node=$(hostname)"

source $REPO/venv/bin/activate
export PYTHONPATH=$REPO
export WANDB_MODE=offline

python3 $REPO/experiments/scripts/rlhf_openllama_rpi_deep.py \
    --seed ${SEED} \
    --rpi_time_steps 5 \
    --rpi_num_interpolation 5 \
    --rpi_num_samples 1 \
    --max_epochs 200 \
    --beta 0.8 \
    --batch_size 2 \
    --mini_batch_size 1 \
    --l_rate 3e-5 \
    --min_generation 8 \
    --max_generation 256 \
    --method rpibc_deep \
    --project_name rlhf_openllama_rpi_deep
    
echo "Finished seed=${SEED}"
