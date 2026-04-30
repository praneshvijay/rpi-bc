#!/bin/bash
#SBATCH --job-name=rpi_sweep
#SBATCH --output=/home/bt2/22CS10013/attention-based-credit/rpi_sweep_out_%A_%a.txt
#SBATCH --error=/home/bt2/22CS10013/attention-based-credit/rpi_sweep_err_%A_%a.txt
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --array=0-8    # 9 hyperparameter configs (3 T values x 3 K values)

REPO=/home/bt2/22CS10013/attention-based-credit
SCRIPT=$REPO/experiments/scripts/rlhf_imdb_rpi_sweep.py

source $REPO/venv/bin/activate
export WANDB_MODE=offline
export PYTHONPATH=$REPO

# Define the sweep configs as parallel arrays
# Format: T (time_steps) K (num_interpolation) S (num_samples)
TIME_STEPS=(5  5  5  10 10 10 20 20 20)
NUM_INTERP=(3  5  10  3  5  10  3  5  10)
NUM_SAMP=(  1  1   1  1  1   1  1  1   1)

T=${TIME_STEPS[$SLURM_ARRAY_TASK_ID]}
K=${NUM_INTERP[$SLURM_ARRAY_TASK_ID]}
S=${NUM_SAMP[$SLURM_ARRAY_TASK_ID]}

echo "Starting config T=$T K=$K S=$S on node=$(hostname)"

# Run 3 seeds per config for statistical reliability
for SEED in 1 2 3; do
    echo "  Seed=$SEED"
    python3 $SCRIPT \
        --rpi_time_steps $T \
        --rpi_num_interpolation $K \
        --rpi_num_samples $S \
        --beta 0.8 \
        --batch_size 16 \
        --max_epochs 150 \
        --seed $SEED
done

echo "Finished config T=$T K=$K S=$S"
