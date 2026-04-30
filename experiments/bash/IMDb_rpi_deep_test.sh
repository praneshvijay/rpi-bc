#!/bin/bash
#SBATCH --job-name=rpi_deep_test
#SBATCH --output=/home/bt2/22CS10013/attention-based-credit/rpi_deep_test_out_%j.txt
#SBATCH --error=/home/bt2/22CS10013/attention-based-credit/rpi_deep_test_err_%j.txt
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_l40

REPO=/home/bt2/22CS10013/attention-based-credit

echo "=== DEEP RPIBC SMOKE TEST on $(hostname) ==="

source $REPO/venv/bin/activate
export PYTHONPATH=$REPO
export WANDB_MODE=offline
export RPI_RUN_SUFFIX=_test   # pkl filename will be rpibc_deep_..._test → skipped by plotter

python3 $REPO/experiments/scripts/rlhf_imdb_rpi_deep.py \
    --seed 1 \
    --rpi_time_steps 5 \
    --rpi_num_interpolation 10 \
    --rpi_num_samples 1 \
    --max_epochs 5 \
    --beta 0.8 \
    --batch_size 16 \
    --project_name rlhf_imdb_rpi_deep_TEST

echo "=== SMOKE TEST DONE ==="
