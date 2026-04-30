"""
Kaggle Notebook: Deep RPIBC on TL;DR (1 seed)
==============================================
GPU: T4 x2 | Estimated time: 5-8 hours
"""
import subprocess, sys, os

# ── CRITICAL: Uninstall Kaggle's pre-installed peft first, then install compatible versions ──
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "peft", "trl", "transformers", "accelerate", "diffusers"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
    "transformers==4.38.2", "peft==0.8.2", "trl==0.7.11",
    "accelerate==0.27.2", "bitsandbytes", "datasets==2.15.0",
    "wandb", "python-dotenv"])

# ── Clone repo ──────────────────────────────────────────────
if not os.path.exists("/kaggle/working/rpi-bc"):
    subprocess.check_call(["git", "clone", "https://github.com/praneshvijay/rpi-bc.git",
                           "/kaggle/working/rpi-bc"])

os.chdir("/kaggle/working/rpi-bc")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".", "-q"])

# ── Patch sys.path ──────────────────────────────────────────
sys.path.insert(0, "/kaggle/working/rpi-bc")
os.environ["WANDB_MODE"] = "offline"
os.environ["PYTHONPATH"] = "/kaggle/working/rpi-bc"

# ── Verify GPU ──────────────────────────────────────────────
import torch
assert torch.cuda.is_available(), "CUDA not available! Enable GPU in Kaggle settings."
print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# ── Run training ────────────────────────────────────────────
from experiments.scripts.rlhf_tldr_rpi_deep import main

main(
    seed=1,
    max_epochs=150,
    beta=0.8,
    l_rate=1.41e-6,
    batch_size=1,
    mini_batch_size=1,
    min_generation=8,
    max_generation=48,
    rpi_time_steps=5,
    rpi_num_interpolation=5,
    rpi_num_samples=1,
    project_name="rlhf-tldr-rpi-deep-kaggle",
    logging_level="WARNING",
    reward_model_8bit=True,
)

# ── Copy results ────────────────────────────────────────────
import shutil, glob
for f in glob.glob("/kaggle/working/rpi-bc/results/numerics/TLDr_*.pkl"):
    shutil.copy(f, "/kaggle/working/")
    print(f"Copied: {os.path.basename(f)}")
print("\n✅ TL;DR training complete!")
