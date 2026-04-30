"""
Kaggle Notebook: Deep RPIBC on TL;DR (1 seed)
==============================================
GPU: T4 x2 | Estimated time: 5-8 hours
Saves results to /kaggle/working/ for download
"""
import subprocess, sys, os

# ── Install deps ────────────────────────────────────────────
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.38.2", "trl==0.7.11", "peft", "bitsandbytes",
    "accelerate", "datasets", "wandb", "python-dotenv"])

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

# ── Run training ────────────────────────────────────────────
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB")

# Import and run main directly to avoid subprocess issues
from experiments.scripts.rlhf_tldr_rpi_deep import main

main(
    seed=1,
    max_epochs=150,
    beta=0.8,
    l_rate=1.41e-6,
    batch_size=1,          # batch_size=1 to fit on T4 16GB
    min_generation=8,
    max_generation=48,
    rpi_time_steps=5,
    rpi_num_interpolation=5,   # reduced from 10 to save time on T4
    rpi_num_samples=1,
    project_name="rlhf-tldr-rpi-deep-kaggle",
    logging_level="WARNING",
)

# ── Copy results to /kaggle/working for download ───────────
import shutil, glob
for f in glob.glob("/kaggle/working/rpi-bc/results/numerics/TLDr_*.pkl"):
    shutil.copy(f, "/kaggle/working/")
    print(f"Copied: {os.path.basename(f)}")

print("\n✅ TL;DR training complete! Download .pkl from Output tab.")
