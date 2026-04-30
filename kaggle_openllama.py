"""
Kaggle Notebook: Deep RPIBC on OpenLLaMA (1 seed)
==================================================
GPU: T4 x2 | Estimated time: 8-12 hours
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
print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

from experiments.scripts.rlhf_openllama_rpi_deep import main

main(
    seed=1,
    method="rpibc_deep",
    max_epochs=200,
    beta=0.8,
    l_rate=3e-5,
    batch_size=1,          # batch_size=1 to fit on T4 16GB
    mini_batch_size=1,
    min_generation=8,
    max_generation=256,
    rpi_time_steps=5,
    rpi_num_interpolation=5,   # reduced from 10 to save time on T4
    rpi_num_samples=1,
    project_name="rlhf-openllama-rpi-deep-kaggle",
    logging_level="WARNING",
)

# ── Save results ────────────────────────────────────────────
# The OpenLLaMA script saves local_res to logs dir; let's also save reward history
import pickle, numpy as np

logs_dir = "/kaggle/working/rpi-bc/experiments/logs/"
for d in os.listdir(logs_dir):
    if "rpibc_deep" in d:
        res_path = os.path.join(logs_dir, d, "local_res.th")
        if os.path.exists(res_path):
            local_res = torch.load(res_path)
            rewards = [s.get("env/reward_mean", 0) for s in local_res]
            out_path = f"/kaggle/working/OpenLLaMA_rpibc_deep_seed1.pkl"
            with open(out_path, "wb") as f:
                pickle.dump({"rpibc_deep_seed1": rewards}, f)
            print(f"Saved: {out_path}")

print("\n✅ OpenLLaMA training complete! Download .pkl from Output tab.")
