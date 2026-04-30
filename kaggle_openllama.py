"""
Kaggle Notebook: Deep RPIBC on OpenLLaMA (1 seed)
==================================================
GPU: T4 x2 | Estimated time: 8-12 hours
"""
import subprocess, sys, os

# ── CRITICAL: Uninstall Kaggle's pre-installed peft first, then install compatible versions ──
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "peft", "trl", "transformers", "accelerate", "diffusers"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
    "transformers==4.38.2", "peft==0.8.2", "trl==0.7.11",
    "accelerate==0.27.2", "bitsandbytes", "datasets>=2.20.0",
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
from experiments.scripts.rlhf_openllama_rpi_deep import main

main(
    seed=1,
    method="rpibc_deep",
    max_epochs=200,
    beta=0.8,
    l_rate=3e-5,
    batch_size=1,
    mini_batch_size=1,
    min_generation=8,
    max_generation=256,
    rpi_time_steps=5,
    rpi_num_interpolation=5,
    rpi_num_samples=1,
    project_name="rlhf-openllama-rpi-deep-kaggle",
    logging_level="WARNING",
    reward_model_half=True,
)

# ── Save results ────────────────────────────────────────────
import pickle
logs_dir = "/kaggle/working/rpi-bc/experiments/logs/"
if os.path.exists(logs_dir):
    for d in os.listdir(logs_dir):
        if "rpibc_deep" in d:
            res_path = os.path.join(logs_dir, d, "local_res.th")
            if os.path.exists(res_path):
                local_res = torch.load(res_path, map_location="cpu")
                rewards = [s.get("env/reward_mean", 0) for s in local_res]
                out_path = f"/kaggle/working/OpenLLaMA_rpibc_deep_seed1.pkl"
                with open(out_path, "wb") as f:
                    pickle.dump({"rpibc_deep_seed1": rewards}, f)
                print(f"Saved: {out_path}")
print("\n✅ OpenLLaMA training complete!")
