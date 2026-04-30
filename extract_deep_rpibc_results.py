"""
Extract reward history from wandb offline runs created by the deep RPIBC jobs
and save as properly-named pkl files for cross_dataset_deep_rpibc.py plotter.

Usage (from repo root):
    python3 extract_deep_rpibc_results.py

Expects wandb/ directory to be present (rsync it from the cluster first).
"""

import json
import os
import pathlib
import pickle
import re

try:
    import wandb.sdk.internal.datastore as datastore
    from wandb.proto import wandb_internal_pb2
    USE_PROTO = True
except Exception:
    USE_PROTO = False
    print("[WARN] wandb proto not available, will try JSON logs instead")

BASE_PATH = pathlib.Path(__file__).parent
NUMERICS  = BASE_PATH / "results" / "numerics"
WANDB_DIR = BASE_PATH / "wandb"

# ── Metric keys used in each script ──────────────────────────────────────────
# rlhf_tldr_rpi_deep.py and rlhf_openllama_rpi_deep.py both log env/reward_mean
REWARD_KEYS = ["env/reward_mean", "ppo/returns/mean", "ppo/env/reward_mean"]


def read_jsonl_history(run_dir: pathlib.Path):
    """Try to read reward history from the JSONL log file (simpler, no proto)."""
    # wandb stores per-step history in files/wandb-history.jsonl
    jsonl = run_dir / "files" / "wandb-history.jsonl"
    if not jsonl.exists():
        return []
    rewards = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in REWARD_KEYS:
                if key in row:
                    rewards.append(float(row[key]))
                    break
    return rewards


def read_proto_history(run_dir: pathlib.Path):
    """Read reward history from the binary .wandb protobuf file."""
    wandb_files = list(run_dir.glob("run-*.wandb"))
    if not wandb_files:
        return []
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_files[0]))
    rewards = []
    while True:
        try:
            record = ds.scan_record()
        except Exception:
            break
        if record is None:
            break
        data = record[-1] if isinstance(record, tuple) else record
        try:
            pb = wandb_internal_pb2.Record()
            pb.ParseFromString(data)
        except Exception:
            continue
        if pb.history:
            for item_pb in pb.history.item:
                if item_pb.key in REWARD_KEYS:
                    rewards.append(json.loads(item_pb.value_json))
                    break
    return rewards


def get_run_name(run_dir: pathlib.Path):
    """Read run name from wandb-metadata.json."""
    meta = run_dir / "files" / "wandb-metadata.json"
    if meta.exists():
        with open(meta) as f:
            d = json.load(f)
        # run name is sometimes in 'name' key
        name = d.get("name") or d.get("run_name") or ""
        if name:
            return name
    # Fall back to directory timestamp + id
    return run_dir.name.replace("offline-run-", "")


def classify_run(run_dir: pathlib.Path):
    """Return ('tldr' | 'llama' | None) based on run name / args."""
    meta = run_dir / "files" / "wandb-metadata.json"
    if meta.exists():
        with open(meta) as f:
            d = json.load(f)
        args = d.get("args", [])
        if isinstance(args, dict):
            args = list(args.values())
        # project name or script name gives us the dataset
        program = d.get("program", "") or d.get("codePath", "")
        project = d.get("project", "") or ""
        name    = d.get("name", "") or ""
        combined = " ".join([program, project, name]).lower()
        if "tldr" in combined or "summariz" in combined:
            return "tldr", name
        if "llama" in combined or "openllama" in combined or "hh_rlhf" in combined:
            return "llama", name
    # fall back to dir timestamp ordering
    return None, ""


def main():
    if not WANDB_DIR.exists():
        print(f"[ERROR] {WANDB_DIR} not found. Rsync the wandb/ directory from cluster first.")
        return

    run_dirs = sorted(WANDB_DIR.glob("offline-run-*"))
    print(f"Found {len(run_dirs)} offline wandb run(s) in {WANDB_DIR}\n")

    tldr_runs  = {}   # run_name -> [rewards]
    llama_runs = {}   # run_name -> [rewards]

    for run_dir in run_dirs:
        kind, run_name = classify_run(run_dir)
        if kind is None:
            print(f"  SKIP {run_dir.name} (cannot classify)")
            continue

        # Try JSONL first (fast), fall back to proto
        rewards = read_jsonl_history(run_dir)
        if not rewards and USE_PROTO:
            rewards = read_proto_history(run_dir)

        print(f"  [{kind}] {run_dir.name}  run={run_name!r}  steps={len(rewards)}")

        if kind == "tldr":
            tldr_runs[run_name] = rewards
        else:
            llama_runs[run_name] = rewards

    NUMERICS.mkdir(parents=True, exist_ok=True)

    # Save TL;DR pkl
    if tldr_runs:
        # Keep only runs that have rpibc_deep in the name, or all if none match
        deep_tldr = {k: v for k, v in tldr_runs.items() if "rpibc_deep" in k}
        save_dict = deep_tldr if deep_tldr else tldr_runs
        # Use run name as filename key
        for run_name, rewards in save_dict.items():
            out = NUMERICS / f"TLDr_{run_name}.pkl"
            with open(out, "wb") as f:
                pickle.dump({run_name: rewards}, f)
            print(f"\n  Saved TL;DR pkl → {out}  (len={len(rewards)})")
    else:
        print("\n[WARN] No TL;DR runs found.")

    # Save OpenLLaMA pkl
    if llama_runs:
        deep_llama = {k: v for k, v in llama_runs.items() if "rpibc_deep" in k}
        save_dict = deep_llama if deep_llama else llama_runs
        for run_name, rewards in save_dict.items():
            out = NUMERICS / f"OpenLLaMA_{run_name}.pkl"
            with open(out, "wb") as f:
                pickle.dump({run_name: rewards}, f)
            print(f"  Saved OpenLLaMA pkl → {out}  (len={len(rewards)})")
    else:
        print("[WARN] No OpenLLaMA runs found.")

    print("\nDone! Run the plotter:")
    print("  python3 experiments/plotting/cross_dataset_deep_rpibc.py --save_fig")


if __name__ == "__main__":
    main()
