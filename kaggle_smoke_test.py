"""
Kaggle Smoke Test for Deep RPIBC (TL;DR + OpenLLaMA)
=====================================================
Run this as a Kaggle notebook with GPU T4 x2 or P100 enabled.

Cell 1: Install dependencies
Cell 2: Paste this entire script and run
"""

# ============================================================
# CELL 1 — Run this cell FIRST to install deps
# ============================================================
# !pip install -q torch transformers trl==0.7.11 peft bitsandbytes accelerate datasets wandb python-dotenv

# ============================================================
# CELL 2 — Clone repo and install
# ============================================================
# !git clone https://github.com/XanderJC/attention-based-credit.git
# %cd attention-based-credit
# !pip install -e . -q

# ============================================================
# CELL 3 — Paste and run this to test TL;DR
# ============================================================
import os
import sys
import torch
import numpy as np

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

print("=" * 60)
print("SMOKE TEST: Deep RPIBC on Kaggle T4")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print("=" * 60)

# ── Test 1: TL;DR Dataset Loading ──────────────────────────
print("\n[1/4] Testing TL;DR dataset loading (trust_remote_code fix)...")
from datasets import load_dataset
try:
    ds = load_dataset("openai/summarize_from_feedback", "comparisons", split="train", trust_remote_code=True)
except (ValueError, TypeError):
    ds = load_dataset("openai/summarize_from_feedback", "comparisons", split="train")
print(f"  ✓ Dataset loaded: {len(ds)} samples")
del ds

# ── Test 2: TL;DR Model Loading (no device_map) ───────────
print("\n[2/4] Testing TL;DR model loading (device_map fix)...")
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
)
lora_config = LoraConfig(
    r=32, lora_alpha=32, lora_dropout=0.0,
    bias="none", task_type="CAUSAL_LM",
)

# This is the line that USED TO crash with device_map={"": 0}
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "XanderJC/gptj-sft-tldr-merged",
    peft_config=lora_config,
    quantization_config=nf4_config,
    # NOTE: no device_map here — that's the fix!
)
print(f"  ✓ TL;DR model loaded on {model.pretrained_model.device}")

# Quick: load reward model + run 1 attribution
print("\n[3/4] Testing TL;DR reward model + Deep RPIBC attribution...")
from peft import AutoPeftModelForSequenceClassification

rank_model = AutoPeftModelForSequenceClassification.from_pretrained(
    "Holarissun/trl_rm_tldr_gptj",
    num_labels=1,
    output_attentions=True,
    return_dict_in_generate=True,
    attn_implementation="eager",
).to("cuda:0")
rank_tokenizer = AutoTokenizer.from_pretrained("Holarissun/trl_rm_tldr_gptj")
rank_tokenizer.pad_token = rank_tokenizer.eos_token
rank_model.config.pad_token_id = rank_model.config.eos_token_id

# Fake a single input
text = "### Text to Summarize: The quick brown fox jumps over the lazy dog.\n ### Summary: A fox jumps over a dog."
inputs = rank_tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True).to("cuda:0")

with torch.no_grad():
    out = rank_model(**inputs)

attn_per_layer = [a.to("cuda:0") for a in out.attentions]
print(f"  ✓ Reward model forward pass OK — {len(attn_per_layer)} attention layers, logits shape {out.logits.shape}")

# Run Deep RPIBC attribution
sys.path.insert(0, os.getcwd())
from abcrl.attention.redistribution import get_rpi_deep_attention_distribution_batch_gptj

fake_response = torch.tensor([1, 2, 3, 4, 5])  # 5 tokens
fake_query = torch.tensor([10, 20, 30])  # 3 tokens

dists = get_rpi_deep_attention_distribution_batch_gptj(
    model=rank_model,
    inputs={k: v.to("cuda:0") for k, v in inputs.items()},
    responses=[fake_response],
    queries=[fake_query],
    attention_probs_per_layer=attn_per_layer,
    num_time_steps=2,
    num_interpolation=2,
    num_samples=1,
)
print(f"  ✓ Deep RPIBC attribution OK — distribution shape: {dists[0].shape}, sum: {dists[0].sum().item():.4f}")

# Cleanup TL;DR models
del model, rank_model, attn_per_layer, out
torch.cuda.empty_cache()
import gc; gc.collect()

# ── Test 3: OpenLLaMA Model Loading ────────────────────────
print("\n[4/4] Testing OpenLLaMA model loading (device_map fix)...")
from transformers import AutoModelForSequenceClassification, LlamaTokenizer

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "VMware/open-llama-7b-open-instruct",
    peft_config=lora_config,
    quantization_config=nf4_config,
    # NOTE: no device_map here — that's the fix!
)
print(f"  ✓ OpenLLaMA model loaded on {model.pretrained_model.device}")

# Load OpenLLaMA reward model
rank_model_llama = AutoModelForSequenceClassification.from_pretrained(
    "weqweasdas/hh_rlhf_rm_open_llama_3b",
    output_attentions=True,
    return_dict_in_generate=True,
    attn_implementation="eager",
    device_map="cuda:0",
)
rank_tokenizer_llama = LlamaTokenizer.from_pretrained(
    "weqweasdas/hh_rlhf_rm_open_llama_3b", use_fast=False
)

text = "Human: What is the meaning of life?\n\nAssistant: The meaning of life is subjective."
inputs = rank_tokenizer_llama(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True).to("cuda:0")

with torch.no_grad():
    out = rank_model_llama(**inputs)

print(f"  ✓ OpenLLaMA reward model OK — logits: {out.logits.shape}, attentions: {len(out.attentions)} layers")

# Run Deep RPIBC LLaMA attribution
from abcrl.attention.redistribution import get_rpi_deep_attention_distribution_batch_llama

fake_response = torch.tensor([1, 2, 3, 4, 5])
fake_query = torch.tensor([10, 20, 30])

dists = get_rpi_deep_attention_distribution_batch_llama(
    model=rank_model_llama,
    inputs={k: v.to("cuda:0") for k, v in inputs.items()},
    responses=[fake_response],
    queries=[fake_query],
    num_time_steps=2,
    num_interpolation=2,
    num_samples=1,
)
print(f"  ✓ Deep RPIBC LLaMA attribution OK — distribution shape: {dists[0].shape}, sum: {dists[0].sum().item():.4f}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("\nBoth scripts are safe to push to the cluster.")
