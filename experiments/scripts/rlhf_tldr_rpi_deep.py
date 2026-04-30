"""
Deep Multi-Layer RPIBC on TL;DR summarization.
Generator: GPT-J 6B (LoRA + 4-bit, XanderJC/gptj-sft-tldr-merged)
Reward Model: Holarissun/trl_rm_tldr_gptj (GPT-J fine-tuned)
Best config from IMDb sweep: T=5, K=10, L=all
Run 10 seeds on cluster to beat ABC baseline.
"""
import argparse
import datetime
import logging
import os
import pickle
import time

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import AutoPeftModelForSequenceClassification, LoraConfig
from pkg_resources import resource_filename
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import AutoModelForCausalLMWithValueHead, PPOConfig
from trl.core import LengthSampler

import wandb
from abcrl.attention.redistribution import get_rpi_deep_attention_distribution_batch_gptj
from abcrl.rl.ppo import PPOTrainerABC




def build_tldr_dataset(config, dataset_name="openai/summarize_from_feedback"):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    try:
        ds = load_dataset(dataset_name, "comparisons", split="train", trust_remote_code=True)
    except ValueError:
        ds = load_dataset(dataset_name, "comparisons", split="train")

    def tokenize(sample):
        _, post, _, _ = (
            sample["choice"],
            sample["info"]["post"],
            sample["summaries"][0]["text"],
            sample["summaries"][1]["text"],
        )
        query = f"### Text to Summarize: {post}\n ### Summary: "
        sample["input_ids"] = tokenizer.encode(query)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds = ds.filter(lambda x: len(x["input_ids"]) < 450, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def main(
    max_epochs: int = 150,
    beta: float = 0.8,
    l_rate: float = 1.41e-6,
    min_generation: int = 8,
    max_generation: int = 48,
    batch_size: int = 4,
    mini_batch_size: int = 1,
    seed: int = 1,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    rpi_time_steps: int = 5,
    rpi_num_interpolation: int = 10,
    rpi_num_samples: int = 1,
    project_name: str = "rlhf-tldr-rpi-deep",
    logging_level: str = "WARNING",
    reward_model_8bit: bool = False,
):
    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M")
    run_name = (
        f"rpibc_deep_T{rpi_time_steps}_K{rpi_num_interpolation}"
        f"_S{rpi_num_samples}_Lall"
        f"_{int(beta*100)}_{min_generation}_{max_generation}_{date_time}"
    )
    print(f"Run name: {run_name}")
    print(f"Setting seed to {seed}")
    print(f"Deep RPI config: T={rpi_time_steps}, K={rpi_num_interpolation}, "
          f"S={rpi_num_samples}, layers=all")

    BASE_PATH = resource_filename("abcrl", "/..")
    LOG_DIRECTORY = f"{BASE_PATH}/experiments/logs/{run_name}"
    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)

    if seed is not None:
        set_seed(seed)

    logger = logging.getLogger(__name__)
    level = logging.getLevelName(logging_level)
    logger.setLevel(level)

    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(LOG_DIRECTORY + "/debug.log")
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(level)

    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    config = PPOConfig(
        model_name="XanderJC/gptj-sft-tldr-merged",
        learning_rate=l_rate,
        log_with="wandb",
        ppo_epochs=4,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        optimize_device_cache=True,
        seed=seed,
    )

    dataset = build_tldr_dataset(config)
    dataset = dataset.shuffle(seed=seed)

    load_dotenv()
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb.init(project=project_name, name=run_name, entity=wandb_entity)

    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type="CAUSAL_LM",
    )
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        peft_config=lora_config,
        quantization_config=nf4_config,
    )
    if not reward_model_8bit:
        # On cluster (plenty of VRAM) keep explicit ref model
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name,
            quantization_config=nf4_config,
        )
    else:
        # On Kaggle T4: skip ref model — PEFT auto-creates one by disabling adapter
        ref_model = None

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Reward model — GPT-J fine-tuned for summarization quality
    rm_kwargs = dict(
        num_labels=1,
        output_attentions=True,
        return_dict_in_generate=True,
        attn_implementation="eager",
    )
    if reward_model_8bit:
        # Load in fp16 to halve VRAM (12GB vs 24GB fp32)
        rm_kwargs["torch_dtype"] = torch.float16
        rm_kwargs["device_map"] = "auto"
        rank_model = AutoPeftModelForSequenceClassification.from_pretrained(
            "Holarissun/trl_rm_tldr_gptj", **rm_kwargs
        )
    else:
        rank_model = AutoPeftModelForSequenceClassification.from_pretrained(
            "Holarissun/trl_rm_tldr_gptj", **rm_kwargs
        ).to("cuda:0")
    rank_model.eval()
    rank_model.requires_grad_(False)
    rank_tokenizer = AutoTokenizer.from_pretrained("Holarissun/trl_rm_tldr_gptj")
    rank_tokenizer.pad_token = rank_tokenizer.eos_token
    rank_model.config.pad_token_id = rank_model.config.eos_token_id

    ppo_trainer = PPOTrainerABC(
        config=config, model=model, ref_model=ref_model,
        tokenizer=tokenizer, dataset=dataset, data_collator=collator,
    )

    generation_kwargs = {
        "min_length": -1, "top_k": 0.0, "top_p": 1.0,
        "do_sample": True, "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True, "batch_size": batch_size,
    }

    local_res = []

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # ── Generation ──────────────────────────────────────────────
        response_tensors = []
        for query in query_tensors:
            generation_kwargs["max_new_tokens"] = max_generation
            generation_kwargs["min_new_tokens"] = min_generation
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response[0].squeeze()[len(query):])

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # ── Reward Model Forward ─────────────────────────────────────
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        inputs = rank_tokenizer(
            texts, return_tensors="pt", max_length=512,
            padding="max_length", truncation=True,
        ).to("cuda:0")

        with torch.no_grad():
            out = rank_model(**inputs)

        # Collect all-layer attentions for Deep RPIBC
        # attentions: tuple of [B, H, S, S], one per transformer block
        attn_per_layer = [a.to("cuda:0") for a in out.attentions]

        # ── Deep RPIBC Attribution ───────────────────────────────────
        rm_inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        reward_distributions = get_rpi_deep_attention_distribution_batch_gptj(
            model=rank_model,
            inputs=rm_inputs,
            responses=response_tensors,
            queries=query_tensors,
            attention_probs_per_layer=attn_per_layer,
            num_time_steps=rpi_time_steps,
            num_interpolation=rpi_num_interpolation,
            num_samples=rpi_num_samples,
        )

        # ── Compute Rewards ──────────────────────────────────────────
        rewards = []
        for logit, response, dist in zip(out.logits, response_tensors, reward_distributions):
            total = logit.squeeze().detach()
            reward = torch.zeros_like(response, dtype=float)
            reward[-1] = (1 - beta) * total
            redist = dist.to(reward.device) * total * beta
            reward += redist
            rewards.append(reward)

        # ── PPO Step ─────────────────────────────────────────────────
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        og_rewards = [score.cpu().sum() for score in rewards]
        ppo_trainer.log_stats(stats, batch, og_rewards)

        stats["env/reward_mean"] = np.mean(og_rewards)
        stats["env/reward_std"] = np.std(og_rewards)
        for key in ["objective/logprobs", "objective/ref_logprobs",
                    "ppo/policy/advantages", "ppo/policy/ratio"]:
            stats.pop(key, None)
        local_res.append(stats)

        if epoch >= max_epochs:
            break

    # ── Save Results ─────────────────────────────────────────────────
    results_dir = os.path.join(BASE_PATH, "results", "numerics")
    os.makedirs(results_dir, exist_ok=True)

    rewards_series = [float(s["env/reward_mean"]) for s in local_res]
    out_path = os.path.join(results_dir, f"TLDr_{run_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({run_name: rewards_series}, f)

    print(f"Saved results to {out_path}")
    print(f"Final mean reward: {rewards_series[-1]:.4f}")
    print(f"Finished seed={seed}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--l_rate", type=float, default=1.41e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--min_generation", type=int, default=8)
    parser.add_argument("--max_generation", type=int, default=48)
    parser.add_argument("--rpi_time_steps", type=int, default=5)
    parser.add_argument("--rpi_num_interpolation", type=int, default=10)
    parser.add_argument("--rpi_num_samples", type=int, default=1)
    parser.add_argument("--logging_level", type=str, default="WARNING")
    parser.add_argument("--project_name", type=str, default="rlhf-tldr-rpi-deep")
    args = parser.parse_args()
    main(**args.__dict__)
