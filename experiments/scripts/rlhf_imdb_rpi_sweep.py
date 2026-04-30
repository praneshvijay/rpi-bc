"""
RPIBC Hyperparameter Sweep on IMDb.
This is a standalone script that mirrors rlhf_imdb.py but adds RPI-specific
hyperparameter arguments (rpi_time_steps, rpi_num_interpolation, rpi_num_samples).
The method is always 'rpibc'. Run names encode the config: rpibc_T{T}_K{K}_S{S}.
"""
import argparse
import datetime
import os
import pickle

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          set_seed)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig
from trl.core import LengthSampler

import wandb
from abcrl.attention.redistribution import get_rpi_attention_distribution_batch
from abcrl.rl.ppo import PPOTrainerABC


def build_dataset(
    config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8
):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def main(
    max_epochs: int = 150,
    beta: float = 0.8,
    l_rate: float = 1.41e-5,
    min_generation: int = 8,
    max_generation: int = 16,
    project_name: str = "rlhf_imdb_rpi_sweep",
    batch_size: int = 16,
    seed: int = 1,
    # RPI hyperparameters
    rpi_time_steps: int = 10,
    rpi_num_interpolation: int = 5,
    rpi_num_samples: int = 1,
):
    method = "rpibc"

    if seed is not None:
        print(f"Setting seed to {seed}")
        set_seed(seed)

    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M")

    # Encode hyperparams in run name for easy identification
    config_tag = f"T{rpi_time_steps}_K{rpi_num_interpolation}_S{rpi_num_samples}"
    run_name = f"{method}_{config_tag}_{int(beta*100)}_{min_generation}_{max_generation}_{date_time}"

    print(f"Run name: {run_name}")
    print(f"RPI config: time_steps={rpi_time_steps}, num_interpolation={rpi_num_interpolation}, num_samples={rpi_num_samples}")

    load_dotenv()
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb.init(**{"project": project_name, "name": run_name, "entity": f"{wandb_entity}"})

    config = PPOConfig(
        model_name="lvwerra/gpt2-imdb",
        learning_rate=l_rate,
        log_with="wandb",
        ppo_epochs=4,
        batch_size=batch_size,
    )

    dataset = build_dataset(config)
    dataset = dataset.shuffle()

    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    reward_name = "XanderJC/gpt2-rm-imdb"
    rank_model = AutoModelForSequenceClassification.from_pretrained(
        reward_name, output_attentions=True
    )
    rank_tokenizer = AutoTokenizer.from_pretrained(reward_name)
    rank_model.config.pad_token_id = rank_model.config.eos_token_id

    ppo_trainer = PPOTrainerABC(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    output_length_sampler = LengthSampler(min_generation, max_generation)

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_attentions": True,
    }

    all_rewards = []

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        response_tensors = []
        response_attentions = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            generation_kwargs["min_new_tokens"] = min_generation
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response[0].squeeze()[-gen_len:])
            response_attentions.append(response.attentions)

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        inputs = rank_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        out = rank_model(**inputs)

        rpibc_distributions = get_rpi_attention_distribution_batch(
            rank_model,
            inputs,
            response_tensors,
            query_tensors,
            out.attentions[-1],
            num_time_steps=rpi_time_steps,
            num_interpolation=rpi_num_interpolation,
            num_samples=rpi_num_samples,
        )
        attention = out.attentions[-1]

        rewards = []
        for i, (logit, response, query, att) in enumerate(zip(
            out.logits, response_tensors, query_tensors, attention
        )):
            total = (logit[1] - logit[0]).detach()
            reward = torch.zeros_like(response, dtype=float)
            reward[-1] = (1 - beta) * total
            redist_reward = (
                rpibc_distributions[i].to(reward.device)
                * total
                * beta
            )
            reward += redist_reward
            rewards.append(reward)

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        og_rewards = [score.sum() for score in rewards]
        ppo_trainer.log_stats(stats, batch, og_rewards)

        if epoch >= max_epochs:
            break

        all_rewards.append(torch.tensor(og_rewards).mean().item())

    # Save results
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/numerics"))
    results_path = os.path.join(results_dir, f"IMDb_{run_name}.pkl")
    try:
        os.makedirs(results_dir, exist_ok=True)
        with open(results_path, "wb") as f:
            pickle.dump({run_name: all_rewards}, f)
        print(f"Saved results to {results_path}")
    except Exception as e:
        print(f"Failed to save results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--l_rate", type=float, default=1.41e-5)
    parser.add_argument("--min_generation", type=int, default=8)
    parser.add_argument("--max_generation", type=int, default=16)
    parser.add_argument("--project_name", type=str, default="rlhf_imdb_rpi_sweep")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    # RPI-specific hyperparameters
    parser.add_argument("--rpi_time_steps", type=int, default=10,
                        help="Diffusion time steps T (noise level of baseline)")
    parser.add_argument("--rpi_num_interpolation", type=int, default=5,
                        help="Number of interpolation steps K (accuracy of attribution)")
    parser.add_argument("--rpi_num_samples", type=int, default=1,
                        help="Number of noise samples S (variance reduction)")

    args = parser.parse_args()
    main(**args.__dict__)
