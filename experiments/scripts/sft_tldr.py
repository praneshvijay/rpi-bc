import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

try:
    dataset = load_dataset("openai/summarize_from_feedback", "comparisons", split="train", trust_remote_code=True)
except ValueError:
    dataset = load_dataset("openai/summarize_from_feedback", "comparisons", split="train")


lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True,
)
model = get_peft_model(model, lora_config)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["info"])):
        choice_i = example["choice"][i]
        text = f"### Text to Summarize: {example['info'][i]['post']}\n ### Summary: {example['summaries'][i][choice_i]['text']}"
        output_texts.append(text)
    return output_texts


response_template = " ### Summary:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

training_args = TrainingArguments(
    num_train_epochs=1,
    output_dir="phi2-sft-tldr",
    save_strategy="no",
    per_device_train_batch_size=2,
    logging_steps=10,
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
# push to hub
trainer.push_to_hub("phi2-sft-tldr")
