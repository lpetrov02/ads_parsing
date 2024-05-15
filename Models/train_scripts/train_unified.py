import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import factorial

import json
import re

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

from datasets import Dataset
from transformers import AutoModelForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model

import sys
from eval_utils import get_metrics_computer, PrintCallback
from utils import get_preprocessor, Format, ShuffleCollator, count_parameters


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

with open("configs/config.json", "rb") as config:
    params = json.load(config)

out_format = Format.SpecTokens if params["format"] == "SpecTokens" else Format.JustJson

print("\nData...")
train_data = pd.read_csv("~/work/resources/data/ads_train.csv")
train_data = train_data[train_data["n_bundles"] <= params.get("max_bundles", np.inf)]
train_data.set_index(np.arange(len(train_data)), inplace=True)
print(f"We have train datset of size {len(train_data)}")

eval_data = pd.read_csv("~/work/resources/data/ads_eval.csv")
eval_data = eval_data[:params.get("max_eval_size", len(eval_data))]
eval_data.set_index(np.arange(len(eval_data)), inplace=True)
print(f"We have eval datset of size {len(eval_data)}")

print("\nModel...")
model_checkpoint = params["model"]
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

tokenizer.add_tokens(["₾", "$", "€"], special_tokens=False)
if params.get("add_nl_token", False):
    tokenizer.add_tokens(["<NL>"], special_tokens=False)
if out_format == Format.SpecTokens:
    tokenizer.add_tokens(["<BOB>", "<EOB>", "<BOT>", "<EOT>", "<BOP>", "<EOP>", "<BOC1>", "<EOC1>", "<BOC2>", "<EOC2>"], special_tokens=False)
elif out_format == Format.JustJson:
    tokenizer.add_tokens(["{", "}"], special_tokens=False)
model.resize_token_embeddings(len(tokenizer))

print("\nDataset...")
train_dataset = Dataset.from_pandas(train_data[["Text", "bundles"]])
train_ads = train_dataset.map(
    get_preprocessor(tokenizer, out_format, params.get("add_eos_token", False)),
    batched=True,
    num_proc=4,
    remove_columns=train_dataset.column_names
)
train_ads = train_ads.flatten()

eval_dataset = Dataset.from_pandas(eval_data[["Text", "bundles"]])
eval_ads = eval_dataset.map(
    get_preprocessor(tokenizer, out_format),
    batched=True,
    num_proc=4,
    remove_columns=eval_dataset.column_names
)
eval_ads = eval_ads.flatten()

if params.get("change_pad_to_eos", False):
    tokenizer.pad_token = tokenizer.eos_token
data_collator = ShuffleCollator(tokenizer, out_format, params.get("add_eos_token", False), params.get("shuffle_bundles", False))

print("\nTrainer & Training...")
train_params = params["train"]
eval_params = params["eval"]

lora_rank = params.get("lora", {}).get("rank")
if lora_rank is not None:
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=32,
        target_modules=["k", "q", "v", "o", "lm_head"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, lora_config)

count_parameters(model)

training_args = Seq2SeqTrainingArguments(
    output_dir="tmp_checkpoints",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=train_params["lr"],
    per_device_train_batch_size=train_params["batch_size"],
    per_device_eval_batch_size=eval_params["batch_size"],
    weight_decay=train_params.get("weight_decay", 0),
    save_total_limit=1,
    num_train_epochs=train_params["n_epochs"],
    predict_with_generate=True,
    generation_max_length=256,
    lr_scheduler_type=train_params.get("scheduler", "cosine"),
    group_by_length=False,
    warmup_steps=train_params.get("warmup_steps", 0),
    fp16=train_params.get("fp16", False),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ads,
    eval_dataset=eval_ads,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=get_metrics_computer(tokenizer, out_format),
    callbacks=[PrintCallback(out_format, show=eval_params.get("show", 3), device=device)]
)

print("\nStart of training...")
trainer.train()

if lora_rank is not None:
    model = model.merge_and_unload()
output_dir = f"../good_checkpoints/{params["save_folder"]}"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)