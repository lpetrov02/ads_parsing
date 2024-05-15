import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import factorial

import json
import re
from tqdm.notebook import tqdm, trange
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from peft import LoraConfig, get_peft_model

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq

import sys
sys.path.append('..')
from eval_utils import get_parser, get_metrics_computer, json_to_string


# torch Dataset block
class AdsDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, bundles_to_string_processor, max_length=256, shuffle_bundles=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.processor = bundles_to_string_processor
        self.max_length = max_length
        self.shuffle = shuffle_bundles

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = "<LM>" + self.data.loc[self.data.index[idx], "Text"]
        inputs = re.sub("\n", " <NL> ", inputs)
        bundles = json.loads(self.data.loc[self.data.index[idx], "bundles"])
        if self.shuffle:
            np.random.shuffle(bundles)
        target = self.processor(json.dumps(bundles))
        return tokenizer(inputs, text_target=target, max_length=self.max_length, truncation=True)
    

def train_fn(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()

        # with torch.cuda.amp.autocast():
        outputs = model(**batch)
        del batch
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        torch.cuda.empty_cache()

    return total_loss / len(dataloader)


def eval_fn(model, dataloader, eval_metrics, device):
    model.eval()
    all_metrics = defaultdict(float)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.cuda.amp.autocast():
                outputs = model.generate(input_ids=batch["input_ids"].to(device), max_length=256).cpu()
            metrics = eval_metrics((outputs, batch["labels"]), show=int(i == 0) * 3)
            for key in metrics:
                all_metrics[key] += metrics[key]

    for key in all_metrics:
        all_metrics[key] /= len(dataloader)
    return all_metrics


def get_trainable_params(model):
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    return trainable_params


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Working on device {device}")

print("\nDataset...")
data_ads = pd.read_csv("/home/vlad/15k_for_train.csv")
data_ads = data_ads[data_ads["n_bundles"] <= 5][:100]
data_ads.set_index(np.arange(len(data_ads)), inplace=True)
print(f"Data of shape {data_ads.shape}")


model_checkpoint = "ai-forever/FRED-T5-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
tokenizer.add_tokens(["€", "<BOB>", "<EOB>", "<BOT>", "<EOT>", "<BOP>", "<EOP>", "<BOC1>", "<EOC1>", "<BOC2>", "<EOC2>", "<NL>"], special_tokens=False)
model.resize_token_embeddings(len(tokenizer))
count_parameters(model)
print(f"Tokenizer has {len(tokenizer)} tokens")

dataset = AdsDataset(data_ads, tokenizer, json_to_string)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

train_dataset, eval_dataset = random_split(dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)

metrics_computer = get_metrics_computer(tokenizer)


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v", "lm_head"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

peft_model = get_peft_model(model, lora_config)
# peft_model.half()
peft_model.to(device)
count_parameters(peft_model)

n_epochs = 20
optimizer = Adam(get_trainable_params(peft_model), lr=5e-5, weight_decay=1e-3)
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=3)
main_scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)
scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[3])

train_losses = []
metrics = defaultdict(lambda: list())
for epoch in trange(n_epochs):
    print(f"Epoch #{epoch + 1}")
    train_loss = train_fn(peft_model, train_dataloader, optimizer, device)
    train_losses.append(train_loss)
    epoch_metrics = eval_fn(peft_model, eval_dataloader, metrics_computer, device)
    print(f"\tTrain loss: {train_loss}")
    for key in epoch_metrics:
        metrics[key].append(epoch_metrics[key])
        print(f"\t{key}: {epoch_metrics[key]}")
    scheduler.step()
    state = {
        "epoch": epoch,
        "model_state": peft_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(state, 'fredT5/base_checkpoint.pth')

lora_model = peft_model.to('cpu').merge_and_unload()
output_dir = "../good_checkpoints/fred_noLoRa_train20ep"
lora_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
