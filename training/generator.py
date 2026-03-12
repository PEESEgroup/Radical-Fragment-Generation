import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Config
from datasets.generator import RadicalPairDataset
from models.generator import BDEConditionedGPT
from tqdm.auto import tqdm

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bde = batch["bde"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bde=bde,
            labels=input_ids
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def validate_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Validating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bde = batch["bde"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bde=bde,
            labels=input_ids
        )

        total_loss += outputs.loss.item()

    return total_loss / len(loader)

TOKENIZER_NAME = "seyonec/ChemBERTa-zinc-base-v1"
SAVE_DIR = "generator/bde_conditioned_gpt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# add special tokens if missing
special_tokens = ["<SEP>"]
tokenizer.add_special_tokens({
    "additional_special_tokens": special_tokens
})

bde2_v3 = pd.read_csv("data/bde_rdf_with_multi_halo_cfc_model_3.csv")
train_df = bde2_v3[bde2_v3["set"] == "train"]
val_df   = bde2_v3[bde2_v3["set"] == "valid"]

train_ds = RadicalPairDataset(train_df, tokenizer, max_len=128, augment=True)
val_ds   = RadicalPairDataset(val_df, tokenizer, max_len=128, augment=False)

train_dl = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_dl = DataLoader(
    val_ds,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=128,
    n_embd=256,
    n_layer=8,
    n_head=8,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
model = BDEConditionedGPT(
    config
).to(device)

from transformers import get_linear_schedule_with_warmup

epochs = 10

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dl) * epochs
)

for epoch in range(epochs):
    train_loss = train_epoch(
        model, train_dl, optimizer, scheduler, device
    )

    val_loss = validate_epoch(
        model, val_dl, device
    )

    print(
        f"Epoch {epoch+1:02d} | "
        f"train CE {train_loss:.4f} | "
        f"val CE {val_loss:.4f}"
    )


os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

