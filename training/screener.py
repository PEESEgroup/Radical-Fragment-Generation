import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from datasets.screener import BDEFragmentDataset
from models.screener import BDEFragmentModel
from rdkit import RDLogger
# Disable all logs from RDKit
RDLogger.DisableLog('rdApp.*')

SAVE_DIR = "pretrained/screener/bde_pretrained.pt"

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total, n = 0.0, 0

    for parent, f1, f2, y in tqdm(loader):
        parent, f1, f2, y = (
            parent.to(device),
            f1.to(device),
            f2.to(device),
            y.to(device),
        )

        pred = model(parent, f1, f2)
        loss = F.l1_loss(pred, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total += loss.item()
        n += 1

    return total / n

bde2_v3 = pd.read_csv("data/bde_rdf_with_multi_halo_cfc_model_3.csv")
train_df = bde2_v3[bde2_v3["set"] == "train"]
val_df   = bde2_v3[bde2_v3["set"] == "valid"]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BDEFragmentModel().to(device)

# load pretrained weights if available
# model.load_state_dict(torch.load("pretrained/screener/bde_pretrained.pt"))

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

train_ds = BDEFragmentDataset(train_df)
val_ds   = BDEFragmentDataset(val_df)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=128)

for epoch in range(1, 50):
    tr = run_epoch(train_loader, True)
    va = run_epoch(val_loader, False)
    scheduler.step(va)

    print(f"{epoch:03d} | Train MAE {tr:.3f} | Val MAE {va:.3f}")

torch.save(model.state_dict(), SAVE_DIR)