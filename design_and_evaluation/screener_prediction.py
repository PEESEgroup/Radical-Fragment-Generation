import pandas as pd
import torch
from datasets.screener import BDEFragmentDataset
from torch_geometric.loader import DataLoader
from training.screener import BDEFragmentModel
import torch.nn.functional as F
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BDEFragmentModel().to(device)
model.load_state_dict(torch.load("pretrained/screener/bde_pretrained.pt"))

# Need to have run "design_and_evaluation/generate.py" first to create the "generated_radical_pairs.csv" file.
test_df = pd.read_csv("SAVED_GENERATED_PAIRS/generated_radical_pairs.csv")

def run_gen_epoch(loader, collect=True):
    model.eval()

    total, n = 0.0, 0
    all_preds, all_targets = [], []

    for parent, f1, f2, y in tqdm(loader):
        parent, f1, f2, y = (
            parent.to(device),
            f1.to(device),
            f2.to(device),
            y.to(device),
        )
        pred = model(parent, f1, f2)
        loss = F.l1_loss(pred, y)

        total += loss.item()
        n += 1

        if collect:
            all_preds.append(pred.detach().cpu())
            all_targets.append(y.detach().cpu())

    if collect:
        return (
            total / n,
            torch.cat(all_preds).numpy(),
            torch.cat(all_targets).numpy(),
        )

    return total / n


test_ds = BDEFragmentDataset(test_df)
test_loader = DataLoader(test_ds, batch_size=256)

test_loss, preds, targets = run_gen_epoch(
    test_loader,
    collect=True
)

test_df['pred_bde'] = preds

test_df["_pair_key"] = test_df.apply(
    lambda r: tuple(sorted((r["fragment1"], r["fragment2"]))),
    axis=1
)

test_df = test_df.drop_duplicates(subset="_pair_key").drop(columns="_pair_key").reset_index(drop=True)
test_df.to_csv('SAVED_GENERATED_PAIRS/generated_radical_pairs_with_screener_predictions.csv', index=False)
