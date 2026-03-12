import torch
import random
from torch.utils.data import Dataset
from rdkit import Chem


class RadicalPairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, augment=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def get_random_smiles(self, smi):
            """Generates a non-canonical, randomized SMILES string using correct RDKit kwargs."""
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return smi

            # RDKit uses doRandom (camelCase)
            return Chem.MolToSmiles(
                mol,
                doRandom=True,
                canonical=False,
                isomericSmiles=True
            )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        f1_raw = row.fragment1
        f2_raw = row.fragment2

        if self.augment:
            # 1. SMILES Randomization
            f1 = self.get_random_smiles(f1_raw)
            f2 = self.get_random_smiles(f2_raw)

            # 2. Randomize fragment order (Order Permutation)
            if random.random() > 0.5:
                f1, f2 = f2, f1
        else:
            # Use fixed order and canonical forms for validation/test
            # Using your existing sorting logic
            frags = sorted([f1_raw, f2_raw], key=lambda x: (len(x), x))
            f1, f2 = frags[0], frags[1]

        seq = f"{f1} <SEP> {f2}"

        enc = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "bde": torch.tensor(row.bde, dtype=torch.float)
        }