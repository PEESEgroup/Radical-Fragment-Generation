from rdkit import Chem
import torch
from torch_geometric.data import Data, Dataset

from rdkit import RDLogger

# Disable all logs from RDKit
RDLogger.DisableLog('rdApp.*')

def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetTotalNumHs(includeNeighbors=True),
    ]

def bond_features(bond):
    bt = bond.GetBondType()
    bond_type = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }.get(bt, 0)
    return [
        bond_type,
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float
    )

    edge_index, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_index += [[i, j], [j, i]]
        edge_attr  += [bf, bf]

    if len(edge_attr) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class BDEFragmentDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.cache = {}

    def _get_graph(self, smiles):
        if smiles not in self.cache:
            g = smiles_to_graph(smiles)
            if g is None:
                raise ValueError(smiles)
            self.cache[smiles] = g
        return self.cache[smiles].clone()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        parent = self._get_graph(row["molecule"])
        frag1  = self._get_graph(row["fragment1"])
        frag2  = self._get_graph(row["fragment2"])

        if "bde" in row:
          y = torch.tensor(row["bde"], dtype=torch.float)
          return parent, frag1, frag2, y
        else:
          return parent, frag1, frag2, -100

