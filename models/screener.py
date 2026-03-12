import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool


class BondAtomLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.bond_mlp = nn.Sequential(
            nn.Linear(3*h, h),
            nn.ReLU(),
            nn.Linear(h, h),
        )
        self.atom_mlp = nn.Sequential(
            nn.Linear(2*h, h),
            nn.ReLU(),
            nn.Linear(h, h),
        )

    def forward(self, x, edge_index, e):
        src, dst = edge_index
        e = self.bond_mlp(torch.cat([x[src], x[dst], e], dim=-1))
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, e)
        x = self.atom_mlp(torch.cat([x, agg], dim=-1))
        return x, e
    
class MPNNEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, h=128, layers=6):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, h)
        self.edge_emb = nn.Linear(edge_dim, h)
        self.layers = nn.ModuleList([BondAtomLayer(h) for _ in range(layers)])
        self.readout = nn.Linear(h, 1)

    def forward(self, data, return_embedding=False):
        x = self.node_emb(data.x)
        e = self.edge_emb(data.edge_attr)

        for layer in self.layers:
            x, e = layer(x, data.edge_index, e)

        graph_emb = global_add_pool(x, data.batch)  # [batch, h]

        atom_energy = self.readout(x)
        mol_energy  = global_add_pool(atom_energy, data.batch)

        if return_embedding:
            return mol_energy.squeeze(-1), graph_emb
        else:
            return mol_energy.squeeze(-1)
        
class BDEFragmentModel(nn.Module):
    def __init__(self, node_dim=6, edge_dim=3):
        super().__init__()
        self.encoder = MPNNEncoder(node_dim, edge_dim)

    def forward(self, parent, frag1, frag2, return_embedding=False):
        if return_embedding:
            Ep,  zp  = self.encoder(parent, return_embedding=True)
            Ef1, zf1 = self.encoder(frag1,  return_embedding=True)
            Ef2, zf2 = self.encoder(frag2,  return_embedding=True)

            # choose what you want to visualize
            z = zp  # parent embedding (recommended)
            return Ef1 + Ef2 - Ep, z
        else:
            Ep  = self.encoder(parent)
            Ef1 = self.encoder(frag1)
            Ef2 = self.encoder(frag2)
            return Ef1 + Ef2 - Ep
        