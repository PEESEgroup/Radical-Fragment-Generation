import pandas as pd
from rdkit import Chem, DataStructs
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from rdkit.Chem import AllChem
from functools import lru_cache
from tqdm.auto import tqdm
from design_and_evaluation.combine_radicals import combine_fragments

bde2_v3 = pd.read_csv("data/bde_rdf_with_multi_halo_cfc_model_3.csv")
train_df = bde2_v3[bde2_v3["set"] == "train"]

# Need to have run "design_and_evaluation/generate.py" first to create the "generated_radical_pairs.csv" file.
saved_df = pd.read_csv("SAVED_GENERATED_PAIRS/generated_radical_pairs.csv")

def canonical_pair(f1, f2):
    m1 = Chem.MolFromSmiles(f1)
    m2 = Chem.MolFromSmiles(f2)
    if m1 is None or m2 is None:
        return None

    c1 = Chem.MolToSmiles(m1, canonical=True)
    c2 = Chem.MolToSmiles(m2, canonical=True)
    return tuple(sorted((c1, c2)))

gen_pairs = [
    p for p in (
        canonical_pair(f1, f2)
        for f1, f2 in zip(saved_df.fragment1, saved_df.fragment2)
    )
    if p is not None
]

train_pairs = [
    p for p in (
        canonical_pair(f1, f2)
        for f1, f2 in zip(train_df.fragment1, train_df.fragment2)
    )
    if p is not None
]

train_pair_set = set(train_pairs)

def fragment_pair_uniqueness(pairs):
    return len(set(pairs)) / len(pairs)

def fragment_pair_novelty(pairs, train_pair_set):
    novel = sum(p not in train_pair_set for p in pairs)
    return novel / len(pairs)

results = []

for bde, sub in tqdm(saved_df.groupby("bde"), desc="BDE groups"):

    total_raw = len(sub)

    canonical_pairs = []
    valid_merge_count = 0

    for f1, f2 in zip(sub.fragment1, sub.fragment2):

        # ---- Canonical pair (for uniqueness/novelty)
        p = canonical_pair(f1, f2)
        if p is not None:
            canonical_pairs.append(p)

        # ---- Validity (mergeability)
        if combine_fragments(f1, f2) is not None:
            valid_merge_count += 1

    if len(canonical_pairs) == 0:
        continue

    validity = valid_merge_count / total_raw
    uniqueness = fragment_pair_uniqueness(canonical_pairs)
    novelty = fragment_pair_novelty(canonical_pairs, train_pair_set)

    results.append({
        "bde": bde,
        "n_generated": total_raw,
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
    })

stats_df = pd.DataFrame(results).sort_values("bde")

print("Fragment-pair metrics by BDE")
print("------------------------------------------------------------")
for _, r in stats_df.iterrows():
    print(
        f"BDE {r.bde:>6.1f} | "
        f"N {int(r.n_generated):>4d} | "
        f"Validity {r.validity:.4f} | "
        f"Unique {r.uniqueness:.4f} | "
        f"Novel {r.novelty:.4f}"
    )

### Fragmena-pair-level Tanimoto similarity to training pairs ###

FS = 16
RADIUS = 2
N_BITS = 2048

GEN_SAMPLE   = 10000    # generated pairs per BDE
TRAIN_SAMPLE = 50000   # training pairs total

TARGETS = [55, 75, 95, 115]


@lru_cache(maxsize=300_000)
def fragment_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, RADIUS, nBits=N_BITS
    )

def pair_fp(s1, s2):
    fp1 = fragment_fp(s1)
    fp2 = fragment_fp(s2)
    if fp1 is None or fp2 is None:
        return None
    fp = DataStructs.ExplicitBitVect(N_BITS)
    fp |= fp1
    fp |= fp2
    return fp


train_fps = []
for _, r in tqdm(
    train_df.iterrows(),
    total=len(train_df),
    desc="TRAIN fragment-pair FPs",
):
    fp = pair_fp(r["fragment1"], r["fragment2"])
    if fp is not None:
        train_fps.append(fp)

# Subsample training for tractability
if len(train_fps) > TRAIN_SAMPLE:
    idx = np.random.choice(len(train_fps), TRAIN_SAMPLE, replace=False)
    train_fps = [train_fps[i] for i in idx]


all_sims = {}

for t in tqdm(TARGETS, desc="Targets"):
    gsub = saved_df[saved_df["bde"] == t]

    gen_fps = []
    for _, r in gsub.iterrows():
        fp = pair_fp(r["fragment1"], r["fragment2"])
        if fp is not None:
            gen_fps.append(fp)

    if len(gen_fps) == 0:
        all_sims[t] = []
        continue

    # Subsample generated
    if len(gen_fps) > GEN_SAMPLE:
        idx = np.random.choice(len(gen_fps), GEN_SAMPLE, replace=False)
        gen_fps = [gen_fps[i] for i in idx]

    sims = []
    for fp in tqdm(
        gen_fps,
        desc=f"GEN → TRAIN similarities (BDE={t})",
        leave=False,
    ):
        sims.extend(
            DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        )

    all_sims[t] = sims


fig, axes = plt.subplots(
    nrows=len(TARGETS),
    ncols=1,
    figsize=(8, 6),
    sharex=True,
)

x_grid = np.linspace(0, 1, 500)

palette = {
    55:  "#1f77b4",
    75:  "#ff7f0e",
    95:  "#2ca02c",
    115: "#d62728",
}

for ax, t in zip(axes, TARGETS):
    sims = np.array(all_sims[t])
    if len(sims) == 0:
        continue

    # KDE (fast, explicit control)
    kde = gaussian_kde(sims, bw_method=0.15)
    y = kde(x_grid)

    ax.fill_between(
        x_grid,
        y,
        color=palette[t],
        alpha=0.45,
    )
    ax.plot(
        x_grid,
        y,
        color=palette[t],
        lw=2,
    )

    # Label on the left
    ax.text(
        -0.03,
        0.5,
        f"{t} kcal/mol",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=FS,
    )

    # Clean style
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=FS * 0.8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

axes[-1].set_xlabel(
    "Tanimoto similarity \n (generated vs training fragment pairs)",
    fontsize=FS,
)

plt.tight_layout(h_pad=0.8)
plt.show()
