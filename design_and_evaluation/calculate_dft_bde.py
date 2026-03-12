from pathlib import Path
import re
import pandas as pd
import numpy as np

# This folder must contain the ORCA outputs (not inputs this time) for both SCF and FREQ calculations for parent and the two corresponding fragments
SCF_FREQ_DIR = Path("/content/drive/MyDrive/Molecular design/BDE_DFT/orca_sp_freq_extremes_out")

# Need to have run "design_and_evaluation/sample_examples_for_dft.py" first to calculate DFT BDEs.
novel_examples = pd.read_csv("SAVED_GENERATED_PAIRS/novel_examples_for_dft.csv")

HARTREE_TO_KCAL = 627.509

pattern = re.compile(r"(novel_\d+_bde_\d+)_parent_sp\.out")

index_to_base = {}

for f in SCF_FREQ_DIR.glob("*_parent_sp.out"):
    match = pattern.search(f.name)
    if match:
        base = match.group(1)              
        idx = int(base.split("_")[1])      
        index_to_base[idx] = base

print("Valid ORCA parent jobs found:", len(index_to_base))


novel_df = novel_examples.copy().reset_index(drop=True)
novel_df["novel_idx"] = novel_df.index

sampled_df = novel_df[
    novel_df["novel_idx"].isin(index_to_base.keys())
].copy().reset_index(drop=True)

print("Matched rows in novel_examples:", len(sampled_df))


def extract_energy(outfile):
    with open(outfile) as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                return float(line.split()[-1])
    raise RuntimeError(f"Energy not found in {outfile.name}")

def extract_zpe(outfile):
    with open(outfile) as f:
        for line in f:
            if "Zero point energy" in line:
                parts = line.replace("=", " ").split()
                for i, tok in enumerate(parts):
                    if tok == "Eh":
                        return float(parts[i-1])
    raise RuntimeError(f"ZPE not found in {outfile.name}")


dft_bdes = []
bad_rows = []

for idx, row in sampled_df.iterrows():

    novel_idx = int(row["novel_idx"])

    if novel_idx not in index_to_base:
        dft_bdes.append(np.nan)
        bad_rows.append(idx)
        continue

    base = index_to_base[novel_idx]   # exact filename prefix

    try:
        E_parent = extract_energy(SCF_FREQ_DIR / f"{base}_parent_sp.out")
        ZPE_parent = extract_zpe(SCF_FREQ_DIR / f"{base}_parent_freq.out")

        E_f1 = extract_energy(SCF_FREQ_DIR / f"{base}_frag1_sp.out")
        ZPE_f1 = extract_zpe(SCF_FREQ_DIR / f"{base}_frag1_freq.out")

        E_f2 = extract_energy(SCF_FREQ_DIR / f"{base}_frag2_sp.out")
        ZPE_f2 = extract_zpe(SCF_FREQ_DIR / f"{base}_frag2_freq.out")

        BDE = (
            (E_f1 + ZPE_f1)
            + (E_f2 + ZPE_f2)
            - (E_parent + ZPE_parent)
        ) * HARTREE_TO_KCAL

        if BDE < 0:
            dft_bdes.append(np.nan)
            bad_rows.append(idx)
        else:
            dft_bdes.append(BDE)

    except Exception as e:
        print("Failed:", base, e)
        dft_bdes.append(np.nan)
        bad_rows.append(idx)

sampled_df["dft_calculated_bde"] = dft_bdes


sampled_df_clean = sampled_df.dropna(
    subset=["dft_calculated_bde"]
).reset_index(drop=True)

sampled_df_clean.to_csv("SAVED_GENERATED_PAIRS/novel_examples_with_dft_bdes.csv", index=False)