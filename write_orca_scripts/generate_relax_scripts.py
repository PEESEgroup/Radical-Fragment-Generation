import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Need to have run "generate_and_analyse/sample_examples_for_dft.py" first to create the "novel_examples_for_dft.csv" file.
novel_examples = pd.read_csv("SAVED_GENERATED_PAIRS/novel_examples_for_dft.csv")

BASE_DIR = "ORCA_RELAX_INPUTS"
XYZ_DIR  = os.path.join(BASE_DIR, "mmff_xyz")
INP_DIR  = os.path.join(BASE_DIR, "orca_inp")

METHOD_LINE = "! wB97X-D3BJ def2-TZVP DefGrid3 TightSCF Opt"
NPROCS = 8
RANDOM_SEED = 42

os.makedirs(XYZ_DIR, exist_ok=True)
os.makedirs(INP_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)


selected_df = novel_examples.reset_index(drop=True)
print(f"Selected {len(selected_df)} novel reactions.")


def write_orca(smiles, tag):

    # Special handling for atomic hydrogen radical
    if smiles.strip() in ["[H]", "H"]:
        inp_path = os.path.join(INP_DIR, f"{tag}.inp")

        with open(inp_path, "w") as f:
            f.write(f"{METHOD_LINE}\n\n")
            f.write("%pal\n")
            f.write(f"  nprocs {NPROCS}\n")
            f.write("end\n\n")
            f.write("* xyz 0 2\n")
            f.write("H  0.000000  0.000000  0.000000\n")
            f.write("*\n")

        print("Saved (atomic H):", tag)
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Failed:", smiles)
        return

    mol = Chem.AddHs(mol)

    # Single atom handling
    if mol.GetNumAtoms() == 1:
        atom = mol.GetAtomWithIdx(0)
        radical_e = atom.GetNumRadicalElectrons()
        charge = Chem.GetFormalCharge(mol)
        mult = radical_e + 1 if radical_e > 0 else 1

        inp_path = os.path.join(INP_DIR, f"{tag}.inp")

        with open(inp_path, "w") as f:
            f.write(f"{METHOD_LINE}\n\n")
            f.write("%pal\n")
            f.write(f"  nprocs {NPROCS}\n")
            f.write("end\n\n")
            f.write(f"* xyz {charge} {mult}\n")
            f.write(f"{atom.GetSymbol()}  0.000000  0.000000  0.000000\n")
            f.write("*\n")

        print("Saved (single atom):", tag)
        return

    # Normal embedding
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED

    if AllChem.EmbedMolecule(mol, params) != 0:
        print("Embedding failed:", smiles)
        return

    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    radical_e = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    charge = Chem.GetFormalCharge(mol)
    mult = radical_e + 1 if radical_e > 0 else 1

    inp_path = os.path.join(INP_DIR, f"{tag}.inp")

    with open(inp_path, "w") as f:
        f.write(f"{METHOD_LINE}\n\n")
        f.write("%pal\n")
        f.write(f"  nprocs {NPROCS}\n")
        f.write("end\n\n")
        f.write(f"* xyz {charge} {mult}\n")

        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(
                f"{atom.GetSymbol():<2} "
                f"{pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n"
            )

        f.write("*\n")

    print("Saved:", tag)

for i, row in selected_df.iterrows():

    bde_val = row["pred_bde"]
    parent  = row["parent"]
    frag1   = row["fragment1"]
    frag2   = row["fragment2"]

    base_tag = f"novel_{i}_bde_{int(round(bde_val))}"

    write_orca(parent, base_tag + "_parent")
    write_orca(frag1,  base_tag + "_frag1")
    write_orca(frag2,  base_tag + "_frag2")

print("All ORCA inputs generated.")