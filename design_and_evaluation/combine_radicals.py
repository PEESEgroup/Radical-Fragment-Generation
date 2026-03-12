from rdkit import Chem

def find_radical_atom(mol):
    # Standardizing search for the radical center
    idxs = [a.GetIdx() for a in mol.GetAtoms()
            if a.GetNumRadicalElectrons() == 1]
    if len(idxs) != 1:
        raise ValueError(f"Expected 1 radical, found {len(idxs)}")
    return idxs[0]

def combine_fragments(frag1_smiles, frag2_smiles, bond_type=Chem.rdchem.BondType.SINGLE):
    try:
        frag1 = Chem.MolFromSmiles(frag1_smiles)
        frag2 = Chem.MolFromSmiles(frag2_smiles)

        if frag1 is None or frag2 is None:
            return None

        # Find attachment atoms via radicals
        a1 = find_radical_atom(frag1)
        a2 = find_radical_atom(frag2)

        # Combine molecules
        combo = Chem.CombineMols(frag1, frag2)
        rw = Chem.RWMol(combo)

        # Global index for second fragment
        a2_global = frag1.GetNumAtoms() + a2

        # Add bond (defaulting to SINGLE as is typical for BDE pairs)
        rw.AddBond(a1, a2_global, bond_type)

        # Remove radicals
        rw.GetAtomWithIdx(a1).SetNumRadicalElectrons(0)
        rw.GetAtomWithIdx(a2_global).SetNumRadicalElectrons(0)

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)

        return Chem.MolToSmiles(mol, canonical=True)

    except (ValueError, RuntimeError):
        # Catches 'Expected 1 radical' and RDKit sanitization errors
        return None