# Bond dissociation energy as a programmable thermodynamic coordinate for inverse radical design

---

## Abstract and workflow

Homolytic bond dissociation energy (BDE) governs radical formation and bond-breaking thermodynamics, yet in molecular design it is typically evaluated only after candidate structures are proposed. Here we establish direct generative control over BDE, transforming bond strength from a post hoc descriptor into a programmable thermodynamic design coordinate. A BDE-conditioned transformer generates radical fragment pairs whose recombined parents achieve prescribed single-bond strengths across chemically accessible regimes (50-130 kcal·mol⁻¹), producing monotonic and target-resolved energetic distributions while preserving intrinsic bond-class thermodynamic limits and maintaining high validity and novelty across targets. Independent ωB97X-D3BJ/def2-TZVP calculations confirm target fidelity and energetic ordering, including prospective validation in unseen chemical regimes, demonstrating that the learned BDE axis reflects transferable thermodynamic structure rather than functional-specific regression. Electronic-structure analyses further reveal systematic evolution in radical SOMO-α localization and parent frontier orbitals along the conditioned axis, demonstrating that the model has internalized the physical determinants of homolytic bond strength. These findings support the broader use of reaction-relevant thermodynamic quantities as primary generative coordinates, with applicability to bond-strength regimes of practical consequence, including the high C-F dissociation energies that underpin PFAS environmental persistence.
```
Training Data (BDE labels + SMILES)
        │
        ├──► Train Generator (GPT-2 + FiLM conditioning)
        └──► Train Screener  (MPNN Graph Neural Network)
                    │
          Generate novel radical pairs
          at target BDE values (55–125 kcal/mol)
                    │
          Screen & filter with GNN predictions
                    │
          Sample novel candidates for DFT
                    │
          Write ORCA input scripts
          (Geometry Relax → SCF → Freq)
                    │
          Calculate DFT BDEs & compare to model
```

---

## Repository Structure

```
Code/
├── data/                              # Datasets
│   ├── bde_rdf_with_multi_halo_cfc_model_3.csv   # Main training data
│   ├── test_set_exp_bdes.csv                      # Experimental BDE test set
│   └── test_set_pfas.csv                          # PFAS test set
│
├── datasets/                          # PyTorch dataset classes
│   ├── generator.py                   # RadicalPairDataset (tokenized SMILES for GPT)
│   └── screener.py                    # BDEFragmentDataset (molecular graphs for GNN)
│
├── models/                            # Neural network architectures
│   ├── generator.py                   # BDEConditionedGPT (GPT-2 + FiLM conditioning)
│   └── screener.py                    # BDEFragmentModel (MPNN encoder)
│
├── training/                          # Training entry points
│   ├── generator.py                   # Train the generator
│   └── screener.py                    # Train the screener
│
├── design_and_evaluation/             # Generation, filtering, and analysis
│   ├── generate.py                    # Sample novel pairs from trained generator
│   ├── combine_radicals.py            # Bond two radical fragments into a parent molecule
│   ├── screener_prediction.py         # Predict BDE for generated pairs using screener
│   ├── evaluate_generated_pairs.py    # Validity / uniqueness / novelty / Tanimoto metrics
│   ├── sample_examples_for_dft.py     # Select novel candidates for DFT validation
│   └── calculate_dft_bde.py           # Parse ORCA outputs and compute DFT BDEs
│
└── write_orca_scripts/                # Quantum chemistry input generation
    ├── generate_relax_scripts.py      # SMILES → 3D coords → ORCA geometry relaxation inputs
    └── generate_scf_freq_scripts.py   # Optimized geometry → ORCA SCF + frequency inputs
│
└── generated_structures/              # Radical pairs generated after model training
    ├── generated_candidates_full_list.csv   # Full list with only the screener-predicted BDEs
    └── dft_candidates_with_dft_predicted_bde.csv  # Randomly-sampled list from the full list for dft validation
```

---

## Models

### Generator — `BDEConditionedGPT`
A GPT-2 language model conditioned on a target BDE value using **FiLM (Feature-wise Linear Modulation)**. Given a target BDE, it autoregressively generates a SMILES string encoding a `fragment1 <SEP> fragment2` radical pair.

- Tokenizer: ChemBERTa
- Architecture: GPT-2 (n_embd=256, n_layer=8, n_head=8)
- Training: 10 epochs, AdamW, batch size 128

### Screener — `BDEFragmentModel`
A message-passing neural network (MPNN) that predicts BDE from molecular graphs. It encodes the parent molecule and both fragments separately, then computes:

```
BDE = E(fragment1) + E(fragment2) − E(parent)
```

- Node features: atomic number, degree, formal charge, aromaticity, ring membership, H count
- Edge features: bond type, conjugation, ring membership
- Training: 50 epochs, Adam (lr=5e-4), L1 loss, ReduceLROnPlateau

---

## Data Format

The main dataset (`data/bde_rdf_with_multi_halo_cfc_model_3.csv`) contains:

| Column | Description |
|--------|-------------|
| `molecule` | Parent molecule SMILES |
| `fragment1` | First radical fragment SMILES |
| `fragment2` | Second radical fragment SMILES |
| `bond_type` | Type of cleaved bond (e.g., C-F, C-Cl) |
| `bde` | Bond dissociation energy (kcal/mol) |
| `set` | `train` or `test` split |

---

## Usage

### 1. Train the models

```bash
# Train the GNN screener
python -m training.screener

# Train the GPT generator
python -m training.generator
```

### 2. Generate novel radical pairs

```bash
python -m design_and_evaluation.generate
```

Generates 1000 fragment pairs at each of 8 target BDE values: 55, 65, 75, 85, 95, 105, 115, 125 kcal/mol.

### 3. Screen and filter

```bash
python -m design_and_evaluation.screener_prediction
```

### 4. Evaluate generated pairs

```bash
python -m design_and_evaluation.evaluate_generated_pairs
```

Reports validity, uniqueness, novelty, and Tanimoto similarity to training data.

### 5. Sample candidates for DFT

```bash
python -m design_and_evaluation.sample_examples_for_dft
```

Selects up to 20 novel pairs per BDE target and reconstructs parent molecules.

### 6. Write ORCA input scripts

```bash
# Step 1: Geometry relaxation
python -m write_orca_scripts.generate_relax_scripts

# Step 2: SCF + frequency calculations (after relaxation completes)
python -m write_orca_scripts.generate_scf_freq_scripts
```

### 7. Calculate DFT BDEs

```bash
python -m design_and_evaluation.calculate_dft_bde
```

Parses ORCA output files and computes BDE from SCF energies and zero-point energies (ZPE), converting from Hartree to kcal/mol (×627.509).

---

## Libraries and computational dependencies

- **PyTorch** + **PyTorch Geometric** — model training and GNN
- **Transformers** (HuggingFace) — GPT-2 and ChemBERTa tokenizer
- **RDKit** — cheminformatics (SMILES parsing, 3D coordinate generation)
- **pandas**, **numpy** — data handling
- **tqdm** — progress bars
- **ORCA** — external quantum chemistry package for DFT calculations

## Citation
```
@article{SheshanarayanaRadGen2026,
  author    = {R. Sheshanarayana and Fengqi You},
  title     = {Bond dissociation energy as a programmable thermodynamic coordinate for inverse radical design},
  journal   = {insert after publication},
  year      = {insert after publication},
  volume    = {insert after publication},
  pages     = {insert after publication},
  doi       = {insert after publication}
}
```
