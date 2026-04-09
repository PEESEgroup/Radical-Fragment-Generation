# Harnessing homolytic bond energetics to steer inverse radical design
---

## Abstract and workflow

Homolytic bond dissociation energy (BDE) governs radical formation and bond-breaking thermodynamics, yet in molecular design it is typically evaluated only after candidate structures are proposed. Here we treat BDE as a continuous generative design coordinate rather than a post hoc descriptor, enabling distributional control over bond strength at the regime level rather than exact set-point attainment. A BDE-conditioned transformer generates radical fragment pairs steered toward prescribed single-bond strength regimes (50-130 kcal·mol⁻¹), achieving monotonic, rank-resolved energetic control with 81-94% validity and 84-92% novelty across targets. At the screening level, an MPNN screener trained on the same data source provides high-throughput energetic ranking consistent with regime-level steering. Same-functional M06-2X/def2-TZVP recalculation on randomly sampled novel generations yields target-attainment MAE of 7.7-12.7 kcal·mol⁻¹ consistent with regime-level steering, isolating generative accuracy from inter-functional offsets. Separate cross-functional evaluation at the ωB97X-D3BJ/def2-TZVP level shows preserved energetic ordering (DFT-calculated means spanning ~64 to ~119 kcal·mol⁻¹), indicating that the learned BDE axis reflects transferable thermodynamic structure rather than functional-specific artifacts. Spin-delocalization analysis further reveals a statistically significant correlation between radical localization and bond strength across the DFT-validated set. More broadly, reaction-relevant thermodynamic quantities may serve as primary generative coordinates, with PFAS-relevant C-F bond strengths motivating extension into higher-energy regimes.

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
│   ├── bde_rdf_with_multi_halo_cfc_model_3.pkl.zip # Main training data (use Pandas ".read_pickle" to read it)
│   ├── test_set_exp_bdes.csv                       # Experimental BDE test set
│   └── test_set_pfas.csv                           # PFAS test set
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
| `set` | `train`,  `valid`, or `test` split |

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
  title     = {Harnessing homolytic bond energetics to steer inverse radical design},
  journal   = {insert after publication},
  year      = {insert after publication},
  volume    = {insert after publication},
  pages     = {insert after publication},
  doi       = {insert after publication}
}
```
