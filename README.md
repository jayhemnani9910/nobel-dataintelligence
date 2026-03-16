[![Live Demo](https://img.shields.io/badge/Live_Demo-Explore-22d3ee?style=for-the-badge&logo=github)](https://jayhemnani9910.github.io/nobel-dataintelligence/)
[![Tests](https://img.shields.io/github/actions/workflow/status/jayhemnani9910/nobel-dataintelligence/tests.yml?style=for-the-badge&label=Tests&logo=githubactions)](https://github.com/jayhemnani9910/nobel-dataintelligence/actions)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

# Quantum Data Decoder

**Predicting protein stability and enzyme kinetics through the physics of molecular vibrations.**

Traditional ML treats proteins as static sequences or frozen structures. QDD treats them as what they actually are: **vibrating machines** whose dynamics encode function. We extract the Vibrational Density of States (VDOS) from Normal Mode Analysis and feed it — alongside sequence and chemical features — into a tri-modal deep learning architecture.

```
Protein Structure ──► NMA ──► VDOS Spectrum ──► SpectralCNN ──┐
                                                               ├──► TriModalFusion ──► Prediction
Amino Acid Sequence ──► ProtT5 Encoder ───────────────────────┤
                                                               │
Substrate SMILES ──► ChemBERTa + DRFP ───────────────────────┘
```

## Why This Matters

| What we do differently | Why it works |
|------------------------|--------------|
| **VDOS as input feature** | Encodes conformational dynamics invisible to static models |
| **3-branch fusion** | Sequence, spectral, and chemical modalities complement each other |
| **MM-Drop training** | Randomly drops branches during training for robustness to missing data |
| **Physics-informed** | Vibrational entropy directly relates to thermodynamic stability |

## Benchmarks

| Model | Year | R² | Key Modalities |
|-------|------|----|----------------|
| TurNuP | 2022 | 0.62 | Sequence + Fingerprints |
| DLKcat | 2021 | 0.64 | Sequence + Distance Graph |
| UniKP | 2023 | 0.68 | ProtT5 + SMILES |
| MPEK | 2023 | 0.70 | ESM-2 + SMILES |
| **VibroPredict** | **2025** | **>0.75** | **ProtT5 + VDOS + ChemBERTa** |

## Quick Start

```bash
git clone https://github.com/jayhemnani9910/nobel-dataintelligence.git
cd nobel-dataintelligence
conda env create -f environment.yml && conda activate quantum_decoder
pip install -r requirements.txt
python -m pytest tests/ -v  # 103+ tests
```

## Usage

### Predict enzyme kinetics

```python
from vibropredict.models import VibroPredictHybrid

model = VibroPredictHybrid(fusion_dim=512, dropout=0.2)
logkcat, gates = model(
    sequences=["MKTIIALSYIF..."],
    vdos=vdos_tensor,
    substrate_smiles=["CC(=O)O"],
)
print(f"k_cat = {10**logkcat:.1f} s⁻¹")
print(f"Attention gates: seq={gates[0,0]:.2f} spec={gates[0,1]:.2f} chem={gates[0,2]:.2f}")
```

### Train with MM-Drop

```python
from vibropredict.training import TrainerWithMMDrop, MutantRankingLoss

trainer = TrainerWithMMDrop(model, optimizer, device="cuda")
trainer.fit(
    train_loader, val_loader,
    loss_fn=MutantRankingLoss(lambda_rank=0.1),
    epochs=50, p_drop=0.25, patience=10,
)
```

### Run NMA on any protein

```python
from src.nma_analysis import ANMAnalyzer

anm = ANMAnalyzer("structure.pdb", cutoff=15.0)
frequencies, modes = anm.compute_modes(k=100)
vdos = anm.compute_vdos(k=100, broadening=5.0)
entropy = anm.compute_vibrational_entropy(k=100, temperature=298.15)
```

### Inference (with trained checkpoint)

```bash
# Predict enzyme stability
python -m src.cli predict-stability --sequence "ACDEFGHIKLMNPQRSTVWY" --pH 7.0

# Predict catalytic turnover
python -m src.cli predict-kcat --sequence "MKTIIALSYIF..." --smiles "CC(=O)O"
```

### Train on Google Colab

Open the ready-to-run notebooks on Colab Pro (A100):

| Notebook | What it trains | Time |
|----------|---------------|------|
| [`colab/train_novozymes.ipynb`](colab/train_novozymes.ipynb) | VibroStructuralModel on Novozymes (4k mutations) | ~15 min |
| [`colab/train_vibropredict.ipynb`](colab/train_vibropredict.ipynb) | VibroPredictHybrid with ProtT5 + ChemBERTa | ~30 min |

### CLI pipelines

```bash
python -m src.cli novozymes --data-dir ./data/kaggle --epochs 5
python -m src.cli cafa5 --data-dir ./data/cafa5 --top-k-terms 25
```

## Architecture

### Phase 1-2: Quantum Data Decoder (Core)

| Module | Purpose |
|--------|---------|
| `src/nma_analysis.py` | ANM/GNM normal mode analysis |
| `src/spectral_generation.py` | VDOS synthesis with Lorentzian broadening |
| `src/models/gnn.py` | GATv2 graph encoder for protein structures |
| `src/models/cnn.py` | 1D CNN spectral encoder |
| `src/models/multimodal.py` | Bilinear/attention fusion + task heads |
| `src/models/losses.py` | Ranking, focal, contrastive, correlation losses |
| `src/training.py` | Trainer with early stopping and checkpointing |
| `src/datasets.py` | Novozymes + CAFA5 dataset loaders |

### Phase 3: VibroPredict (Enzyme Kinetics)

| Module | Purpose |
|--------|---------|
| `vibropredict/models/` | ProtT5 encoder, ChemBERTa+DRFP encoder, TriModalFusion, hybrid model |
| `vibropredict/training/` | TrainerWithMMDrop, MutantRankingLoss, metrics |
| `vibropredict/data/` | KinHub-27k loader, EnzyExtractDB filter, standardization |
| `vibropredict/structures/` | SIFTS mapper, ESMFold predictor, pLDDT quality control |
| `vibropredict/spectra/` | GNM calculator, batch VDOS engine |
| `vibropredict/evaluation/` | Ablation runner, SOTA comparison, visualization |

## Project Structure

```
├── src/                    # QDD core (Phase 1-2)
│   ├── models/             # GNN, CNN, fusion, losses
│   ├── training.py         # Trainer + metrics
│   ├── datasets.py         # Dataset classes
│   └── nma_analysis.py     # Normal mode analysis
├── vibropredict/           # VibroPredict (Phase 3)
│   ├── models/             # ProtT5, ChemBERTa, fusion, hybrid
│   ├── training/           # MM-Drop trainer, ranking loss
│   ├── data/               # KinHub, EnzyExtract, dataset
│   ├── structures/         # SIFTS, ESMFold, QC
│   ├── spectra/            # GNM, VDOS engine
│   └── evaluation/         # Ablation, benchmarks, plots
├── notebooks/              # 4 QDD notebooks
├── vibropredict/notebooks/ # 4 VibroPredict notebooks
├── tests/                  # 103+ unit tests
├── demo/                   # Interactive GitHub Pages demo
└── docs/                   # Phase 2 report, future plans
```

## Notebooks

| # | Notebook | Topic |
|---|----------|-------|
| 01 | `notebooks/01_quickstart.ipynb` | Full pipeline overview |
| 02 | `notebooks/02_nma_prototype.ipynb` | NMA on Ubiquitin with VDOS plots |
| 03 | `notebooks/03_novozymes_execution.ipynb` | Enzyme stability prediction |
| 04 | `notebooks/04_cafa5_execution.ipynb` | GO term function prediction |
| 05 | `vibropredict/notebooks/05_vibropredict_intro.ipynb` | Tri-modal architecture demo |
| 06 | `vibropredict/notebooks/06_database_construction.ipynb` | VP-DB construction |
| 07 | `vibropredict/notebooks/07_model_training.ipynb` | Training with MM-Drop |
| 08 | `vibropredict/notebooks/08_ablation_study.ipynb` | Ablation + SOTA comparison |

## References

1. Bahar & Rader. "Coarse-Grained Normal Mode Analysis" (Curr. Opin. Struct. Biol., 2005)
2. Markelz et al. "Protein Dynamics and Hydration Water" (Biophys. J., 2010)
3. Engel et al. "Quantum Effects in Photosynthesis" (Nature, 2007)
4. Probst & Reymond. "Differential Reaction Fingerprints" (J. Chem. Inf. Model., 2020)

## Citation

```bibtex
@software{qdd2025,
  title={Quantum Data Decoder: Physics-Aware Deep Learning for Protein Function},
  year={2025},
  url={https://github.com/jayhemnani9910/nobel-dataintelligence}
}
```

## License

MIT License
