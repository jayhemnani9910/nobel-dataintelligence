[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open-2ea44f?style=for-the-badge)](https://jayhemnani9910.github.io/nobel-dataintelligence/)

# Quantum Data Decoder (QDD)

A multimodal deep learning framework for predicting protein stability and function by combining vibrational spectroscopy analysis (Normal Mode Analysis) with Graph Neural Networks and 1D CNNs.

**Core idea:** Proteins are dynamic vibrational systems. The Vibrational Density of States (VDOS) encodes functional information orthogonal to static 3D structure. QDD bridges physics-informed spectral simulation with deep learning to predict melting temperature (Tm) and catalytic function.

## Installation

### Prerequisites

- **Conda** (Miniconda or Anaconda)
- **GPU with CUDA 11.8** (optional but recommended)
- **Kaggle API credentials** (`~/.kaggle/kaggle.json`) for competition data

### Setup

```bash
conda env create -f environment.yml
conda activate quantum_decoder
pip install -r requirements.txt

# Verify
python -c "import torch, prody; print('OK')"
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
├── src/
│   ├── data_acquisition.py       # PDB, Kaggle, spectral data fetching
│   ├── nma_analysis.py           # Normal Mode Analysis (ANM/GNM)
│   ├── spectral_generation.py    # VDOS synthesis and features
│   ├── training.py               # Trainer, metrics, early stopping
│   ├── datasets.py               # PyTorch dataset classes
│   ├── cli.py                    # CLI pipelines (novozymes, cafa5)
│   ├── utils.py                  # Constants, helpers, FASTA parsing
│   └── models/
│       ├── gnn.py                # Graph Attention Network (GATv2)
│       ├── cnn.py                # 1D CNN spectral encoder
│       ├── multimodal.py         # Fusion + task heads
│       └── losses.py             # Ranking, focal, contrastive losses
├── notebooks/
│   ├── 01_quickstart.ipynb
│   ├── 02_nma_prototype.ipynb
│   ├── 03_novozymes_execution.ipynb
│   └── 04_cafa5_execution.ipynb
├── tests/
├── docs/
│   ├── PHASE2_REPORT.md          # Phase 2 completion details
│   └── future/                   # VibroPredict Phase 3 planning
├── environment.yml
└── requirements.txt
```

---

## Architecture

```
Protein Structure
    ├── GNN Encoder (GATv2, 3 layers)
    │   └── Node features: AA type + hydrophobicity + pLDDT → 128-dim
    │
    └── NMA → VDOS → CNN Encoder (residual 1D conv)
        └── 1000-bin spectrum → 128-dim

Fusion (bilinear / attention / concat)
    └── Combined embedding

Task Heads
    ├── Novozymes: MLP → Tm regression
    └── CAFA 5:    MLP → multi-label GO terms
```

---

## API Reference

### Model

```python
from src.models.multimodal import VibroStructuralModel

model = VibroStructuralModel(
    latent_dim=128,
    gnn_input_dim=24,
    fusion_type='bilinear',    # 'bilinear', 'concat', 'attention'
    dropout=0.2,
    num_go_terms=10000,
)

output = model(graph, spectra, global_features=None, task='novozymes')
# Novozymes: (batch, 1)  |  CAFA 5: (batch, num_go_terms)
```

### Training

```python
from src.training import Trainer, MetricComputer, create_training_config

config = create_training_config(task='novozymes')
trainer = Trainer(model, optimizer, device='cuda')

best_loss = trainer.fit(
    train_loader, val_loader, loss_fn,
    epochs=100,
    metric_fn=MetricComputer.spearman_correlation,
    early_stopping_patience=10,
    task='novozymes',
)
```

### Datasets

```python
from src.datasets import NovozymesDataset, CAFA5Dataset, create_dataloaders

# Novozymes
dataset = NovozymesDataset(
    csv_file='train.csv',
    structure_file='wildtype.pdb',
    spectra_dir='./spectral',
    include_updates=True,
)

# CAFA 5
dataset = CAFA5Dataset(
    sequences_fasta='sequences.fasta',
    terms_csv='train_terms.csv',
    spectra_dir='./spectral',
    structure_dir='./structures',
)

train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset, val_dataset, batch_size=32
)
```

### Loss Functions

```python
from src.models.losses import (
    MarginRankingLossCustom,   # Pairwise ranking for Novozymes
    FocalLoss,                 # Class-imbalanced multi-label
    PearsonCorrelationLoss,    # Correlation-based loss
    WeightedBCELoss,           # Weighted binary cross-entropy
    ContrastiveLoss,           # Self-supervised pre-training
)
```

### CLI

```bash
# Train + predict Novozymes
python -m src.cli novozymes --data-dir ./data/kaggle --epochs 5

# Baseline CAFA5 predictions
python -m src.cli cafa5 --data-dir ./data/cafa5 --top-k-terms 25
```

### NMA Analysis

```python
from src.nma_analysis import ANMAnalyzer

anm = ANMAnalyzer('structure.pdb', cutoff=15.0)
frequencies, eigenvectors = anm.compute_modes(k=100)
vdos = anm.compute_vdos(k=100, broadening=5.0)
s_vib = anm.compute_vibrational_entropy(k=100, temperature=298.15)
```

---

## Troubleshooting

**ModuleNotFoundError when importing:**
```python
import sys; sys.path.insert(0, './src')
```

**GPU out of memory:** Reduce batch size or use `device='cpu'`.

**Dataset loading fails:** Ensure data files are downloaded (use notebooks or `KaggleDataAcquisition`). Use absolute paths.

**Training loss doesn't decrease:** Try lower learning rate (`1e-4`), check data normalization with `normalize_spectrum()`, monitor gradient norms.

---

## Kaggle Competitions

| Competition | Task | Metric | Approach |
|-------------|------|--------|----------|
| Novozymes Enzyme Stability | Rank mutations by Tm | Spearman correlation | Mass-perturbation NMA + delta features |
| CAFA 5 Function Prediction | Predict GO terms | Hierarchical F-max | Spectrum-function correlation + taxon embeddings |

---

## VibroPredict (Phase 3): Enzyme Kinetics Prediction

VibroPredict extends QDD to predict enzyme catalytic turnover (k_cat) using a 3-branch hybrid model:

```
Sequence ──► ProtT5 Encoder ──────► h_seq (1024-dim)  ─┐
VDOS ──────► SpectralCNN ─────────► h_spec (128-dim)   ├─► TriModalFusion ──► MLP ──► log₁₀(k_cat)
SMILES ────► ChemBERTa + DRFP ───► h_chem (1024-dim)  ─┘
```

### Quick Start

```python
from vibropredict.models.vibropredict_hybrid import VibroPredictHybrid

model = VibroPredictHybrid(fusion_dim=512, dropout=0.2)
logkcat, gates = model(sequences, vdos, substrate_smiles)
```

### Training with MM-Drop

```python
from vibropredict.training.trainer import TrainerWithMMDrop
from vibropredict.training.losses import MutantRankingLoss

trainer = TrainerWithMMDrop(model, optimizer, device='cuda')
trainer.fit(train_loader, val_loader, MutantRankingLoss(), epochs=50, p_drop=0.25)
```

### Components

| Module | Purpose |
|--------|---------|
| `vibropredict/data/` | KinHub-27k loader, EnzyExtractDB filter, standardization, dataset |
| `vibropredict/structures/` | SIFTS mapper, ESMFold predictor, pLDDT quality control |
| `vibropredict/spectra/` | GNM calculator, VDOS engine |
| `vibropredict/models/` | ProtT5 encoder, ChemBERTa+DRFP encoder, 3-way fusion, hybrid model |
| `vibropredict/training/` | MutantRankingLoss, TrainerWithMMDrop, metrics |
| `vibropredict/evaluation/` | Ablation study, SOTA comparison, visualization |

See `vibropredict/notebooks/` for full walkthroughs (05-08).

---

## References

1. Markelz et al. "Protein Dynamics and Hydration Water" (Biophys. J., 2010)
2. Bahar & Rader. "Coarse-Grained Normal Mode Analysis" (Curr. Opin. Struct. Biol., 2005)
3. Engel et al. "Quantum Effects in Photosynthesis" (Nature, 2007)

## License

MIT License
