# Quantum Data Decoder - Phase 2 Complete

## Overview

**Status**: Phase 1 Complete ✓ | Phase 2 Complete ✓

The Quantum Data Decoder is a production-ready framework for predicting protein stability and function through multimodal deep learning, combining vibrational spectroscopy analysis with structural information.

## Quick Links

- [Installation & Setup](#installation--setup)
- [Phase 1: Core Framework](#phase-1-core-framework) (Foundation infrastructure)
- [Phase 2: Competition Pipelines](#phase-2-competition-pipelines) (New in this phase)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)

---

## Installation & Setup

### 1. Clone and Setup Environment

```bash
cd /path/to/nobel_dataintelligence
conda env create -f environment.yml
conda activate qdd-env
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import torch, torch_geometric, prody, rdkit; print('All dependencies loaded successfully!')"
```

### 3. Test Framework

```bash
# Run all tests
python -m pytest tests/ -v

# Or use unittest
python -m unittest discover tests/
```

---

## Phase 1: Core Framework

### Architecture Overview

```
Input (Protein Structure)
    ├─ GNN Encoder (Graph Attention Networks)
    │   └─ Residue features (position, properties) → Graph edges (Cα contacts)
    │   
    └─ NMA Analysis (Vibrational Spectra)
        ├─ Anisotropic Network Model (100 modes)
        ├─ Vibrational Density of States (VDOS)
        └─ Spectral features → CNN Encoder

Fusion Layer (Bilinear/Attention/Concat)
    └─ 256-dim fused representation

Task-Specific Heads
    ├─ Regression: Melting temperature (Tm)
    └─ Multi-label: GO term classification (10K terms)
```

### Phase 1 Modules

| Module | Purpose | Key Classes |
|--------|---------|------------|
| `data_acquisition.py` | Multi-source data fetching | `PDBDataAcquisition`, `KaggleDataAcquisition`, `SpectralDatabaseAcquisition` |
| `nma_analysis.py` | Vibrational spectroscopy | `ANMAnalyzer`, `GNMAnalyzer` |
| `spectral_generation.py` | Synthetic spectrum synthesis | `SpectralGenerator`, `DeltaSpectralFeatures` |
| `models/gnn.py` | Graph neural network encoder | `ProteinGNN`, `GraphConstruction` |
| `models/cnn.py` | 1D CNN for spectra | `SpectralCNN`, `ResidualBlock1D` |
| `models/multimodal.py` | Fusion architecture | `VibroStructuralFusion`, `VibroStructuralModel` |
| `models/losses.py` | Custom loss functions | `MarginRankingLoss`, `FocalLoss`, `SpearmanCorrelationLoss` |
| `utils.py` | Utilities and constants | Constants, helpers, device detection |

---

## Phase 2: Competition Pipelines

### New Infrastructure

#### 1. Training Pipeline (`src/training.py`)

**Trainer Class**: Full training orchestration
- `train_epoch()`: Single epoch iteration with loss accumulation
- `validate()`: Validation loop with metric computation
- `fit()`: Complete training loop with early stopping and checkpointing
- `save_checkpoint()` / `load_checkpoint()`: Model persistence

**MetricComputer**: Static metric computation
- `spearman_correlation()`: Ranking metric for Novozymes
- `f_max_score()`: Multi-label metric for CAFA 5
- `mean_squared_error()`, `mean_absolute_error()`: Regression metrics
- `accuracy()`: Classification metric

**EarlyStopping**: Training callback
- Monitors validation metric
- Saves best checkpoint
- Stops if no improvement after patience epochs

```python
from training import Trainer, MetricComputer, create_training_config

# Create config
config = create_training_config(task='novozymes')
trainer = Trainer(model, optimizer, scheduler, device='cuda')

# Train
best_loss = trainer.fit(
    train_loader, val_loader, loss_fn,
    epochs=100,
    metric_fn=MetricComputer.spearman_correlation,
    early_stopping_patience=10,
    task='novozymes'
)
```

#### 2. Dataset Classes (`src/datasets.py`)

**ProteinStructureDataset**: Generic dataset for structures + spectra
- Lazy loading with optional caching
- Custom collation for PyG Data objects
- Handles protein structures, spectra, and optional metadata

**NovozymesDataset**: Specialized for stability prediction
- Applies `train_updates.csv` corrections
- Handles pH as global feature
- Implements mutation masking

**CAFA5Dataset**: Multi-label function prediction
- Loads FASTA sequences and GO annotations
- Creates multi-hot label vectors
- Manages 10K GO term vocabulary

```python
from datasets import NovozymesDataset, CAFA5Dataset, create_dataloaders

# Novozymes
nov_dataset = NovozymesDataset(
    csv_file='train.csv',
    structure_file='wildtype.pdb',
    spectra_dir='./spectral_data',
    include_updates=True
)

# CAFA 5
cafa_dataset = CAFA5Dataset(
    terms_file='train_terms.csv',
    sequences_file='test_sequences.fasta',
    structures_dir='./pdb_structures',
    vocab_file='go_vocabulary.csv'
)

# Create DataLoaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset, val_dataset, test_dataset,
    batch_size=32, num_workers=4
)
```

### Competition Notebooks

#### 3. Novozymes Execution (`notebooks/03_novozymes_execution.ipynb`)

**Task**: Predict melting temperature (Tm) for single-point mutations

**Pipeline**:
1. Download Novozymes competition data
2. Load and explore Tm distribution
3. Precompute mutation spectra (NMA-based)
4. Create NovozymesDataset and DataLoaders
5. Initialize VibroStructuralModel for regression
6. Train with MSE loss, optimize Spearman correlation
7. Evaluate on held-out test set
8. Generate Kaggle submission with predictions

**Key Sections**:
- Data exploration: Distribution analysis, pH stratification
- Training configuration: Batch size, learning rate, patience
- Model inference: Batch predictions with uncertainty quantification
- Submission format: CSV with predicted Tm values

#### 4. CAFA 5 Execution (`notebooks/04_cafa5_execution.ipynb`)

**Task**: Predict Gene Ontology (GO) terms for novel proteins

**Pipeline**:
1. Download CAFA 5 data (30K+ proteins, 10K GO terms)
2. Build GO vocabulary and term frequency analysis
3. Load sequences and retrieve 3D structures (AlphaFold DB)
4. Precompute spectral features for all proteins
5. Create CAFA5Dataset with multi-hot labels
6. Initialize VibroStructuralModel for multi-label classification
7. Train with weighted BCE or Focal loss
8. Optimize classification threshold for F-max metric
9. Generate hierarchical GO predictions
10. Ensemble with ESM-2 baseline (optional)

**Key Sections**:
- GO analysis: Term frequency, protein-term relationships
- Threshold optimization: F-max score across decision thresholds
- Label hierarchy: GO directed acyclic graph constraints
- Submission format: Protein-GO predictions with confidence scores

---

## API Reference

### Model Classes

#### VibroStructuralModel

```python
from models.multimodal import VibroStructuralModel

model = VibroStructuralModel(
    latent_dim=128,              # Embedding dimension
    gnn_input_dim=24,            # Node feature dimension
    fusion_type='bilinear',      # 'bilinear', 'concat', 'attention'
    dropout=0.2,                 # Dropout rate
    num_go_terms=10000           # Number of GO terms (for CAFA 5)
)

# Forward pass
output = model(
    graph,                       # PyG Data object or list of Data
    spectra,                     # (batch, 1, spectrum_length) tensor
    global_features=None,        # (batch, n_features) optional (pH, etc.)
    task='novozymes'             # 'novozymes' or 'cafa5'
)
# Novozymes: output shape (batch, 1)
# CAFA 5: output shape (batch, num_go_terms)
```

#### ProteinGNN

```python
from models.gnn import ProteinGNN

gnn = ProteinGNN(
    latent_dim=128,
    num_layers=4,
    heads=4,
    dropout=0.2
)

# Input: PyG Data object
output = gnn(data)  # Output: (batch_size, latent_dim)
```

#### SpectralCNN

```python
from models.cnn import SpectralCNN

cnn = SpectralCNN(
    latent_dim=128,
    dropout=0.2
)

# Input: (batch, 1, spectrum_length)
output = cnn(spectra)  # Output: (batch, latent_dim)
```

### Loss Functions

```python
from models.losses import (
    MarginRankingLossCustom,
    FocalLoss,
    WeightedBCELoss,
    SpearmanCorrelationLoss
)

# Novozymes: Ranking loss
loss_fn = MarginRankingLossCustom(margin=1.0)
loss = loss_fn(preds_wt, preds_mut, labels)

# CAFA 5: Focal loss for class imbalance
loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
loss = loss_fn(logits, targets)

# Alternative: Weighted BCE
loss_fn = WeightedBCELoss(weight=2.0)
loss = loss_fn(logits, targets)
```

### Training

```python
from training import Trainer, MetricComputer, create_training_config

# Get task-specific config
config = create_training_config(task='novozymes')
# Returns: {'batch_size': 32, 'learning_rate': 1e-3, ...}

# Create trainer
trainer = Trainer(model, optimizer, scheduler, device)

# Train
best_loss = trainer.fit(
    train_loader, val_loader, loss_fn,
    epochs=100,
    metric_fn=MetricComputer.spearman_correlation,
    early_stopping_patience=10,
    task='novozymes'
)

# Compute metrics
spearman = MetricComputer.spearman_correlation(predictions, targets)
f_max = MetricComputer.f_max_score(probabilities, labels)
mse = MetricComputer.mean_squared_error(predictions, targets)
```

### Data Loading

```python
from datasets import NovozymesDataset, CAFA5Dataset

# Novozymes
dataset = NovozymesDataset(
    csv_file='train.csv',
    structure_file='wildtype.pdb',
    spectra_dir='./spectral_data',
    include_updates=True  # Apply train_updates.csv corrections
)

# CAFA 5
dataset = CAFA5Dataset(
    terms_file='train_terms.csv',
    sequences_file='test_sequences.fasta',
    structures_dir='./pdb_structures',
    vocab_file='go_vocabulary.csv'
)

# Access sample
sample = dataset[0]
# Contains: {'graph', 'spectra', 'labels', 'global_features' (optional)}
```

---

## Running Tests

### Test Suite Overview

The test suite covers ~80% of core modules:

| Test File | Coverage | Tests |
|-----------|----------|-------|
| `test_data_loading.py` | Dataset classes, DataLoader batching | 8 tests |
| `test_models.py` | GNN, CNN, fusion, complete models | 14 tests |
| `test_training.py` | Trainer, metrics, early stopping, checkpoints | 12 tests |

### Run Tests

```bash
# All tests
python -m pytest tests/ -v
python -m unittest discover tests/

# Specific module
python -m pytest tests/test_models.py -v
python -m unittest tests.test_models

# Specific test
python -m pytest tests/test_models.py::TestProteinGNN::test_forward_pass_small_graph -v
python -m unittest tests.test_models.TestProteinGNN.test_forward_pass_small_graph
```

### Test Examples

```python
# Data loading test
from tests.test_data_loading import TestNovozymesDataset
test = TestNovozymesDataset()
test.setUp()
test.test_novozymes_dataset_creation()

# Model test
from tests.test_models import TestProteinGNN
test = TestProteinGNN()
test.setUp()
test.test_forward_pass_small_graph()

# Training test
from tests.test_training import TestTrainer
test = TestTrainer()
test.setUp()
test.test_train_epoch_basic()
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError" when importing

**Solution**:
```python
import sys
sys.path.insert(0, './src')
from models import VibroStructuralModel
```

### Issue: GPU out of memory

**Solution**:
```python
# Reduce batch size
config = create_training_config(task='novozymes')
config['batch_size'] = 16  # Instead of 32

# Or use CPU
trainer = Trainer(model, optimizer, device='cpu')
```

### Issue: Dataset loading fails

**Common causes**:
1. PDB files not downloaded: Use `KaggleDataAcquisition.download_novozymes()`
2. Spectrum files not found: Run precomputation in notebook
3. CSV paths incorrect: Use absolute paths

**Solution**:
```python
from pathlib import Path
csv_file = Path('./data/kaggle/train.csv').resolve()
structure_file = Path('./data/kaggle/wildtype.pdb').resolve()
```

### Issue: Training loss doesn't decrease

**Causes**: Learning rate too high/low, incorrect loss function, bad data normalization

**Solution**:
```python
# Try different learning rate
config = create_training_config(task='novozymes')
config['learning_rate'] = 1e-4  # Try lower

# Check data normalization
from utils import normalize_spectrum
spectrum_normalized = normalize_spectrum(spectrum, method='zscore')

# Monitor gradients
for name, param in model.named_parameters():
    print(f"{name}: grad norm = {param.grad.norm()}")
```

### Issue: Low validation metrics

**Causes**: Insufficient training data, model underfitting, wrong metric

**Solution**:
1. Increase training epochs
2. Use more complex model (more layers, larger latent_dim)
3. Verify metric computation: `MetricComputer.spearman_correlation()`
4. Check train/val split (should be 70/15/15)

### Issue: Notebook imports failing

**Solution**:
```python
import sys
sys.path.insert(0, './src')

# Then import
from models.multimodal import VibroStructuralModel
from datasets import NovozymesDataset
from training import Trainer
```

---

## File Structure

```
nobel_dataintelligence/
├── README.md                          # This file
├── environment.yml                    # Conda environment
├── requirements.txt                   # Pip dependencies
├── IMPLEMENTATION_GUIDE.md            # Phase reference
├── FILE_MANIFEST.md                   # File listing and stats
│
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py           # Data downloading (PDB, Kaggle)
│   ├── nma_analysis.py               # Vibrational analysis
│   ├── spectral_generation.py        # Spectrum synthesis
│   ├── training.py                   # [Phase 2] Trainer, metrics, callbacks
│   ├── datasets.py                   # [Phase 2] PyTorch dataset classes
│   ├── utils.py                      # Utilities and constants
│   └── models/
│       ├── gnn.py                    # Graph neural network
│       ├── cnn.py                    # 1D CNN for spectra
│       ├── multimodal.py             # Fusion and complete model
│       └── losses.py                 # Custom loss functions
│
├── notebooks/
│   ├── 01_quickstart.ipynb           # Phase 1: Full workflow overview
│   ├── 02_nma_prototype.ipynb        # Phase 1: Detailed NMA analysis
│   ├── 03_novozymes_execution.ipynb  # [Phase 2] Stability prediction
│   └── 04_cafa5_execution.ipynb      # [Phase 2] Function prediction
│
├── tests/                             # [Phase 2] Unit tests
│   ├── __init__.py
│   ├── test_data_loading.py          # Dataset and DataLoader tests
│   ├── test_models.py                # Architecture and forward pass tests
│   └── test_training.py              # Trainer and metric tests
│
└── data/
    ├── kaggle/                        # Competition datasets (download via notebook)
    ├── pdb_structures/                # PDB files
    ├── spectral/                      # Precomputed spectra
    └── cafa5/                         # CAFA 5 data
```

---

## Performance Summary

| Component | Lines | Tests | Status |
|-----------|-------|-------|--------|
| Core framework | 4,450 | - | ✓ Complete |
| Training pipeline | 800 | 12 | ✓ Complete |
| Competitions | 1,200 | - | ✓ Complete |
| Unit tests | 1,100 | 34 | ✓ Complete |
| **Total** | **~8,000** | **34** | **✓ Complete** |

---

## Next Steps

1. **Download competition data**: Use Kaggle CLI to download Novozymes and CAFA 5 datasets
2. **Run notebooks**: Execute 03_novozymes_execution.ipynb and 04_cafa5_execution.ipynb with real data
3. **Optimize hyperparameters**: Tune learning rate, batch size, fusion type for your data
4. **Ensemble models**: Combine predictions from multiple trained models
5. **Submit predictions**: Generate Kaggle submissions for both competitions

---

## Citation

```bibtex
@article{quantum_data_decoder,
  title={Quantum Data Decoder: Vibrational Spectroscopy and Deep Learning for Protein Prediction},
  author={Your Names},
  year={2024},
  note={Quantum Data Intelligence Framework}
}
```

---

## License

MIT License - See LICENSE file for details.

---

**Last Updated**: Phase 2 Complete
**Next Release**: Optimizations and ensemble methods
