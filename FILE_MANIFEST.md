# Quantum Data Decoder: Complete File Manifest

## Root Configuration Files
```
environment.yml                 - Conda environment specification with all dependencies
requirements.txt                - Pip requirements (fallback for non-conda users)
.gitignore                       - Git ignore patterns for data, models, cache
README.md                        - Main project documentation
IMPLEMENTATION_GUIDE.md          - Quick reference and next steps
FILE_MANIFEST.md                 - This file
```

## Source Code (`src/`)

### Core Modules
```
src/__init__.py                  - Package initialization
src/utils.py                     - Constants, helpers, utilities (~400 lines)
                                  • Physical constants (Boltzmann, Planck, c)
                                  • Amino acid properties (hydrophobicity, MW, pI)
                                  • Sequence encoding/decoding
                                  • Spectrum normalization methods
                                  • Logger setup and device detection

src/data_acquisition.py          - Data fetching and preprocessing (~500 lines)
                                  • PDBDataAcquisition: Query, filter, download PDB
                                  • KaggleDataAcquisition: Download competition data
                                  • SpectralDatabaseAcquisition: Experimental spectra
                                  • Batch processing with parallel downloads

src/nma_analysis.py              - Vibrational analysis pipeline (~600 lines)
                                  • ANMAnalyzer: Anisotropic Network Model
                                  • GNMAnalyzer: Gaussian Network Model
                                  • Normal mode computation
                                  • Vibrational entropy (S_vib)
                                  • Mode collectivity and fluctuations

src/spectral_generation.py       - Spectral synthesis (~450 lines)
                                  • SpectralGenerator: DOS with Lorentzian broadening
                                  • Raman-weighted and IR spectra
                                  • Instrumental response simulation
                                  • Feature extraction (integral, centroid, peaks)
                                  • DeltaSpectralFeatures for mutations
```

### Model Architecture (`src/models/`)
```
src/models/__init__.py           - Model package initialization
src/models/gnn.py                - Graph Neural Network encoder (~350 lines)
                                  • ProteinGNN: Graph Attention Network (GATv2)
                                  • GraphConstruction utilities
                                  • Node feature construction
                                  • Edge construction and weighting
                                  • Global pooling (mean + max)

src/models/cnn.py                - Spectral CNN encoder (~350 lines)
                                  • SpectralCNN: 1D CNN for vibrational spectra
                                  • ResidualBlock1D: Residual blocks with BN
                                  • SpectralFeatureExtractor: Handcrafted features
                                  • MultiScaleSpectralCNN: Multi-resolution variant

src/models/multimodal.py         - Fusion and end-to-end architecture (~400 lines)
                                  • VibroStructuralFusion: Concat/Bilinear/Attention
                                  • VibroStructuralModel: Complete architecture
                                  • Novozymes head (regression)
                                  • CAFA 5 head (multi-label classification)
                                  • RankingHead and ClassificationHead variants

src/models/losses.py             - Custom loss functions (~350 lines)
                                  • MarginRankingLossCustom: Pairwise ranking
                                  • SpearmanCorrelationLoss: Differentiable Spearman
                                  • FocalLoss: Class-imbalanced classification
                                  • ContrastiveLoss: Self-supervised pretraining
                                  • WeightedBCELoss: Class-weighted BCE
                                  • CombinedLoss: Multi-task learning
```

## Jupyter Notebooks (`notebooks/`)
```
notebooks/01_quickstart.ipynb    - Quick start demonstration
                                  • Environment verification
                                  • Core workflow overview
                                  • All 5 phases (Data → NMA → Spectra → GNN → Model)
                                  • Key concepts explanation
                                  • Next steps guide

notebooks/02_nma_prototype.ipynb - Detailed NMA analysis
                                  • Ubiquitin structure download and inspection
                                  • ANM computation (100 modes)
                                  • VDOS generation with variable broadening
                                  • Vibrational entropy at multiple temperatures
                                  • Spectral feature extraction
                                  • Comprehensive visualizations
                                  • Mode collectivity and residue fluctuations

notebooks/03_novozymes_execution.ipynb   - [To be created in Phase 2]
                                  • Competition data download and validation
                                  • Mutation preprocessing
                                  • Mass-perturbation NMA
                                  • Delta-feature engineering
                                  • Ranking model training
                                  • Test set evaluation

notebooks/04_cafa5_execution.ipynb       - [To be created in Phase 2]
                                  • AlphaFold structure processing
                                  • Spectral fingerprint generation
                                  • Multi-label classification pipeline
                                  • Taxon embedding integration
                                  • ESM-2 baseline ensemble
                                  • Submission preparation
```

## Data Directories (`data/`)
```
data/pdb/                        - PDB structure files
                                  • Downloaded .pdb files
                                  • Metadata and filtering results

data/kaggle/                     - Kaggle competition data
                                  • train.csv, test.csv (Novozymes)
                                  • train_updates.csv (corrections)
                                  • train_terms.csv, test_sequences.fasta (CAFA 5)

data/spectral/                   - Generated spectral data
                                  • .npz files with VDOS arrays
                                  • Synthetic spectra database
                                  • Visualizations

data/processed/                  - Preprocessed features
                                  • Graph representations
                                  • Tensor caches
                                  • Feature matrices
```

## Testing (`tests/`)
```
tests/                           - [To be populated in Phase 2]
                                  • test_data_acquisition.py
                                  • test_nma_analysis.py
                                  • test_models.py
                                  • Fixtures and utilities
```

---

## File Statistics

| Category | Files | Lines | Purpose |
|----------|-------|-------|---------|
| Configuration | 4 | 200 | Environment, version control |
| Data Acquisition | 1 | 500 | PDB/Kaggle/Spectral data fetching |
| NMA Analysis | 1 | 600 | Vibrational analysis |
| Spectral Generation | 1 | 450 | DOS synthesis and features |
| Model - GNN | 1 | 350 | Graph Neural Network |
| Model - CNN | 1 | 350 | 1D CNN for spectra |
| Model - Fusion | 1 | 400 | Multimodal architecture |
| Model - Losses | 1 | 350 | Custom loss functions |
| Utilities | 1 | 400 | Constants and helpers |
| Documentation | 3 | 300 | README, implementation guide, manifest |
| Notebooks | 2 | 600 | Tutorials and examples |
| **TOTAL** | **17** | **~4800** | **Phase 1 Complete** |

---

## Dependency Graph

```
environment.yml
    ├─ numpy, scipy, pandas          (data processing)
    ├─ prody, biopython              (structural biology)
    ├─ rdkit                         (cheminformatics)
    ├─ pytorch, pyg                  (deep learning)
    ├─ matplotlib, seaborn           (visualization)
    ├─ jupyter, jupyterlab           (notebooks)
    └─ kaggle                        (competition data)
```

## Import Hierarchy

```
src/
├─ utils.py                         (no dependencies on other src modules)
├─ data_acquisition.py              (depends: utils)
├─ nma_analysis.py                  (depends: utils, prody)
├─ spectral_generation.py           (depends: utils)
└─ models/
   ├─ gnn.py                        (depends: torch, torch_geometric)
   ├─ cnn.py                        (depends: torch)
   ├─ multimodal.py                 (depends: gnn, cnn, torch)
   ├─ losses.py                     (depends: torch)
   └─ __init__.py                   (imports all above)
```

---

## Execution Workflow

1. **Setup Phase**
   - Modify: environment.yml (if needed)
   - Run: `conda env create -f environment.yml`
   - Run: `conda activate quantum_decoder`

2. **Validation Phase**
   - Run: `python src/nma_analysis.py`      (tests ANM)
   - Run: `python src/spectral_generation.py` (tests DOS)
   - Run: `python src/models/gnn.py`        (tests GNN)
   - Run: `python src/models/cnn.py`        (tests CNN)
   - Run: `python src/models/multimodal.py` (tests fusion)

3. **Exploration Phase**
   - Open: `notebooks/01_quickstart.ipynb`
   - Open: `notebooks/02_nma_prototype.ipynb`

4. **Development Phase** (Phase 2)
   - Create: `notebooks/03_novozymes_execution.ipynb`
   - Create: `notebooks/04_cafa5_execution.ipynb`
   - Create: `src/training.py` (training loops)
   - Add: `tests/` (unit tests)

5. **Production Phase** (Phase 3)
   - Scale datasets
   - Finalize models
   - Submit to competitions

---

## Key Files for Quick Reference

### For Data Scientists
- [README.md](README.md) - Project overview
- [notebooks/01_quickstart.ipynb](notebooks/01_quickstart.ipynb) - Quick start
- [notebooks/02_nma_prototype.ipynb](notebooks/02_nma_prototype.ipynb) - NMA tutorial

### For ML Engineers
- [src/models/gnn.py](src/models/gnn.py) - GNN architecture
- [src/models/cnn.py](src/models/cnn.py) - CNN architecture
- [src/models/multimodal.py](src/models/multimodal.py) - Fusion and integration

### For Biophysicists
- [src/nma_analysis.py](src/nma_analysis.py) - Vibrational analysis
- [src/spectral_generation.py](src/spectral_generation.py) - Spectral methods
- [src/data_acquisition.py](src/data_acquisition.py) - Data sources

### For DevOps/Sys Admins
- [environment.yml](environment.yml) - Dependencies
- [.gitignore](.gitignore) - Git configuration
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Setup guide

---

**Generated:** December 2, 2025  
**Project Phase:** 1 Complete  
**Total Codebase:** ~4800 lines across 17 files
