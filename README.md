# Quantum Data Decoder (QDD)

## Overview

The Quantum Data Decoder is a comprehensive computational framework for predicting protein function and stability through the integration of vibrational spectroscopy, normal mode analysis (NMA), and multimodal deep learning (Graph Neural Networks + 1D CNNs).

**Core Innovation:** Proteins are treated as dynamic vibrational systems. The Vibrational Density of States (VDOS) encodes functional information orthogonal to static 3D structure. We bridge physics-informed spectral simulation with deep learning to predict macroscopic observables like melting temperature ($T_m$) and catalytic function.

## Project Structure

```
nobel_dataintelligence/
├── data/
│   ├── pdb/              # Downloaded PDB structures
│   ├── kaggle/           # Competition datasets (Novozymes, CAFA 5)
│   ├── spectral/         # Simulated/experimental vibrational spectra
│   └── processed/        # Preprocessed feature tensors
├── src/
│   ├── data_acquisition.py       # PDB, Kaggle, spectral data fetching
│   ├── nma_analysis.py           # Normal Mode Analysis (ANM/GNM)
│   ├── spectral_generation.py    # Density of States synthesis
│   ├── models/
│   │   ├── gnn.py                # Graph Attention Network encoder
│   │   ├── cnn.py                # 1D CNN spectral encoder
│   │   ├── multimodal.py         # Fusion architecture
│   │   └── losses.py             # Custom loss functions
│   ├── utils.py                  # Utilities (constants, helpers)
│   └── training.py               # Training loops
├── notebooks/
│   ├── 01_quickstart.ipynb               # Quick start and workflow overview
│   ├── 02_nma_prototype.ipynb            # NMA workflow validation
│   ├── 03_novozymes_execution.ipynb      # Enzyme stability prediction
│   └── 04_cafa5_execution.ipynb          # Function prediction
├── tests/
│   └── test_*.py                 # Unit tests
├── environment.yml               # Conda environment specification
├── requirements.txt              # Pip requirements (fallback)
└── README.md                     # This file
```

## Installation

### Prerequisites

- **Conda** installation (Miniconda or Anaconda)
- **GPU with CUDA 11.8** (optional but recommended for training)
- **~50GB disk space** for full PDB + competition datasets
- **Kaggle API credentials** (~/.kaggle/kaggle.json) for competition data

### Setup

1. **Clone and navigate to project:**
   ```bash
   cd /home/jey/projects/nobel_dataintelligence
   ```

2. **Create Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate quantum_decoder
   ```

3. **Configure Kaggle API** (if not already configured):
   ```bash
   # Download kaggle.json from https://www.kaggle.com/settings/account
   mkdir -p ~/.kaggle
   cp /path/to/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Verify installation:**
   ```python
   python -c "import prody; import torch; import rdkit; print('All imports successful!')"
   ```

## Key Components

### 1. Data Acquisition Pipeline (`src/data_acquisition.py`)
- **PDB Mining:** Download and filter high-resolution structures ($\text{resolution} < 2.5\text{Å}$)
- **Kaggle Integration:** Fetch Novozymes and CAFA 5 competition data
- **Spectral Database:** Ingest RamanBioLib, Sadtler, RRUFF experimental spectra

### 2. Vibrational Analysis (`src/nma_analysis.py`, `src/spectral_generation.py`)
- **Anisotropic Network Model (ANM):** Coarse-grained elastic network with $C_\alpha$ nodes
- **Normal Modes:** Compute first 100 non-zero modes via sparse linear algebra
- **Vibrational Entropy:** Calculate thermodynamic feature $S_{vib}$ for each structure
- **Density of States (DOS):** Synthesize continuous spectra using Lorentzian broadening

### 3. Multimodal Architecture (`src/models/`)
- **Spectral Encoder (1D CNN):** Process VDOS fingerprints
  - Input: 1D spectrum (1000 frequency bins)
  - Residual blocks with adaptive pooling
  - Output: 128-dim latent vector
  
- **Structural Encoder (GNN):** Process 3D graph topology
  - Graph Attention Network (GATv2)
  - Node features: amino acid type, physicochemical properties, pLDDT scores
  - Output: 128-dim latent vector via pooling
  
- **Fusion Mechanism:** Combine modalities via bilinear transformation or cross-attention
- **Task Heads:**
  - Novozymes: Ranking MLP + MarginRankingLoss for $\Delta\Delta G$ prediction
  - CAFA 5: Multi-label MLP + BCEWithLogitsLoss for GO term prediction

### 4. Competition Workflows
- **Novozymes Enzyme Stability:** Single-point mutation ranking via mass-perturbation NMA
- **CAFA 5 Function Prediction:** Multi-label classification with taxon embeddings + ESM-2 ensemble

## Quick Start

### 1. Prototype NMA on Ubiquitin
```python
# Run notebook: notebooks/02_nma_prototype.ipynb
from src.nma_analysis import ANMAnalyzer
from prody import parsePDB

pdb = parsePDB('1UBQ')  # Ubiquitin
analyzer = ANMAnalyzer(pdb)
modes = analyzer.compute_modes(k=10)
vdos = analyzer.compute_vdos()
```

### 2. Generate Spectral Data
```python
from src.spectral_generation import SpectralGenerator

gen = SpectralGenerator()
spectrum = gen.generate_dos(modes, broadening_factor=10)
s_vib = gen.compute_vibrational_entropy(frequencies)
```

### 3. Train Multimodal Model
```python
# Run notebook: notebooks/03_novozymes_execution.ipynb
from src.models.multimodal import Vibro_StructuralModel
from torch.utils.data import DataLoader

model = VibroStructuralModel(latent_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    for batch in train_loader:
        loss = train_step(model, batch, optimizer)
```

## Kaggle Competitions

### Novozymes Enzyme Stability Prediction
- **Task:** Rank single-point mutations by stability
- **Data:** ~4,000 mutations of a wildtype enzyme
- **Metric:** Spearman correlation of $T_m$ predictions
- **Approach:** Mass-perturbation NMA on wildtype structure + delta-feature engineering

### CAFA 5 Protein Function Prediction
- **Task:** Predict Gene Ontology (GO) terms from sequence and structure
- **Data:** ~4M+ proteins with AlphaFold structures
- **Metric:** Hierarchical F-max over GO hierarchy
- **Approach:** Spectrum-function correlation + taxon embeddings + ESM-2 ensemble

## Scientific References

1. Markelz, A. G., et al. "Protein Dynamics and Hydration Water" (Biophys. J., 2010)
2. Nelson, K. A., & Fayer, M. D. "Ultrafast Optical Measurements" (Chem. Rev., 1996)
3. Engel, G. S., et al. "Quantum Effects in Photosynthesis" (Nature, 2007)
4. Bahar, I., & Rader, A. J. "Coarse-Grained Normal Mode Analysis" (Curr. Opin. Struct. Biol., 2005)

## Outreach Strategy

We will engage leading biophysics laboratories to validate experimental hypotheses:
- **Dr. Andrea Markelz (Buffalo):** Anisotropic Terahertz Microspectroscopy
- **Dr. Keith Nelson (MIT):** Ultrafast Coherent Spectroscopy
- **Dr. Greg Engel (Chicago):** Vibronic Coupling in Photosynthesis

## Citation

If you use this project in your research, please cite:

```
@software{qdd2025,
  title={Quantum Data Decoder: Vibrational Analysis for Protein Function Prediction},
  author={[Your Name]},
  year={2025},
  url={https://github.com/user/nobel_dataintelligence}
}
```

## License

MIT License (see LICENSE file)

## Contact

For questions or collaboration inquiries, reach out to the project maintainers.
