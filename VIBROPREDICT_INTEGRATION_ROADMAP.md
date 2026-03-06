# VibroPredict Integration Roadmap: Aligning with Quantum Data Decoder

**Integration Date**: December 2, 2025  
**Status**: Planning Phase  
**Target**: Phase 3 Implementation (Weeks 5-8)

---

## Executive Summary

The VibroPredict Phase II Report outlines a paradigm shift in enzyme kinetics prediction through explicit vibrational spectral modeling. This document maps VibroPredict's technical requirements onto the existing Quantum Data Decoder (QDD) framework, identifying:

1. **Reusable Components** from QDD that accelerate VibroPredict
2. **New Modules** required specifically for enzyme kinetics
3. **Architecture Enhancements** to support KinHub-27k scale
4. **Timeline** for Phase 3 implementation

---

## Part I: Mapping VibroPredict to Quantum Data Decoder

### 1.1 Overlapping Capabilities

| Capability | QDD Status | VibroPredict Need | Reusability |
|-----------|-----------|------------------|------------|
| **Elastic Network Modeling** | ✓ Complete (nma_analysis.py) | GNM via ProDy | 100% |
| **VDOS Generation** | ✓ Complete (spectral_generation.py) | VDOS binning/broadening | 95% |
| **1D-CNN Spectral Encoder** | ✓ Complete (models/cnn.py) | Spectral branch | 90% |
| **Multimodal Fusion** | ✓ Complete (models/multimodal.py) | Attention fusion | 85% |
| **Training Infrastructure** | ✓ Complete (training.py) | Training loop | 80% |
| **Loss Functions** | ✓ Complete (models/losses.py) | Ranking loss available | 70% |
| **DataLoader Pipeline** | ✓ Complete (datasets.py) | Structure adaptation | 60% |

### 1.2 QDD Components Ready for Reuse

#### 1.2.1 Physics Pipeline (Direct Reuse)

```python
# From QDD src/nma_analysis.py
# ✓ ANMAnalyzer class → Adapt for enzyme structures
# ✓ GNM eigenmode computation → Use for VDOS generation
# ✓ Vibrational entropy calculation → Auxiliary features

from src.nma_analysis import ANMAnalyzer, GNMAnalyzer

# For VibroPredict enzymes:
gnm = GNMAnalyzer(pdb_structure, cutoff=10.0)
modes = gnm.compute_modes(k=None)  # All modes for VDOS
entropy = gnm.compute_vibrational_entropy()
```

#### 1.2.2 Spectral Generation (Direct Reuse)

```python
# From QDD src/spectral_generation.py
# ✓ SpectralGenerator class → Direct adoption
# ✓ Gaussian broadening → VDOS synthesis
# ✓ Feature extraction → Auxiliary scalar features

from src.spectral_generation import SpectralGenerator

sg = SpectralGenerator(freq_min=0, freq_max=500, n_points=1024)
vdos = sg.generate_dos(frequencies, broadening=1.0)  # 1D-CNN input
```

#### 1.2.3 Neural Network Encoders (Partial Reuse)

**Spectral CNN** (90% reusable):
```python
# From QDD src/models/cnn.py
from src.models.cnn import SpectralCNN

spectral_encoder = SpectralCNN(
    latent_dim=128,
    dropout=0.2,
    input_shape=(1, 1024)  # VDOS vector
)
```

**Multimodal Fusion** (85% reusable):
```python
# From QDD src/models/multimodal.py
# Use VibroStructuralFusion but adapt for:
# - Sequence (ProtT5 instead of graph)
# - Chemistry (SMILES Transformer instead of structured)
```

---

## Part II: New Modules Required for VibroPredict

### 2.1 Database Layer (New)

#### 2.1.1 KinHub-27k Loader

```python
# vibropredict/data/kinhub.py (NEW)

class KinHubDataset:
    """
    Load and parse KinHub-27k entries
    
    Attributes:
        - uniprot_id: UniProt accession
        - kinetic_values: {k_cat, K_m}
        - organism: Species
        - substrate_smiles: Canonicalized SMILES
        - mutation: Optional point mutation
        - source_paper: PubMed ID / DOI
    
    Methods:
        - parse_kinhub_csv()
        - validate_sequence()
        - resolve_ambiguities()
    """
```

**Integration with QDD**:
- Extends `src/datasets.py` base class
- Uses `src/utils.py` for normalization
- Output: Standardized DataFrame

#### 2.1.2 EnzyExtractDB Integration

```python
# vibropredict/data/enzyextract.py (NEW)

class EnzyExtractDB:
    """
    Filter and integrate EnzyExtractDB entries
    
    Quality gates:
        - UniProt ID validity
        - SMILES canonicalization
        - LLM confidence > 0.9
        - No overlap with KinHub-27k
    """
```

### 2.2 Structural Bioinformatics Layer (New)

#### 2.2.1 SIFTS UniProt↔PDB Mapper

```python
# vibropredict/structures/sifts_mapper.py (NEW)

class SIFTSMapper:
    """
    Map UniProt entries to optimal PDB structures
    
    Selection criteria:
        1. Sequence coverage (catalytic domain)
        2. Resolution (<2.5 Å preferred)
        3. Structure state (Apo or ligand-free)
        4. Completeness (minimal gaps)
    
    Methods:
        - score_pdb_candidates()
        - select_best_structure()
        - handle_unstructured()
    """
```

**Integration with QDD**:
- Uses `src/data_acquisition.py` for PDB fetching
- Extends filtering logic from QDD

#### 2.2.2 ESMFold Pipeline

```python
# vibropredict/structures/esmfold_runner.py (NEW)

class ESMFoldPredictor:
    """
    High-throughput structure prediction for sequences lacking PDB
    
    Features:
        - Batch inference (parallel across GPUs)
        - pLDDT quality control (>70 per-residue)
        - Global confidence filtering (<60 → exclusion)
        - Integration with GNM quality gates
    
    Methods:
        - predict_structure()
        - validate_plddt()
        - flag_disordered_regions()
    """
```

### 2.3 Spectral Feature Engineering Layer

#### 2.3.1 Enhanced VDOS Generator

```python
# vibropredict/spectra/vdos_engine.py (MODIFIED from QDD)

class VibroEnzymePipeline:
    """
    Enhanced version of spectral_generation.py for enzymes
    
    Additions:
        - Force field-aware GNM (distance-dependent spring constants)
        - Multi-mode spectrum (all 3N-6 modes, not just slow)
        - Thermodynamic features extraction
        - Spectral quality metrics (entropy convergence)
    
    Reuses from QDD:
        - SpectralGenerator.generate_dos() with tuned broadening
        - Normalization utilities from utils.py
    """
```

**Extension to QDD spectral_generation.py**:

```python
# Add to SpectralGenerator class:

class SpectralGenerator:
    def compute_enzyme_spectra(self, gnm_object, n_residues):
        """
        Generate VDOS from GNM for enzymes specifically
        
        Args:
            gnm_object: ProDy GNM instance
            n_residues: Number of residues (for 3N-6 calculation)
        
        Returns:
            vdos_vector: (1024,) numpy array
            auxiliary_features: dict with entropy, B-factors, etc.
        """
        # Calculate all eigenvalues
        eigenvalues = gnm_object.getEigens()[1]
        frequencies = np.sqrt(eigenvalues)
        
        # Gaussian broadening (ProDy-standard σ=1.0)
        vdos = self.generate_dos(frequencies, broadening=1.0)
        
        # Bin to 1024 points
        vdos_binned = np.histogram(vdos, bins=1024)[0]
        
        # Extract auxiliary features
        aux = {
            'vibrational_entropy': gnm_object.calcEntropy(),
            'b_factors': gnm_object.getDebyeWallerFactors(),
            'collectivity': self._compute_collectivity(eigenvalues)
        }
        
        return vdos_binned, aux
```

### 2.4 Model Architecture Extensions

#### 2.4.1 ProtT5 Sequence Encoder (New)

```python
# vibropredict/models/sequence_encoder.py (NEW)

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ProtT5Encoder:
    """
    Per-residue attention-weighted sequence embeddings
    
    Uses: ProtT5-XL-UniRef50 (huggingface)
    
    Architecture:
        - Tokenize sequence → T5 → Last hidden layer (1024-dim per residue)
        - Per-residue attention: learnable weights
        - Output: Global sequence embedding (1024-dim)
    
    Methods:
        - encode_sequence()
        - compute_attention_weights()
        - extract_active_site_focus()
    """
    
    def __init__(self, model_name="Rostlab/prot_t5_xl_uniref50"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.attention_layer = torch.nn.Linear(1024, 1)  # Learnable
    
    def forward(self, sequence):
        """
        sequence: str, e.g., "MKTIIALSYIF..."
        returns: (1024,) tensor
        """
        tokens = self.tokenizer(sequence, return_tensors="pt")
        embeddings = self.model.encoder(**tokens).last_hidden_state  # (L, 1024)
        
        # Compute per-residue attention
        weights = torch.softmax(self.attention_layer(embeddings), dim=0)
        
        # Weighted sum
        output = torch.sum(embeddings * weights, dim=0)
        return output
```

#### 2.4.2 SMILES Transformer + DRFP Encoder (New)

```python
# vibropredict/models/chemical_encoder.py (NEW)

from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from transformers import AutoTokenizer, AutoModel

class ChemicalEncoder:
    """
    Dual-branch chemical encoding:
        1. SMILES Transformer (semantic understanding)
        2. Differential Reaction Fingerprint (explicit bond changes)
    
    Combines:
        - Semantic: SMILES → BERT-like embedding
        - Explicit: Substrate SMILES ⊕ Product SMILES → DRFP
    
    Output: Concatenated (512 + 512 = 1024) vector
    """
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/SMILES_BERT_PubChem")
        self.model = AutoModel.from_pretrained("seyonec/SMILES_BERT_PubChem")
    
    def compute_drfp(self, substrate_smiles, product_smiles):
        """
        DRFP = Morgan(Product) XOR Morgan(Substrate)
        """
        substrate = Chem.MolFromSmiles(substrate_smiles)
        product = Chem.MolFromSmiles(product_smiles)
        
        fp_sub = AllChem.GetMorganFingerprintAsBitVect(substrate, 2, nBits=512)
        fp_prod = AllChem.GetMorganFingerprintAsBitVect(product, 2, nBits=512)
        
        drfp = np.array(fp_sub) ^ np.array(fp_prod)  # XOR
        return drfp
    
    def forward(self, substrate_smiles, product_smiles=None):
        """
        Returns: (1024,) embedding = [Transformer, DRFP]
        """
        # SMILES Transformer embedding
        tokens = self.tokenizer(substrate_smiles, return_tensors="pt")
        embedding = self.model(**tokens).last_hidden_state.mean(dim=1)[0]  # 512-dim
        
        # DRFP (if product available)
        if product_smiles:
            drfp = self.compute_drfp(substrate_smiles, product_smiles)
        else:
            drfp = np.zeros(512)
        
        combined = torch.cat([embedding, torch.from_numpy(drfp).float()])
        return combined
```

#### 2.4.3 VibroPredict Hybrid Model (New)

```python
# vibropredict/models/vibropredict_hybrid.py (NEW)

import torch
import torch.nn as nn
from .sequence_encoder import ProtT5Encoder
from .spectral_encoder import SpectralCNN  # From QDD
from .chemical_encoder import ChemicalEncoder

class VibroPredictHybrid(nn.Module):
    """
    Three-branch encoder with attention fusion
    
    Branches:
        1. Sequence: ProtT5 → (1024,)
        2. Spectral: 1D-CNN on VDOS → (128,)
        3. Chemical: SMILES + DRFP → (1024,)
    
    Fusion:
        - Attention gating for adaptive weighting
        - Handles missing modalities (MM-Drop training)
    
    Head:
        - Regression MLP → log₁₀(k_cat) scalar
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoders
        self.seq_encoder = ProtT5Encoder()
        self.spec_encoder = SpectralCNN(latent_dim=128)
        self.chem_encoder = ChemicalEncoder()
        
        # Fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(1024 + 128 + 1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
            nn.Softmax(dim=-1)  # Gate weights: α_seq, α_spec, α_chem
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(1024 + 128 + 1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, sequence, vdos, substrate_smiles, 
                product_smiles=None, drop_spectral=False):
        """
        drop_spectral: For MM-Drop training (zero out spectral branch)
        """
        # Encode each modality
        h_seq = self.seq_encoder(sequence)  # (1024,)
        h_spec = self.spec_encoder(vdos.unsqueeze(0).unsqueeze(0))  # (128,)
        h_chem = self.chem_encoder(substrate_smiles, product_smiles)  # (1024,)
        
        # Handle missing modality
        if drop_spectral:
            h_spec = torch.zeros_like(h_spec)
        
        # Concatenate
        h_concat = torch.cat([h_seq, h_spec, h_chem])  # (2176,)
        
        # Attention fusion
        gates = self.fusion_gate(h_concat)  # (3,): α_seq, α_spec, α_chem
        
        # Weighted fusion (for auxiliary interpretation)
        h_fused = (gates[0] * h_seq + 
                   gates[1] * h_spec[:min(len(h_spec), len(h_seq))] +
                   gates[2] * h_chem)
        
        # Regression
        logkcat = self.regressor(h_concat)
        
        return logkcat.squeeze(), gates  # (scalar), attention weights
```

### 2.5 Training Extensions

#### 2.5.1 Ranking Loss for Enzyme Kinetics (Extension to QDD)

```python
# Addition to src/models/losses.py

class MutantRankingLoss(nn.Module):
    """
    Extension for enzyme engineering: enforce correct ranking of mutants
    
    If k_cat(M1) > k_cat(M2), enforce ŷ(M1) > ŷ(M2)
    
    Loss = MSE + λ * RankingLoss
    """
    
    def forward(self, pred_pairs, target_pairs, lambda_rank=0.1):
        """
        pred_pairs: [(pred_m1, pred_m2), ...] from batched mutant pairs
        target_pairs: [(true_m1, true_m2), ...] ground truth ranking
        """
        mse_loss = nn.MSELoss()(pred_pairs.flatten(), target_pairs.flatten())
        
        # Ranking loss: penalize if ŷ(M1) < ŷ(M2) when y(M1) > y(M2)
        rank_loss = torch.sum(torch.relu(1 - (pred_pairs[:, 0] - pred_pairs[:, 1])))
        
        total = mse_loss + lambda_rank * rank_loss
        return total
```

#### 2.5.1 MM-Drop Training Strategy (Extension to QDD Trainer)

```python
# Addition to src/training.py

class TrainerWithMMDrop(Trainer):
    """
    Extension of Trainer class with Missing Modality dropout
    
    During training: Randomly drop spectral branch with prob p_drop
    Forces model to learn sequence+chemistry pathway independently
    Enables robustness to missing structures at inference
    """
    
    def train_epoch(self, train_loader, loss_fn, p_drop=0.25):
        """
        p_drop: Probability of dropping spectral branch per batch
        """
        total_loss = 0.0
        
        for batch in train_loader:
            # Randomly drop spectral modality
            drop_spectral = np.random.rand() < p_drop
            
            # Forward pass
            logkcat, gates = self.model(
                batch['sequence'],
                batch['vdos'],
                batch['substrate_smiles'],
                drop_spectral=drop_spectral
            )
            
            # Loss computation
            loss = loss_fn(logkcat, batch['log_kcat'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
```

---

## Part III: Data Pipeline Integration

### 3.1 Modified Dataset Class (Extension to QDD)

```python
# vibropredict/data/enzyme_kinetics_dataset.py (NEW, extends QDD)

from src.datasets import ProteinStructureDataset

class EnzymeKineticsDataset(ProteinStructureDataset):
    """
    Specialized dataset for enzyme kinetics with multimodal inputs
    
    Extends QDD's ProteinStructureDataset with:
        - KinHub-27k / EnzyExtractDB entries
        - Kinetic parameters (k_cat, K_m)
        - Substrate SMILES and products
        - Mutation information (WT vs mutant)
        - Optional structures (PDB or ESMFold)
    
    Returns:
        {
            'sequence': str,           # Protein sequence
            'uniprot_id': str,         # UniProt accession
            'log_kcat': float,         # log₁₀(k_cat) target
            'substrate_smiles': str,   # Canonicalized SMILES
            'product_smiles': str,     # Optional
            'vdos': np.array(1024),    # Vibrational spectrum
            'mutation': str,           # Optional (e.g., "A2C")
            'organism': str,           # Source organism
        }
    """
    
    def __init__(self, kinetics_csv, structures_dir, vdos_dir):
        super().__init__(csv_file=kinetics_csv, spectra_dir=vdos_dir)
        self.structures_dir = structures_dir
        self.kinetics_df = pd.read_csv(kinetics_csv)
    
    def __getitem__(self, idx):
        """
        Returns complete multimodal sample
        """
        row = self.kinetics_df.iloc[idx]
        
        # Load modalities
        sequence = row['protein_sequence']
        log_kcat = np.log10(float(row['k_cat']))
        substrate_smiles = row['substrate_smiles']
        product_smiles = row.get('product_smiles', None)
        
        # Load/compute spectrum
        vdos_file = f"{self.structures_dir}/{row['uniprot_id']}_vdos.npy"
        if os.path.exists(vdos_file):
            vdos = np.load(vdos_file)
        else:
            # Recompute if needed
            vdos = self._compute_vdos(row)
        
        return {
            'sequence': sequence,
            'uniprot_id': row['uniprot_id'],
            'log_kcat': log_kcat,
            'substrate_smiles': substrate_smiles,
            'product_smiles': product_smiles,
            'vdos': vdos,
            'mutation': row.get('mutation', 'WT'),
            'organism': row.get('organism', 'Unknown'),
        }
```

---

## Part IV: Implementation Timeline for Phase 3

### Week 5: Database Construction & Standardization

```
┌─────────────────────────────────────────────────────────────┐
│ Task 5.1: Data Acquisition & Parsing                       │
├─────────────────────────────────────────────────────────────┤
│ • Acquire KinHub-27k CSV (27,000 entries)                  │
│ • Fetch EnzyExtractDB via API (filter for high-quality)    │
│ • Parse UniProt IDs, extract sequences                      │
│ Est. Time: 3 days                                           │
│ Dependencies: None                                          │
│ Deliverable: kinetics_merged.csv (50k rows)               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 5.2: Standardization Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│ • Log-transform k_cat values                                │
│ • Canonicalize SMILES (RDKit)                              │
│ • Compute DRFP vectors (XOR fingerprints)                  │
│ • Validate UniProt identities                              │
│ • Flag and resolve ambiguities                             │
│ Est. Time: 4 days                                          │
│ Dependencies: Task 5.1                                     │
│ Deliverable: kinetics_standardized.csv (48k clean rows)   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 5.3: Data Splitting (UniRef50 Clustering)            │
├─────────────────────────────────────────────────────────────┤
│ • Cluster sequences at 50% identity (MMseqs2)             │
│ • Stratified assignment to Train/Val/Test                  │
│ • Add OOD benchmarks (TurNuP, MPEK test sets)             │
│ Est. Time: 3 days                                          │
│ Dependencies: Task 5.2                                     │
│ Deliverable: train.csv (38k), val.csv (5k), test.csv (5k) │
└─────────────────────────────────────────────────────────────┘
```

### Week 6: Structural Bioinformatics & VDOS Generation

```
┌─────────────────────────────────────────────────────────────┐
│ Task 6.1: SIFTS Mapping & PDB Selection                   │
├─────────────────────────────────────────────────────────────┤
│ • Query SIFTS for UniProt↔PDB mappings                     │
│ • Apply selection hierarchy (coverage, resolution, state)   │
│ • Fetch best PDB for each entry                            │
│ Est. Time: 3 days (parallel API calls)                     │
│ Dependencies: Task 5.2                                     │
│ Deliverable: pdb_mapping.csv (35k matched, 3k unmatched)  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 6.2: ESMFold Structure Prediction                     │
├─────────────────────────────────────────────────────────────┤
│ • Predict structures for 3k unmapped sequences              │
│ • pLDDT quality control (>70 per-residue)                  │
│ • Mark low-confidence for MM-Drop handling                 │
│ Est. Time: 2 days (GPU-parallel, 8 GPUs)                   │
│ Dependencies: Task 6.1                                     │
│ Deliverable: 3k ESMFold PDB files                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 6.3: ProDy GNM Calculations                           │
├─────────────────────────────────────────────────────────────┤
│ • Load all 38k structures (PDB or ESMFold)                 │
│ • Compute GNM for each (Kirchhoff matrix, eigenvalues)     │
│ • Generate all 3N-6 modes                                  │
│ • Extract thermodynamic features (entropy, B-factors)      │
│ Est. Time: 4 days (parallel, 16 cores)                     │
│ Dependencies: Task 6.2                                     │
│ Deliverable: 38k GNM eigenvalue files                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 6.4: VDOS Binning & Feature Engineering              │
├─────────────────────────────────────────────────────────────┤
│ • Gaussian broadening on eigenvalues (σ=1.0)              │
│ • Bin to 1024-point arrays                                 │
│ • Normalize (area = 3N-6)                                  │
│ • Save as .npy (vectorized for fast I/O)                   │
│ Est. Time: 2 days                                          │
│ Dependencies: Task 6.3                                     │
│ Deliverable: 38k VDOS .npy files + scalar features        │
└─────────────────────────────────────────────────────────────┘
```

### Week 7: Model Development & Baseline Training

```
┌─────────────────────────────────────────────────────────────┐
│ Task 7.1: Implement New Encoders                           │
├─────────────────────────────────────────────────────────────┤
│ • ProtT5Encoder (huggingface + attention)                   │
│ • ChemicalEncoder (SMILES Transformer + DRFP)              │
│ • Integration tests with sample data                        │
│ Est. Time: 3 days                                          │
│ Dependencies: None (parallel with Tasks 6.1-6.4)          │
│ Deliverable: vibropredict/models/sequence_encoder.py       │
│              vibropredict/models/chemical_encoder.py        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 7.2: Integrate with SpectralCNN (from QDD)            │
├─────────────────────────────────────────────────────────────┤
│ • Import src/models/cnn.py SpectralCNN                     │
│ • Adapt for 1024-dim VDOS input                            │
│ • Verify latent dimension (128)                             │
│ Est. Time: 1 day                                           │
│ Dependencies: QDD codebase                                 │
│ Deliverable: Verified encoder integration                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 7.3: Build VibroPredictHybrid Model                  │
├─────────────────────────────────────────────────────────────┤
│ • Attention fusion gate (learnable weights)                 │
│ • Regression MLP head                                       │
│ • MM-Drop masking mechanism                                │
│ • Test forward pass with dummy data                         │
│ Est. Time: 3 days                                          │
│ Dependencies: Tasks 7.1, 7.2                              │
│ Deliverable: vibropredict/models/vibropredict_hybrid.py    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 7.4: Training Infrastructure                          │
├─────────────────────────────────────────────────────────────┤
│ • Extend src/training.py with MM-Drop trainer              │
│ • Add MutantRankingLoss                                     │
│ • Composite loss (MSE + Ranking)                            │
│ • Create EnzymeKineticsDataset class                       │
│ Est. Time: 2 days                                          │
│ Dependencies: QDD training.py                              │
│ Deliverable: TrainerWithMMDrop class                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 7.5: Training Three Baselines                         │
├─────────────────────────────────────────────────────────────┤
│ Model 1: VibroPredictSeq (no spectral)                     │
│ • Train on (sequence, chemistry) only                       │
│ • 50 epochs, A100 GPU                                       │
│ • Target: R² ≈ 0.68 (baseline)                             │
│ • Est. Time: 12 hours                                      │
│                                                              │
│ Model 2: VibroPredictSpec (no sequence)                    │
│ • Train on (spectral, chemistry) only                       │
│ • 50 epochs                                                 │
│ • Target: R² ≈ 0.65 (physics-only)                         │
│ • Est. Time: 12 hours                                      │
│                                                              │
│ Model 3: VibroPredictHybrid (full model)                   │
│ • Train on all three branches + MM-Drop                    │
│ • 50 epochs                                                 │
│ • Target: R² > 0.75 (expected improvement)                 │
│ • Est. Time: 12 hours                                      │
│ Dependencies: Tasks 7.1-7.4                               │
│ Deliverable: 3 trained checkpoints                         │
└─────────────────────────────────────────────────────────────┘
```

### Week 8: Validation, Benchmarking & Documentation

```
┌─────────────────────────────────────────────────────────────┐
│ Task 8.1: Ablation Study Analysis                          │
├─────────────────────────────────────────────────────────────┤
│ • Compare VibroPredictSeq vs Hybrid: ΔR² = ?              │
│ • Compare VibroPredictSpec vs Hybrid: ΔR² = ?              │
│ • Focus on mutant subsets:                                  │
│   - Distal mutations (>10 Å from active site)             │
│   - Active site mutations (<5 Å)                           │
│   - Test hypothesis: Spectral gains on distal mutations    │
│ Est. Time: 2 days                                          │
│ Dependencies: Task 7.5                                     │
│ Deliverable: ablation_results.csv + visualizations        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 8.2: SOTA Comparison                                  │
├─────────────────────────────────────────────────────────────┤
│ • Download pre-trained: DLKcat, TurNuP, UniKP, MPEK       │
│ • Retrain on KinHub-27k for fair comparison               │
│ • Evaluate on same test set (5k OOD examples)              │
│ • Report RMSE, R², PCC, Spearman                           │
│ • Create Table 2 comparison                                │
│ Est. Time: 3 days                                          │
│ Dependencies: Task 7.5                                     │
│ Deliverable: sota_comparison_table.csv                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 8.3: Mutant Ranking Performance                       │
├─────────────────────────────────────────────────────────────┤
│ • Evaluate ranking loss on mutant pairs                     │
│ • Deep Mutational Scan compatibility (WT vs library)       │
│ • Enzyme engineering prediction accuracy                    │
│ Est. Time: 1 day                                           │
│ Dependencies: Task 7.5                                     │
│ Deliverable: ranking_metrics.csv                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 8.4: Missing Modality Analysis                        │
├─────────────────────────────────────────────────────────────┤
│ • Evaluate hybrid model when spectral branch zeroed         │
│ • Compare with VibroPredictSeq (sequence-only)             │
│ • Verify graceful degradation                              │
│ • Test on real enzymes lacking high-quality structures     │
│ Est. Time: 1 day                                           │
│ Dependencies: Task 7.5                                     │
│ Deliverable: mm_drop_analysis.md                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 8.5: Documentation & API                              │
├─────────────────────────────────────────────────────────────┤
│ • Update README with VibroPredict section                  │
│ • Document new modules and API                              │
│ • Create usage examples (basic inference)                   │
│ • Add notebooks: intro_vibropredict.ipynb                  │
│ Est. Time: 2 days                                          │
│ Dependencies: All previous tasks                           │
│ Deliverable: Complete documentation                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Task 8.6: Final Integration & Release Prep                 │
├─────────────────────────────────────────────────────────────┤
│ • Merge branches: vibropredict → main                      │
│ • Run full test suite                                       │
│ • Version bump (Phase 3.0)                                  │
│ • Create release notes                                      │
│ • Upload to GitHub/zenodo                                  │
│ Est. Time: 1 day                                           │
│ Dependencies: All previous tasks                           │
│ Deliverable: Public release with DOI                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Part V: Resource Requirements

### 5.1 Computational Resources

| Phase | Component | Hardware | Duration | Cost |
|-------|-----------|----------|----------|------|
| **Week 5** | Data processing | CPU (16 cores) | 10 days | $100 |
| **Week 6.1-6.2** | SIFTS + ESMFold | 8x A100 GPUs | 5 days | $4,000 |
| **Week 6.3-6.4** | ProDy + VDOS | CPU (32 cores) | 6 days | $200 |
| **Week 7.5** | Model training | 1x A100 GPU | 36 hours | $1,500 |
| **Week 8** | Analysis | CPU/GPU mixed | 7 days | $500 |
| **TOTAL** | - | - | 4 weeks | **~$6,300** |

### 5.2 Software Licenses

- All open-source (PyTorch, transformers, ProDy, RDKit)
- No additional licenses required
- GitHub Actions for CI/CD (free tier sufficient)

### 5.3 Personnel

| Role | Weeks | FTE |
|------|-------|-----|
| **Software Engineer** | 4 | 1.0 |
| **Computational Biologist** | 4 | 0.8 |
| **ML Engineer** | 4 | 0.6 |
| **Validation/Testing** | 2 | 0.5 |

---

## Part VI: Expected Deliverables by End of Phase 3

### Code Artifacts

```
vibropredict/
├── data/
│   ├── kinhub.py              # KinHub-27k loader
│   ├── enzyextract.py         # EnzyExtractDB integration
│   ├── standardization.py      # Kinetic parameter normalization
│   └── enzyme_kinetics_dataset.py  # Main dataset class
├── structures/
│   ├── sifts_mapper.py        # UniProt↔PDB mapping
│   ├── esmfold_runner.py      # ESMFold inference
│   └── quality_control.py     # pLDDT validation
├── spectra/
│   ├── vdos_engine.py         # Enhanced spectral generation
│   └── gnm_calculator.py      # ProDy GNM wrapper
├── models/
│   ├── sequence_encoder.py    # ProtT5 + attention
│   ├── spectral_encoder.py    # From QDD (adapted)
│   ├── chemical_encoder.py    # SMILES Transformer + DRFP
│   ├── fusion.py              # Attention fusion gate
│   └── vibropredict_hybrid.py # Complete hybrid model
├── training/
│   ├── losses.py              # Ranking + MSE composite
│   ├── trainer.py             # MM-Drop trainer extension
│   └── metrics.py             # Evaluation functions
├── notebooks/
│   ├── 05_vibropredict_intro.ipynb
│   ├── 06_database_construction.ipynb
│   ├── 07_model_training.ipynb
│   └── 08_ablation_study.ipynb
└── tests/
    ├── test_data_loaders.py
    ├── test_models.py
    └── test_training.py
```

### Data Artifacts

```
vibropredict_data/
├── kinhub_27k/
│   ├── train.csv (38k entries)
│   ├── val.csv (5k)
│   └── test.csv (5k)
├── structures/
│   ├── pdb/ (35k PDB files)
│   └── esmfold/ (3k predicted structures)
├── spectra/
│   └── vdos_vectors/ (38k .npy files)
├── checkpoints/
│   ├── vibropredict_seq.pt (baseline)
│   ├── vibropredict_spec.pt (physics-only)
│   └── vibropredict_hybrid.pt (SOTA)
└── results/
    ├── ablation_study.csv
    ├── sota_comparison.csv
    └── benchmark_plots/
```

### Scientific Contributions

1. **Novel Database**: VP-DB (VibroPredict Database)
   - 38,000 enzyme kinetics entries
   - Multimodal: sequence, structure, spectra, kinetics
   - Publicly available (CC-BY license)

2. **New Model Architecture**: VibroPredictHybrid
   - First to explicitly integrate vibrational dynamics
   - Missing Modality (MM-Drop) training strategy
   - Interpretable attention gates

3. **Ablation Study**: Quantifies spectral contribution
   - Expected ΔR² > 0.05 (hybrid vs sequence-only)
   - Analysis of mutation types (distal vs active site)

4. **Publications Ready**:
   - Main paper: "VibroPredict: Physics-Aware Deep Learning for Enzyme Kinetics"
   - Methods paper: "High-Throughput Vibrational Spectral Fingerprinting via ENM"
   - Benchmark paper: "A Rigorously Curated Enzyme Kinetics Database"

---

## Part VII: Integration with Quantum Data Decoder

### 7.1 Repository Structure (Post-Integration)

```
nobel_dataintelligence/
├── src/
│   ├── (existing Phase 1-2 modules)
│   ├── vibropredict_integration.md  # This document
│   ├── vdos_engine.py              # Extended spectral_generation.py
│   └── enzyme_models/               # New subdirectory
│       ├── sequence_encoder.py
│       ├── chemical_encoder.py
│       └── vibropredict_hybrid.py
├── vibropredict/                    # VibroPredict-specific modules
│   ├── data/ (as detailed above)
│   ├── structures/
│   ├── spectra/
│   ├── models/
│   ├── training/
│   └── notebooks/
├── data/
│   ├── vibropredict_db/            # VP-DB subdirectory
│   │   ├── kinhub/
│   │   ├── enzyextract/
│   │   └── processed/
│   └── (existing Phase 1-2 data)
└── (existing README, tests, docs)
```

### 7.2 Reuse of QDD Components

| QDD Module | VibroPredict Usage | Adaptation |
|-----------|------------------|-----------|
| `src/nma_analysis.py` | GNM calculations | Direct (no modification) |
| `src/spectral_generation.py` | VDOS binning | Minor (add enzyme-specific features) |
| `src/models/cnn.py` | Spectral encoder | Direct (1D-CNN reused) |
| `src/models/losses.py` | Base loss functions | Extend with ranking loss |
| `src/training.py` | Training loop | Extend with MM-Drop |
| `src/datasets.py` | Base dataset class | Subclass for enzyme kinetics |
| `src/utils.py` | Utilities | Direct (normalization, logging) |

### 7.3 Testing & CI/CD Integration

```yaml
# .github/workflows/vibropredict_tests.yml
name: VibroPredict Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest-cov
      - name: Run VibroPredict tests
        run: |
          pytest vibropredict/tests/ --cov=vibropredict
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Part VIII: Success Metrics & Validation Criteria

### 8.1 Quantitative Metrics

| Metric | Baseline (UniKP) | Target (VibroPredictHybrid) | Criterion |
|--------|------------------|----------------------------|-----------|
| **$R^2$ (all test)** | 0.68 | >0.75 | ✓ Success if Δ > 0.07 |
| **$R^2$ (mutants only)** | 0.65 | >0.72 | ✓ Success if Δ > 0.07 |
| **Distal mutation $R^2$** | 0.58 | >0.68 | ✓ Key hypothesis test |
| **RMSE (log scale)** | 0.42 | <0.36 | ✓ Lower is better |
| **Spearman's $\rho$** | 0.75 | >0.82 | ✓ Ranking performance |
| **Top-10 rank accuracy** | 0.68 | >0.75 | ✓ Engineering applications |

### 8.2 Qualitative Milestones

- ✓ **Data Quality**: VP-DB free of unit errors, identity errors (manual validation)
- ✓ **Reproducibility**: All results reproducible from public code + data
- ✓ **Interpretability**: Attention weights identify important protein regions
- ✓ **Robustness**: Model stable with missing modalities (MM-Drop)
- ✓ **Scalability**: Inference <100ms per protein on CPU

---

## Conclusion

VibroPredict Phase II represents an opportunity to transform enzyme kinetics prediction by explicitly incorporating the physics of protein dynamics. By leveraging the mature infrastructure of the Quantum Data Decoder and extending it with domain-specific modules for enzyme kinetics, we can accomplish this ambitious goal within 4 weeks and with realistic resource constraints.

The integration is designed to be non-disruptive—new code lives in a `vibropredict/` subdirectory and reuses QDD components without modification. Success is measurable via both quantitative metrics ($R^2 > 0.75$) and qualitative goals (interpretable, robust, scalable).

---

**Document Status**: Integration Roadmap - Ready for Implementation Approval  
**Prepared By**: VibroPredict Integration Team  
**Date**: December 2, 2025  
**Expected Completion**: End of Week 8 (December 26, 2025)
