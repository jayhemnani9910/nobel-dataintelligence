"""
Utilities for Quantum Data Decoder

Shared constants, helper functions, and data processing utilities.
"""

import logging
from pathlib import Path
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ============================================================================
# Physical Constants
# ============================================================================

# Boltzmann constant (J/K)
BOLTZMANN_CONSTANT = 1.380649e-23

# Reduced Planck constant (J*s)
HBAR = 1.054571817e-34

# Speed of light (m/s)
SPEED_OF_LIGHT = 299792458

# Avogadro's number
AVOGADRO = 6.02214076e23

# ============================================================================
# Amino Acid Properties
# ============================================================================

# Standard amino acids (20) + Unknown (X) + Gap (Z)
AMINO_ACIDS = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
    'X', 'Z'
]

# Amino acid to index mapping
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

# Hydrophobicity scores (Kyte-Doolittle scale)
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5,
    'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
    'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
    'Y': -1.3, 'V': 4.2, 'X': 0.0, 'Z': -3.5
}

# Molecular weight (Da)
MOLECULAR_WEIGHT = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
    'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.15, 'I': 131.17,
    'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
    'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15,
    'X': 137.14, 'Z': 147.13  # Average
}

# Isoelectric point
ISOELECTRIC_POINT = {
    'A': 6.00, 'R': 10.76, 'N': 5.41, 'D': 2.77, 'C': 5.07,
    'Q': 5.65, 'E': 3.22, 'G': 5.97, 'H': 7.59, 'I': 6.02,
    'L': 5.98, 'K': 9.74, 'M': 5.74, 'F': 5.48, 'P': 6.30,
    'S': 5.68, 'T': 5.64, 'W': 5.89, 'Y': 5.66, 'V': 5.96,
    'X': 5.57, 'Z': 5.47
}

# ============================================================================
# Dataset Configurations
# ============================================================================

DATASET_CONFIG = {
    'novozymes': {
        'competition_name': 'novozymes-enzyme-stability-prediction',
        'files': ['train.csv', 'test.csv', 'train_updates.csv'],
        'structure_file': 'wildtype_structure_prediction_af2.pdb',
        'metric': 'spearman',
    },
    'cafa5': {
        'competition_name': 'cafa-5-protein-function-prediction',
        'files': ['train_terms.csv', 'test_sequences.fasta'],
        'metric': 'f_max',
    }
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_data_paths(base_dir: str = './data') -> dict:
    """Get standard data directory paths."""
    base = Path(base_dir)
    return {
        'pdb': base / 'pdb',
        'kaggle': base / 'kaggle',
        'spectral': base / 'spectral',
        'processed': base / 'processed',
    }


def ensure_directories(base_dir: str = './data'):
    """Create all required directories."""
    paths = get_data_paths(base_dir)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def encode_sequence(sequence: str) -> np.ndarray:
    """
    Encode amino acid sequence as one-hot vectors.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        One-hot encoded matrix (len(sequence), 22)
    """
    encoding = np.zeros((len(sequence), len(AMINO_ACIDS)))
    for i, aa in enumerate(sequence):
        if aa in AA_TO_IDX:
            encoding[i, AA_TO_IDX[aa]] = 1.0
        else:
            encoding[i, AA_TO_IDX['X']] = 1.0  # Unknown
    return encoding


def encode_sequence_torch(sequence: str) -> torch.Tensor:
    """PyTorch version of sequence encoding."""
    return torch.from_numpy(encode_sequence(sequence)).float()


def decode_sequence(encoding: np.ndarray) -> str:
    """
    Decode one-hot encoded sequence back to string.
    
    Args:
        encoding: One-hot matrix (len(sequence), 22)
        
    Returns:
        Amino acid sequence
    """
    indices = np.argmax(encoding, axis=1)
    return ''.join([IDX_TO_AA.get(i, 'X') for i in indices])


def compute_sequence_properties(sequence: str) -> dict:
    """
    Compute physicochemical properties of sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary of properties
    """
    properties = {
        'length': len(sequence),
        'mean_hydrophobicity': np.mean([HYDROPHOBICITY.get(aa, 0) for aa in sequence]),
        'mean_mw': np.mean([MOLECULAR_WEIGHT.get(aa, 137) for aa in sequence]),
        'charge_positive': sequence.count('R') + sequence.count('K'),
        'charge_negative': sequence.count('D') + sequence.count('E'),
        'aromatic': sequence.count('F') + sequence.count('Y') + sequence.count('W'),
        'proline': sequence.count('P'),
    }
    return properties


def normalize_spectrum(spectrum: np.ndarray, method: str = 'max') -> np.ndarray:
    """
    Normalize spectrum using various methods.
    
    Args:
        spectrum: Input spectrum
        method: 'max', 'l2', 'integral', or 'zscore'
        
    Returns:
        Normalized spectrum
    """
    if method == 'max':
        max_val = np.max(spectrum)
        return spectrum / max_val if max_val > 0 else spectrum
    
    elif method == 'l2':
        norm = np.linalg.norm(spectrum)
        return spectrum / norm if norm > 0 else spectrum
    
    elif method == 'integral':
        integral = np.sum(spectrum)
        return spectrum / integral if integral > 0 else spectrum
    
    elif method == 'zscore':
        mean = np.mean(spectrum)
        std = np.std(spectrum)
        return (spectrum - mean) / std if std > 0 else spectrum
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def batch_collate_function(batch):
    """
    Custom collate function for DataLoader with mixed data types.
    
    Handles PyG Data objects + spectra + auxiliary features.
    """
    from torch_geometric.data import Batch
    
    graphs = []
    spectra = []
    global_features = []
    labels = []
    
    for item in batch:
        graphs.append(item['graph'])
        spectra.append(item['spectrum'])
        if 'global_features' in item:
            global_features.append(item['global_features'])
        if 'label' in item:
            labels.append(item['label'])
    
    # Batch graphs
    batch_graph = Batch.from_data_list(graphs)
    
    # Stack spectra
    batch_spectra = torch.stack(spectra)
    
    # Stack other tensors
    result = {
        'graph': batch_graph,
        'spectra': batch_spectra,
    }
    
    if global_features:
        result['global_features'] = torch.stack(global_features)
    
    if labels:
        result['labels'] = torch.stack(labels)
    
    return result


class Logger:
    """Simple logger wrapper."""
    
    @staticmethod
    def setup(name: str, level: int = logging.INFO) -> logging.Logger:
        """Setup logger."""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Get available device (cuda or cpu)."""
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    else:
        logger.info("Using CPU device")
        return 'cpu'


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    seq = "MKVLVVAT"
    encoded = encode_sequence(seq)
    decoded = decode_sequence(encoded)
    print(f"Original: {seq}")
    print(f"Decoded: {decoded}")
    
    props = compute_sequence_properties(seq)
    print(f"Properties: {props}")
    
    print("Utilities test passed!")
