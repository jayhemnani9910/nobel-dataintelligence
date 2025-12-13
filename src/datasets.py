"""
PyTorch Dataset Classes for Quantum Data Decoder

Implements efficient data loading for structural and spectral data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import prody as pr

logger = logging.getLogger(__name__)


class ProteinStructureDataset(Dataset):
    """
    Dataset for protein structures with spectral data.
    
    Loads PDB structures and corresponding vibrational spectra
    for training multimodal models.
    """
    
    def __init__(self, pdb_files: List[str],
                 spectra_dir: str,
                 metadata_df: Optional[pd.DataFrame] = None,
                 precompute_graphs: bool = False,
                 cache_spectra: bool = True):
        """
        Initialize dataset.
        
        Args:
            pdb_files: List of PDB file paths
            spectra_dir: Directory containing spectral data
            metadata_df: Optional DataFrame with labels/metadata
            precompute_graphs: Whether to precompute graphs
            cache_spectra: Cache loaded spectra in memory
        """
        self.pdb_files = pdb_files
        self.spectra_dir = Path(spectra_dir)
        self.metadata_df = metadata_df
        self.precompute_graphs = precompute_graphs
        
        # Spectral cache
        self.spectra_cache = {} if cache_spectra else None
        
        logger.info(f"Dataset initialized with {len(pdb_files)} structures")
    
    def __len__(self) -> int:
        return len(self.pdb_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """Load a single sample."""
        pdb_file = self.pdb_files[idx]
        pdb_id = Path(pdb_file).stem.lower()
        
        # Load structure
        try:
            structure = pr.parsePDB(pdb_file)
        except Exception as e:
            logger.warning(f"Failed to load {pdb_file}: {e}")
            return None
        
        # Load spectrum
        spectrum_file = self.spectra_dir / f"{pdb_id}_spectrum.npy"
        if spectrum_file.exists():
            spectrum = np.load(spectrum_file)
        else:
            logger.warning(f"Spectrum not found for {pdb_id}")
            spectrum = np.zeros(1000)
        
        # Load metadata if available
        label = None
        global_features = None
        if self.metadata_df is not None:
            row = self.metadata_df[self.metadata_df['pdb_id'] == pdb_id]
            if not row.empty:
                if 'label' in row.columns:
                    label = float(row['label'].values[0])
                if 'entropy' in row.columns and 'sasa' in row.columns and 'zpe' in row.columns:
                    global_features = np.array([
                        row['entropy'].values[0],
                        row['sasa'].values[0],
                        row['zpe'].values[0]
                    ], dtype=np.float32)
        
        # Construct graph (if not precomputed)
        from .models.gnn import GraphConstruction
        
        ca = structure.select('ca')
        if ca is None:
            logger.warning(f"No C-alpha atoms in {pdb_file}")
            return None
        
        coords = torch.tensor(ca.getCoords(), dtype=torch.float32)
        sequence = pr.getSequence(ca)
        features = GraphConstruction.construct_residue_features(sequence)
        
        graph = GraphConstruction.construct_ca_graph(
            coords, features, distance_cutoff=10.0, edge_features=True
        )
        
        # Prepare output
        sample = {
            'pdb_id': pdb_id,
            'graph': graph,
            'spectra': torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0),
        }
        
        if label is not None:
            sample['labels'] = torch.tensor(label, dtype=torch.float32)
        
        if global_features is not None:
            sample['global_features'] = torch.tensor(global_features, dtype=torch.float32)
        
        return sample


class NovozymesDataset(Dataset):
    """
    Specialized dataset for Novozymes enzyme stability prediction.
    
    Handles mutation data and creates ranking pairs.
    """
    
    def __init__(self, csv_file: str,
                 structure_file: str,
                 spectra_dir: str,
                 include_updates: bool = True):
        """
        Initialize Novozymes dataset.
        
        Args:
            csv_file: Path to train.csv
            structure_file: Path to wildtype structure PDB
            spectra_dir: Directory with precomputed spectra
            include_updates: Whether to apply train_updates.csv corrections
        """
        # Load CSV
        self.df = pd.read_csv(csv_file)
        
        # Apply updates if available
        if include_updates:
            updates_file = Path(csv_file).parent / 'train_updates.csv'
            if updates_file.exists():
                updates = pd.read_csv(updates_file)
                logger.info(f"Applying {len(updates)} updates to training data")
                # Update rows
                for idx, row in updates.iterrows():
                    mask = self.df['seq_id'] == row['seq_id']
                    self.df.loc[mask] = row
        
        # Load structure
        self.structure = pr.parsePDB(structure_file)
        self.spectra_dir = Path(spectra_dir)
        
        logger.info(f"Novozymes dataset: {len(self.df)} mutations")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        """Load mutation data."""
        row = self.df.iloc[idx]
        
        # Extract info
        seq_id = row['seq_id']
        sequence = row['protein_sequence']
        tm = float(row['tm'])
        pH = float(row['pH'])
        
        # Get wildtype spectrum (or compute delta)
        wt_spectrum_file = self.spectra_dir / 'wt_spectrum.npy'
        if wt_spectrum_file.exists():
            spectrum = np.load(wt_spectrum_file)
        else:
            spectrum = np.zeros(1000)
        
        # Construct graph from sequence
        from .models.gnn import GraphConstruction
        
        features = GraphConstruction.construct_residue_features(sequence)
        # Use dummy coordinates (or load from structure if available)
        coords = torch.randn(len(sequence), 3) * 10
        
        graph = GraphConstruction.construct_ca_graph(
            coords, features, distance_cutoff=10.0
        )
        
        # Global features: [entropy, pH, dummy_sasa]
        global_features = torch.tensor([0.0, pH, 0.0], dtype=torch.float32)
        
        sample = {
            'seq_id': seq_id,
            'graph': graph,
            'spectra': torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0),
            'labels': torch.tensor(tm, dtype=torch.float32),
            'global_features': global_features,
            'sequence': sequence,
        }
        
        return sample


class CAFA5Dataset(Dataset):
    """
    Specialized dataset for CAFA 5 protein function prediction.
    
    Handles multi-label GO term prediction.
    """
    
    def __init__(self, sequences_fasta: str,
                 terms_csv: str,
                 spectra_dir: str,
                 structure_dir: str,
                 go_terms_list: Optional[List[str]] = None):
        """
        Initialize CAFA 5 dataset.
        
        Args:
            sequences_fasta: Path to FASTA file with sequences
            terms_csv: Path to train_terms.csv with GO labels
            spectra_dir: Directory with precomputed spectra
            structure_dir: Directory with AlphaFold structures
            go_terms_list: List of GO term IDs to use
        """
        # Load sequences
        self.sequences = {}
        self._load_fasta(sequences_fasta)
        
        # Load GO term labels
        self.terms_df = pd.read_csv(terms_csv)
        
        # Load GO terms list
        if go_terms_list is None:
            self.go_terms = sorted(self.terms_df['go_id'].unique())
        else:
            self.go_terms = go_terms_list
        
        self.go_to_idx = {go: i for i, go in enumerate(self.go_terms)}
        
        self.spectra_dir = Path(spectra_dir)
        self.structure_dir = Path(structure_dir)
        
        logger.info(f"CAFA5 dataset: {len(self.sequences)} proteins, {len(self.go_terms)} GO terms")
    
    def _load_fasta(self, fasta_file: str):
        """Load FASTA sequences."""
        with open(fasta_file, 'r') as f:
            current_id = None
            current_seq = []
            for line in f:
                if line.startswith('>'):
                    if current_id:
                        self.sequences[current_id] = ''.join(current_seq)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_id:
                self.sequences[current_id] = ''.join(current_seq)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """Load protein and GO term labels."""
        protein_id = list(self.sequences.keys())[idx]
        sequence = self.sequences[protein_id]
        
        # Load spectrum
        spectrum_file = self.spectra_dir / f"{protein_id}_spectrum.npy"
        if spectrum_file.exists():
            spectrum = np.load(spectrum_file)
        else:
            spectrum = np.zeros(1000)
        
        # Construct graph
        from .models.gnn import GraphConstruction
        
        features = GraphConstruction.construct_residue_features(sequence)
        coords = torch.randn(len(sequence), 3) * 10  # Dummy coords
        
        graph = GraphConstruction.construct_ca_graph(
            coords, features, distance_cutoff=10.0
        )
        
        # Get GO labels
        go_labels = self.terms_df[self.terms_df['target_id'] == protein_id]['go_id'].values
        label_vector = np.zeros(len(self.go_terms), dtype=np.float32)
        for go in go_labels:
            if go in self.go_to_idx:
                label_vector[self.go_to_idx[go]] = 1.0
        
        sample = {
            'protein_id': protein_id,
            'graph': graph,
            'spectra': torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0),
            'labels': torch.tensor(label_vector, dtype=torch.float32),
            'sequence': sequence,
        }
        
        return sample


def create_dataloaders(train_dataset: Dataset,
                      val_dataset: Dataset,
                      test_dataset: Optional[Dataset] = None,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for train/val/test splits.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from .utils import batch_collate_function
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=batch_collate_function,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=batch_collate_function,
        pin_memory=True,
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=batch_collate_function,
            pin_memory=True,
        )
    
    return train_loader, val_loader, test_loader


def main():
    """Test dataset loading."""
    logger.info("Dataset module loaded successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
