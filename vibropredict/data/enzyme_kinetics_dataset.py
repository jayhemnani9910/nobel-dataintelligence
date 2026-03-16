"""
PyTorch Dataset for Enzyme Kinetics

Provides indexed access to enzyme kinetics samples including
protein sequences, VDOS spectra, substrate SMILES, and log k_cat labels.
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class EnzymeKineticsDataset(Dataset):
    """
    Dataset for enzyme kinetics prediction.

    Each sample contains a protein sequence, optional vibrational
    density of states (VDOS), substrate/product SMILES, mutation
    annotation, and the log-transformed catalytic rate.
    """

    def __init__(self, csv_path: str, vdos_dir: str, n_points: int = 1000):
        """
        Initialize dataset.

        Args:
            csv_path: Path to CSV with columns including uniprot_id,
                      log_kcat, substrate_smiles, and optionally
                      product_smiles and mutation.
            vdos_dir: Directory containing per-protein VDOS files
                      named {uniprot_id}_vdos.npy.
            n_points: Number of frequency points in each VDOS spectrum.
        """
        self.df = pd.read_csv(csv_path)
        self.vdos_dir = Path(vdos_dir)
        self.n_points = n_points
        logger.info(
            f"EnzymeKineticsDataset: {len(self.df)} samples, "
            f"vdos_dir={vdos_dir}, n_points={n_points}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """
        Load a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys: sequence, log_kcat, substrate_smiles,
            product_smiles, vdos, mutation, has_vdos.
        """
        row = self.df.iloc[idx]

        uniprot_id = str(row["uniprot_id"])
        sequence = str(row.get("sequence", ""))
        log_kcat = float(row["log_kcat"])
        substrate_smiles = str(row.get("substrate_smiles", ""))
        product_smiles = str(row.get("product_smiles", "")) if pd.notna(row.get("product_smiles")) else ""
        mutation = str(row.get("mutation", "")) if pd.notna(row.get("mutation")) else ""

        # Load VDOS spectrum
        vdos_path = self.vdos_dir / f"{uniprot_id}_vdos.npy"
        if vdos_path.exists():
            vdos = np.load(vdos_path).astype(np.float32)
            # Pad or truncate to n_points
            if len(vdos) < self.n_points:
                vdos = np.pad(vdos, (0, self.n_points - len(vdos)))
            else:
                vdos = vdos[:self.n_points]
            has_vdos = True
        else:
            vdos = np.zeros(self.n_points, dtype=np.float32)
            has_vdos = False

        return {
            "sequence": sequence,
            "log_kcat": torch.tensor(log_kcat, dtype=torch.float32),
            "substrate_smiles": substrate_smiles,
            "product_smiles": product_smiles,
            "vdos": torch.tensor(vdos, dtype=torch.float32).unsqueeze(0),
            "mutation": mutation,
            "has_vdos": has_vdos,
        }
