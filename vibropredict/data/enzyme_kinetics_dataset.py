"""
PyTorch Dataset for Enzyme Kinetics

Provides indexed access to enzyme kinetics samples including
protein sequences, VDOS spectra, substrate SMILES, and log k_cat labels.
"""

import logging
from pathlib import Path
from typing import Literal

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

    def __init__(
        self,
        csv_path: str,
        vdos_dir: str,
        n_points: int = 1000,
        split_strategy: Literal["random", "ec_holdout"] | None = None,
        split_name: str = "train",
        split_seed: int = 42,
    ):
        """
        Initialize dataset.

        Args:
            csv_path: Path to CSV with columns including uniprot_id,
                      log_kcat, substrate_smiles, and optionally
                      product_smiles and mutation.
            vdos_dir: Directory containing per-protein VDOS files
                      named {uniprot_id}_vdos.npy.
            n_points: Number of frequency points in each VDOS spectrum.
            split_strategy: Optional split strategy to apply:
                - None: Use the full CSV (backward-compatible default).
                - "random": Random train/val/test split.
                - "ec_holdout": OOD split by enzyme family (EC class).
            split_name: Which split to use ('train', 'val', or 'test').
                Only relevant when split_strategy is not None.
            split_seed: Random seed for the split (for reproducibility).
        """
        df = pd.read_csv(csv_path)

        # Apply split strategy if requested
        if split_strategy is not None:
            from vibropredict.data.splits import ECHoldoutSplit, RandomSplit

            if split_strategy == "random":
                splitter = RandomSplit(seed=split_seed)
            elif split_strategy == "ec_holdout":
                splitter = ECHoldoutSplit(seed=split_seed)
            else:
                raise ValueError(
                    f"Unknown split_strategy: {split_strategy!r}. "
                    "Use 'random', 'ec_holdout', or None."
                )

            splits = splitter.split(df)
            if split_name not in splits:
                raise ValueError(
                    f"Unknown split_name: {split_name!r}. Available: {list(splits.keys())}"
                )
            df = splits[split_name]

        self.df = df
        self.vdos_dir = Path(vdos_dir)
        self.n_points = n_points
        logger.info(
            f"EnzymeKineticsDataset: {len(self.df)} samples, "
            f"vdos_dir={vdos_dir}, n_points={n_points}"
            + (f", split={split_strategy}/{split_name}" if split_strategy else "")
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, object]:
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
        product_smiles = (
            str(row.get("product_smiles", "")) if pd.notna(row.get("product_smiles")) else ""
        )
        mutation = str(row.get("mutation", "")) if pd.notna(row.get("mutation")) else ""

        # Load VDOS spectrum
        vdos_path = self.vdos_dir / f"{uniprot_id}_vdos.npy"
        if vdos_path.exists():
            vdos = np.load(vdos_path).astype(np.float32)
            # Pad or truncate to n_points
            if len(vdos) < self.n_points:
                vdos = np.pad(vdos, (0, self.n_points - len(vdos)))
            else:
                vdos = vdos[: self.n_points]
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
