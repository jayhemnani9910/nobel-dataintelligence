"""
ESMFold Structure Predictor

Wraps the ESMFold protein structure prediction model from the
Hugging Face transformers library with lazy loading and batch support.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ESMFoldPredictor:
    """
    Predict 3D protein structures using ESMFold.

    The model is loaded lazily on first prediction to avoid heavy
    imports and GPU allocation at module import time.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize predictor.

        Args:
            device: PyTorch device string ('cpu' or 'cuda').
        """
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load ESMForProteinFolding and its tokenizer from transformers."""
        import torch  # noqa: F811
        from transformers import AutoTokenizer, EsmForProteinFolding  # type: ignore

        logger.info("Loading ESMFold model (this may take a moment)...")
        self._tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self._model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        self._model = self._model.to(self.device)
        self._model.eval()
        logger.info(f"ESMFold model loaded on {self.device}")

    def predict_structure(self, sequence: str) -> str:
        """
        Predict a protein structure from its amino acid sequence.

        Args:
            sequence: Single-letter amino acid sequence.

        Returns:
            Predicted structure as a PDB-format string.
        """
        import torch  # noqa: F811

        if self._model is None:
            self._load_model()

        inputs = self._tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Convert output to PDB string
        pdb_string = self._model.output_to_pdb(outputs)[0]
        return pdb_string

    def predict_batch(
        self, sequences: List[str], output_dir: str
    ) -> List[str]:
        """
        Predict structures for multiple sequences and save to disk.

        Args:
            sequences: List of amino acid sequences.
            output_dir: Directory to save PDB files (named {idx}.pdb).

        Returns:
            List of paths to saved PDB files.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        paths: List[str] = []

        for idx, seq in enumerate(sequences):
            pdb_string = self.predict_structure(seq)
            pdb_file = out_path / f"{idx}.pdb"
            pdb_file.write_text(pdb_string)
            paths.append(str(pdb_file))
            logger.info(f"Predicted structure {idx + 1}/{len(sequences)} -> {pdb_file}")

        return paths

    def validate_plddt(self, pdb_string: str, threshold: float = 70.0) -> bool:
        """
        Validate predicted structure quality via mean pLDDT.

        pLDDT scores are stored in the B-factor column of PDB files
        produced by ESMFold.

        Args:
            pdb_string: PDB-format string.
            threshold: Minimum acceptable mean pLDDT.

        Returns:
            True if mean pLDDT exceeds the threshold.
        """
        plddt_values: List[float] = []

        for line in pdb_string.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    bfactor = float(line[60:66].strip())
                    plddt_values.append(bfactor)
                except (ValueError, IndexError):
                    continue

        if not plddt_values:
            logger.warning("No CA atoms found in PDB string for pLDDT validation")
            return False

        mean_plddt = float(np.mean(plddt_values))
        passed = mean_plddt > threshold
        logger.info(f"Mean pLDDT = {mean_plddt:.1f} (threshold={threshold}, passed={passed})")
        return passed
