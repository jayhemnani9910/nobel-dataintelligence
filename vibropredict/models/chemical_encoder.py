"""
Chemical Encoder for VibroPredict

Encodes substrate/product SMILES using ChemBERTa embeddings and
differential reaction fingerprints (DRFP) via Morgan fingerprints.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ChemicalEncoder(nn.Module):
    """
    Dual-branch chemical encoder combining SMILES language model
    embeddings with differential reaction fingerprints (DRFP).

    The SMILES branch uses a pretrained ChemBERTa model with mean
    pooling, projected to ``output_dim``. The DRFP branch computes
    Morgan fingerprints (optionally XOR-ed between substrate and
    product) and projects to ``output_dim``. The two branches are
    concatenated, yielding an output of ``output_dim * 2``.
    """

    def __init__(
        self,
        smiles_model: str = "seyonec/ChemBERTa-zinc-base-v1",
        fp_dim: int = 512,
        output_dim: int = 512,
    ):
        """
        Initialize chemical encoder.

        Args:
            smiles_model: HuggingFace model identifier for ChemBERTa.
            fp_dim: Length of Morgan fingerprint bit vector.
            output_dim: Output dimension for each branch (final output
                is ``output_dim * 2`` after concatenation).
        """
        super().__init__()
        self.smiles_model_name = smiles_model
        self.fp_dim = fp_dim
        self.output_dim = output_dim

        # Projection layers
        self.smiles_proj = nn.Linear(768, output_dim)
        self.fp_proj = nn.Linear(fp_dim, output_dim)

        # Lazy-loaded components
        self._tokenizer = None
        self._smiles_encoder = None

    def _load_model(self, device: torch.device) -> None:
        """Load the pretrained ChemBERTa tokenizer and model."""
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        logger.info(f"Loading ChemBERTa model: {self.smiles_model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.smiles_model_name)
        self._smiles_encoder = AutoModel.from_pretrained(self.smiles_model_name)
        self._smiles_encoder = self._smiles_encoder.to(device)
        self._smiles_encoder.eval()
        logger.info("ChemBERTa model loaded successfully")

    def _compute_drfp(
        self, substrate: str, product: str = None
    ) -> torch.Tensor:
        """
        Compute a differential reaction fingerprint.

        Uses RDKit Morgan fingerprints. If *product* is provided, the
        fingerprint is the bitwise XOR of substrate and product FPs,
        capturing the chemical transformation. Otherwise returns the
        substrate fingerprint only.

        Args:
            substrate: Substrate SMILES string.
            product: Optional product SMILES string.

        Returns:
            Tensor of shape ``(fp_dim,)`` as float32.
        """
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore

        mol_sub = Chem.MolFromSmiles(substrate)
        if mol_sub is None:
            logger.warning(f"Invalid substrate SMILES: {substrate}")
            return torch.zeros(self.fp_dim)

        fp_sub = AllChem.GetMorganFingerprintAsBitVect(
            mol_sub, radius=2, nBits=self.fp_dim
        )
        arr_sub = torch.zeros(self.fp_dim)
        for bit in fp_sub.GetOnBits():
            arr_sub[bit] = 1.0

        if product is None:
            return arr_sub

        mol_prod = Chem.MolFromSmiles(product)
        if mol_prod is None:
            logger.warning(f"Invalid product SMILES: {product}")
            return arr_sub

        fp_prod = AllChem.GetMorganFingerprintAsBitVect(
            mol_prod, radius=2, nBits=self.fp_dim
        )
        arr_prod = torch.zeros(self.fp_dim)
        for bit in fp_prod.GetOnBits():
            arr_prod[bit] = 1.0

        # XOR: bits present in one but not both
        drfp = torch.abs(arr_sub - arr_prod)
        return drfp

    def forward(
        self,
        substrate_smiles: list[str],
        product_smiles: list[str] = None,
    ) -> torch.Tensor:
        """
        Encode chemical inputs via dual SMILES + DRFP branches.

        Args:
            substrate_smiles: List of substrate SMILES strings.
            product_smiles: Optional list of product SMILES strings
                (same length as *substrate_smiles*).

        Returns:
            Tensor of shape ``(batch, output_dim * 2)``.
        """
        device = self.smiles_proj.weight.device

        if self._smiles_encoder is None:
            self._load_model(device)

        # --- SMILES branch ---
        tokens = self._tokenizer(
            substrate_smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            smiles_output = self._smiles_encoder(**tokens)

        # Mean pooling over token dimension → (batch, 768)
        hidden = smiles_output.last_hidden_state
        attention_mask = tokens["attention_mask"].unsqueeze(-1).float()
        summed = torch.sum(hidden * attention_mask, dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        smiles_emb = summed / counts

        smiles_proj = self.smiles_proj(smiles_emb)  # (batch, output_dim)

        # --- DRFP branch ---
        fp_list = []
        for i, sub in enumerate(substrate_smiles):
            prod = product_smiles[i] if product_smiles is not None else None
            fp_list.append(self._compute_drfp(sub, prod))

        fp_batch = torch.stack(fp_list).to(device)  # (batch, fp_dim)
        fp_proj = self.fp_proj(fp_batch)  # (batch, output_dim)

        # Concatenate both branches
        combined = torch.cat([smiles_proj, fp_proj], dim=-1)  # (batch, output_dim * 2)
        return combined
