"""
VibroPredictHybrid - End-to-End Multimodal Model

Combines protein sequence encoding (ProtT5), vibrational spectral
encoding (SpectralCNN), and chemical encoding (ChemBERTa + DRFP)
with gated tri-modal fusion for enzyme k_cat prediction.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from src.models.cnn import SpectralCNN
from vibropredict.models.chemical_encoder import ChemicalEncoder
from vibropredict.models.fusion import TriModalFusion
from vibropredict.models.sequence_encoder import ProtT5Encoder

logger = logging.getLogger(__name__)


class VibroPredictHybrid(nn.Module):
    """
    Multimodal model for enzyme kinetics prediction.

    Encodes protein sequences, vibrational density of states (VDOS),
    and substrate/product SMILES into a shared representation via
    gated fusion, then predicts log(k_cat) through a regression head.

    Supports spectral dropout (``drop_spectral=True``) to train a
    model that can gracefully degrade when VDOS is unavailable.
    """

    def __init__(
        self,
        seq_model: str = "Rostlab/prot_t5_xl_uniref50",
        chem_model: str = "seyonec/ChemBERTa-zinc-base-v1",
        spec_dim: int = 128,
        seq_dim: int = 1024,
        chem_dim: int = 1024,
        fusion_dim: int = 512,
        dropout: float = 0.2,
    ):
        """
        Initialize VibroPredictHybrid.

        Args:
            seq_model: HuggingFace model name for ProtT5 encoder.
            chem_model: HuggingFace model name for ChemBERTa encoder.
            spec_dim: Output dimension of the spectral CNN encoder.
            seq_dim: Output dimension of the sequence encoder.
            chem_dim: Output dimension of the chemical encoder
                (each branch produces ``chem_dim // 2``; concatenated).
            fusion_dim: Output dimension of the tri-modal fusion layer.
            dropout: Dropout probability for the regression head.
        """
        super().__init__()

        self.seq_encoder = ProtT5Encoder(
            model_name=seq_model, output_dim=seq_dim
        )
        self.spec_encoder = SpectralCNN(
            input_channels=1, output_dim=spec_dim
        )
        self.chem_encoder = ChemicalEncoder(
            smiles_model=chem_model,
            fp_dim=chem_dim // 2,
            output_dim=chem_dim // 2,
        )
        self.fusion = TriModalFusion(
            seq_dim=seq_dim,
            spec_dim=spec_dim,
            chem_dim=chem_dim,
            output_dim=fusion_dim,
        )

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        logger.info(
            f"Initialized VibroPredictHybrid: "
            f"seq_dim={seq_dim}, spec_dim={spec_dim}, chem_dim={chem_dim}, "
            f"fusion_dim={fusion_dim}"
        )

    def forward(
        self,
        sequences: list[str],
        vdos: torch.Tensor,
        substrate_smiles: list[str],
        product_smiles: list[str] = None,
        drop_spectral: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the full multimodal pipeline.

        Args:
            sequences: List of amino-acid sequences.
            vdos: VDOS tensor of shape ``(batch, 1, n_points)``.
            substrate_smiles: List of substrate SMILES strings.
            product_smiles: Optional list of product SMILES strings.
            drop_spectral: If ``True``, zero out the spectral embedding
                (useful for ablation or when VDOS is unavailable).

        Returns:
            Tuple of:
                - logkcat: Predicted log(k_cat) of shape ``(batch,)``.
                - gates: Fusion gate weights of shape ``(batch, 3)``.
        """
        h_seq = self.seq_encoder(sequences)         # (batch, seq_dim)
        h_spec = self.spec_encoder(vdos)             # (batch, spec_dim)

        if drop_spectral:
            h_spec = torch.zeros_like(h_spec)

        h_chem = self.chem_encoder(substrate_smiles, product_smiles)  # (batch, chem_dim)

        fused, gates = self.fusion(h_seq, h_spec, h_chem)  # (batch, fusion_dim), (batch, 3)
        logkcat = self.regressor(fused).squeeze(-1)          # (batch,)

        return logkcat, gates
