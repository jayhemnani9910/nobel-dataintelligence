"""
Tri-Modal Fusion Module for VibroPredict

Implements gated fusion of sequence, spectral, and chemical embeddings
using a learned soft-gating mechanism.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TriModalFusion(nn.Module):
    """
    Gated fusion of three modality embeddings.

    A lightweight gating network learns soft attention weights over
    the sequence, spectral, and chemical branches. Each branch is
    first projected to a common dimensionality, then combined via
    the learned gates.
    """

    def __init__(
        self,
        seq_dim: int = 1024,
        spec_dim: int = 128,
        chem_dim: int = 1024,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize tri-modal fusion.

        Args:
            seq_dim: Dimension of sequence embeddings.
            spec_dim: Dimension of spectral embeddings.
            chem_dim: Dimension of chemical embeddings.
            output_dim: Dimension of fused output.
            dropout: Dropout probability applied to fused representation.
        """
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Linear(seq_dim + spec_dim + chem_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        self.proj_seq = nn.Linear(seq_dim, output_dim)
        self.proj_spec = nn.Linear(spec_dim, output_dim)
        self.proj_chem = nn.Linear(chem_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_seq: torch.Tensor,
        h_spec: torch.Tensor,
        h_chem: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse three modality embeddings with learned gating.

        Args:
            h_seq: Sequence embedding of shape ``(batch, seq_dim)``.
            h_spec: Spectral embedding of shape ``(batch, spec_dim)``.
            h_chem: Chemical embedding of shape ``(batch, chem_dim)``.

        Returns:
            Tuple of:
                - fused: Fused representation of shape ``(batch, output_dim)``.
                - gates: Gate weights of shape ``(batch, 3)``.
        """
        concat = torch.cat([h_seq, h_spec, h_chem], dim=-1)
        gates = torch.softmax(self.gate_net(concat), dim=-1)  # (batch, 3)

        proj_seq = self.proj_seq(h_seq)    # (batch, output_dim)
        proj_spec = self.proj_spec(h_spec)  # (batch, output_dim)
        proj_chem = self.proj_chem(h_chem)  # (batch, output_dim)

        fused = (
            gates[:, 0:1] * proj_seq
            + gates[:, 1:2] * proj_spec
            + gates[:, 2:3] * proj_chem
        )

        return self.dropout(fused), gates
