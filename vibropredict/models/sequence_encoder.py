"""
Protein Sequence Encoder for VibroPredict

Uses ProtT5 (T5EncoderModel) to produce fixed-length protein embeddings
with learned per-residue attention pooling.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ProtT5Encoder(nn.Module):
    """
    Protein sequence encoder based on ProtT5.

    Lazy-loads the pretrained T5 encoder on first forward pass to avoid
    downloading or allocating GPU memory at init time. Per-residue
    attention weights are learned to pool variable-length embeddings
    into a fixed-size representation.
    """

    def __init__(
        self,
        model_name: str = "Rostlab/prot_t5_xl_uniref50",
        output_dim: int = 1024,
    ):
        """
        Initialize ProtT5 encoder.

        Args:
            model_name: HuggingFace model identifier for ProtT5.
            output_dim: Dimension of per-residue embeddings (must match
                the pretrained model hidden size, typically 1024).
        """
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim

        # Per-residue attention for weighted pooling
        self.attention = nn.Linear(output_dim, 1)

        # Lazy-loaded components
        self._tokenizer = None
        self._encoder = None

    def _load_model(self, device: torch.device) -> None:
        """Load the pretrained tokenizer and encoder model."""
        from transformers import T5EncoderModel, T5Tokenizer  # type: ignore

        logger.info(f"Loading ProtT5 model: {self.model_name}")
        self._tokenizer = T5Tokenizer.from_pretrained(
            self.model_name, do_lower_case=False
        )
        self._encoder = T5EncoderModel.from_pretrained(self.model_name)
        self._encoder = self._encoder.to(device)
        self._encoder.eval()
        logger.info("ProtT5 model loaded successfully")

    def forward(self, sequences: list[str]) -> torch.Tensor:
        """
        Encode protein sequences into fixed-length embeddings.

        Tokenizes the input sequences (inserting spaces between residues
        as required by ProtT5), runs them through the T5 encoder, and
        applies learned attention pooling.

        Args:
            sequences: List of amino-acid sequences (e.g. ``["MKTLL..."]``).

        Returns:
            Tensor of shape ``(batch, output_dim)``.
        """
        device = self.attention.weight.device

        if self._encoder is None:
            self._load_model(device)

        # ProtT5 expects spaces between residues
        spaced = [" ".join(list(seq)) for seq in sequences]

        tokens = self._tokenizer(
            spaced,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            encoder_output = self._encoder(**tokens)

        # (batch, seq_len, 1024)
        embeddings = encoder_output.last_hidden_state

        # Attention pooling
        attn_weights = torch.softmax(self.attention(embeddings), dim=1)  # (batch, seq_len, 1)
        pooled = torch.sum(attn_weights * embeddings, dim=1)  # (batch, 1024)

        return pooled
