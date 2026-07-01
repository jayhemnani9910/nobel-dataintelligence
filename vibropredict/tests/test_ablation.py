"""
Regression tests for the ablation runner and modality-drop flags.

Previously, ``run_ablation``'s ``no_sequence`` and ``no_chemical`` variants
re-ran the *full* model instead of zeroing those modalities, so 3 of 4 ablation
rows were identical and the table was meaningless. These tests assert that:

  1. VibroPredictHybrid.forward honours drop_sequence / drop_chemical /
     drop_spectral by zeroing the corresponding embedding, and
  2. the four ablation variants produce genuinely different predictions.

Encoders are replaced with tiny deterministic stubs so no ProtT5 / ChemBERTa
weights are downloaded at test time; the real fusion, regressor, and drop
logic are exercised.
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from vibropredict.models.vibropredict_hybrid import VibroPredictHybrid


class _StubSeq(nn.Module):
    """Deterministic, non-zero sequence embedding depending on batch size."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, sequences):
        n = len(sequences)
        return self.w * torch.arange(1.0, n + 1).unsqueeze(1).repeat(1, self.dim)


class _StubChem(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, substrate_smiles, product_smiles=None):
        n = len(substrate_smiles)
        return self.w * (torch.arange(1.0, n + 1).unsqueeze(1).repeat(1, self.dim) * 2.0)


def _build_stubbed_model():
    """VibroPredictHybrid with encoders swapped for deterministic stubs."""
    model = VibroPredictHybrid(
        spec_dim=8, seq_dim=16, chem_dim=16, fusion_dim=12, dropout=0.0
    )
    model.seq_encoder = _StubSeq(16)
    # spec_encoder is a real SpectralCNN; feed it a real VDOS tensor below.
    model.chem_encoder = _StubChem(16)  # chem_dim total = 16
    model.eval()
    return model


def _dummy_inputs(batch=4, npts=1000):
    torch.manual_seed(0)
    sequences = ["MKT" * (i + 1) for i in range(batch)]
    vdos = torch.rand(batch, 1, npts)
    smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CN"][:batch]
    return sequences, vdos, smiles


class TestModalityDropFlags(unittest.TestCase):
    def test_drop_flags_change_output(self):
        model = _build_stubbed_model()
        seqs, vdos, smiles = _dummy_inputs()
        with torch.no_grad():
            full, _ = model(seqs, vdos, smiles)
            no_seq, _ = model(seqs, vdos, smiles, drop_sequence=True)
            no_chem, _ = model(seqs, vdos, smiles, drop_chemical=True)
            no_spec, _ = model(seqs, vdos, smiles, drop_spectral=True)

        # Each ablation must move the prediction away from the full model.
        self.assertFalse(torch.allclose(full, no_seq), "drop_sequence had no effect")
        self.assertFalse(torch.allclose(full, no_chem), "drop_chemical had no effect")
        self.assertFalse(torch.allclose(full, no_spec), "drop_spectral had no effect")

    def test_ablation_variants_are_distinct(self):
        """The core regression: no two ablation variants collapse to identical
        predictions (which the old re-run-full-model logic produced)."""
        model = _build_stubbed_model()
        seqs, vdos, smiles = _dummy_inputs()
        outs = {}
        with torch.no_grad():
            outs["full"], _ = model(seqs, vdos, smiles)
            outs["no_spectral"], _ = model(seqs, vdos, smiles, drop_spectral=True)
            outs["no_sequence"], _ = model(seqs, vdos, smiles, drop_sequence=True)
            outs["no_chemical"], _ = model(seqs, vdos, smiles, drop_chemical=True)

        names = list(outs)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                self.assertFalse(
                    torch.allclose(outs[names[i]], outs[names[j]]),
                    f"{names[i]} and {names[j]} produced identical predictions",
                )


if __name__ == "__main__":
    unittest.main()
