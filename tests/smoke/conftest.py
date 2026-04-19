"""Shared fixtures for smoke tests."""

from __future__ import annotations

import importlib.util

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None

if _HAS_TORCH:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader


def _make_synthetic_kinhub_data(n: int = 100):
    """Generate synthetic KinHub-shaped data for smoke testing.

    Returns a list of sample dictionaries matching the
    EnzymeKineticsDataset / DataLoader output format.

    Does NOT download real data or hit any external API.
    """
    rng = np.random.RandomState(42)

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    smiles_pool = [
        "CC(=O)O",  # acetic acid
        "OC(=O)CC(O)(CC(=O)O)C(=O)O",  # citric acid
        "C(=O)(O)O",  # formic acid
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  # glucose
        "CC(=O)SCC",  # acetyl-CoA analog
    ]

    sequences = []
    for _ in range(n):
        length = rng.randint(50, 200)
        seq = "".join(rng.choice(list(amino_acids), size=length))
        sequences.append(seq)

    data = []
    for i in range(n):
        data.append(
            {
                "sequences": [sequences[i]],
                "vdos": torch.randn(1, 1000),
                "substrate_smiles": [smiles_pool[i % len(smiles_pool)]],
                "product_smiles": None,
                "log_kcat": torch.tensor(rng.uniform(-2, 4), dtype=torch.float32),
            }
        )

    return data


def _collate_fn(batch):
    """Collate function matching the trainer's expected batch format."""
    return {
        "sequences": [b["sequences"][0] for b in batch],
        "vdos": torch.stack([b["vdos"] for b in batch]),
        "substrate_smiles": [b["substrate_smiles"][0] for b in batch],
        "product_smiles": None,
        "log_kcat": torch.stack([b["log_kcat"] for b in batch]),
    }


class SyntheticDataset:
    """Simple list-backed dataset for smoke tests."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def synthetic_loader():
    """Create train and val DataLoaders from synthetic KinHub-shaped data."""
    if not _HAS_TORCH:
        pytest.skip("torch not installed")

    data = _make_synthetic_kinhub_data(n=100)
    train_data = data[:80]
    val_data = data[80:]

    train_loader = DataLoader(
        SyntheticDataset(train_data),
        batch_size=16,
        shuffle=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        SyntheticDataset(val_data),
        batch_size=16,
        shuffle=False,
        collate_fn=_collate_fn,
    )
    return train_loader, val_loader


@pytest.fixture
def dummy_model():
    """Create a minimal model matching VibroPredictHybrid's interface."""
    if not _HAS_TORCH:
        pytest.skip("torch not installed")

    class SmokeDummyModel(nn.Module):
        """Lightweight stand-in for VibroPredictHybrid.

        Accepts the same call signature but uses a trivial linear layer
        instead of pretrained encoders, so no downloads are needed.
        """

        def __init__(self):
            super().__init__()
            self.spec_proj = nn.Linear(1000, 32)
            self.head = nn.Linear(32, 1)

        def forward(self, sequences, vdos, substrate_smiles, product_smiles, drop_spectral):
            batch_size = vdos.shape[0]
            # Simple spectral branch
            x = vdos.squeeze(1)  # (B, 1000)
            if drop_spectral:
                x = torch.zeros_like(x)
            h = torch.relu(self.spec_proj(x))  # (B, 32)
            logkcat = self.head(h).squeeze(-1)  # (B,)
            gates = torch.ones(batch_size, 3, device=vdos.device) / 3.0
            return logkcat, gates

    return SmokeDummyModel()
