"""
Inference utilities for Quantum Data Decoder.

Provides simple prediction APIs for both Novozymes stability prediction
and VibroPredict enzyme kinetics prediction using saved checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _generate_vdos_for_sequence(sequence: str, n_points: int = 1000) -> np.ndarray:
    """Generate a VDOS spectrum for a protein sequence using pseudo-coordinates.

    For real predictions, pre-computed VDOS from actual PDB structures is preferred.
    This fallback uses deterministic pseudo-coordinates when no structure is available.
    """
    from .spectral_generation import SpectralGenerator

    # Deterministic pseudo-coordinates (linear chain)
    n_residues = len(sequence)
    coords = np.stack([
        np.arange(n_residues, dtype=np.float32) * 3.8,  # ~3.8 Å per residue
        np.zeros(n_residues, dtype=np.float32),
        np.zeros(n_residues, dtype=np.float32),
    ], axis=1)

    # Simple distance-based frequency estimation
    # (Rough approximation — real NMA requires ProDy and a structure)
    frequencies = np.sqrt(np.arange(1, min(n_residues, 100) + 1, dtype=np.float64)) * 15.0

    sg = SpectralGenerator(freq_min=0, freq_max=500, n_points=n_points)
    vdos = sg.generate_dos(frequencies, broadening=5.0)
    return vdos


def predict_stability(
    sequence: str,
    pH: float = 7.0,
    checkpoint_path: str = "checkpoints/novozymes_best.pt",
    device: str = "cpu",
) -> dict:
    """Predict melting temperature (Tm) for a protein mutation.

    Args:
        sequence: Amino acid sequence of the mutant protein.
        pH: pH value for the assay condition.
        checkpoint_path: Path to a trained VibroStructuralModel checkpoint.
        device: Device for inference ('cpu' or 'cuda').

    Returns:
        Dictionary with 'predicted_tm' (float) and 'sequence_length' (int).
    """
    from .models.multimodal import VibroStructuralModel
    from .models.gnn import GraphConstruction

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train a model first using the Colab notebook or CLI."
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Reconstruct model
    model = VibroStructuralModel(
        latent_dim=128,
        gnn_input_dim=24,
        fusion_type="bilinear",
        dropout=0.0,  # No dropout at inference
        num_go_terms=100,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Build graph from sequence
    features = GraphConstruction.construct_residue_features(sequence)
    coords = torch.stack([
        torch.arange(len(sequence), dtype=torch.float32) * 3.8,
        torch.zeros(len(sequence)),
        torch.zeros(len(sequence)),
    ], dim=1)
    graph = GraphConstruction.construct_ca_graph(coords, features, distance_cutoff=10.0)
    graph.batch = torch.zeros(len(sequence), dtype=torch.long)

    # Generate VDOS
    vdos = _generate_vdos_for_sequence(sequence)
    spectra = torch.tensor(vdos, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Global features
    global_features = torch.tensor([[0.0, pH, 0.0]], dtype=torch.float32)

    # Predict
    with torch.no_grad():
        graph = graph.to(device)
        spectra = spectra.to(device)
        global_features = global_features.to(device)
        output = model(graph, spectra, global_features=global_features, task="novozymes")

    predicted_tm = output.squeeze().item()

    return {
        "predicted_tm": round(predicted_tm, 2),
        "sequence_length": len(sequence),
    }


def predict_kcat(
    sequence: str,
    substrate_smiles: str,
    product_smiles: Optional[str] = None,
    checkpoint_path: str = "checkpoints/vibropredict_best.pt",
    device: str = "cpu",
) -> dict:
    """Predict catalytic turnover (k_cat) for an enzyme-substrate pair.

    Args:
        sequence: Amino acid sequence of the enzyme.
        substrate_smiles: SMILES string for the substrate.
        product_smiles: Optional SMILES string for the product.
        checkpoint_path: Path to a trained VibroPredictHybrid checkpoint.
        device: Device for inference ('cpu' or 'cuda').

    Returns:
        Dictionary with 'predicted_log_kcat', 'predicted_kcat', and 'gate_weights'.
    """
    from vibropredict.models.vibropredict_hybrid import VibroPredictHybrid

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train a model first using the Colab notebook."
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Reconstruct model
    model = VibroPredictHybrid(fusion_dim=512, dropout=0.0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Generate VDOS
    vdos = _generate_vdos_for_sequence(sequence)
    vdos_tensor = torch.tensor(vdos, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Predict
    with torch.no_grad():
        vdos_tensor = vdos_tensor.to(device)
        logkcat, gates = model(
            sequences=[sequence],
            vdos=vdos_tensor,
            substrate_smiles=[substrate_smiles],
            product_smiles=[product_smiles] if product_smiles else None,
        )

    log_kcat = logkcat.squeeze().item()
    gate_weights = gates.squeeze().cpu().numpy()

    return {
        "predicted_log_kcat": round(log_kcat, 3),
        "predicted_kcat": round(10 ** log_kcat, 2),
        "gate_weights": {
            "sequence": round(float(gate_weights[0]), 3),
            "spectral": round(float(gate_weights[1]), 3),
            "chemical": round(float(gate_weights[2]), 3),
        },
        "sequence_length": len(sequence),
    }
