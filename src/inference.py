"""
Inference utilities for Quantum Data Decoder.

Provides simple prediction APIs for both Novozymes stability prediction
and VibroPredict enzyme kinetics prediction using saved checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _generate_vdos_for_sequence(
    sequence: str, pdb_path: str | None = None, n_points: int = 1000
) -> np.ndarray:
    """Generate a VDOS spectrum for a protein.

    If ``pdb_path`` is given, computes a *real* VDOS from the structure via
    normal-mode analysis (the same corrected GNM pipeline used in training).
    This is strongly preferred and gives a protein-specific spectrum.

    If no structure is available, falls back to a crude sequence-length-only
    pseudo-spectrum (``sqrt(1..N) * 15``). This fallback carries **no
    structural information** — it is a function of chain length alone — so
    predictions built on it are not physically meaningful. A warning is emitted.
    """
    from .spectral_generation import SpectralGenerator

    if pdb_path is not None:
        # Real NMA-derived VDOS from the structure (corrected eigenvalue scaling).
        from vibropredict.spectra.vdos_engine import VibroEnzymePipeline

        pipeline = VibroEnzymePipeline(n_points=n_points, freq_max=500.0, broadening=5.0)
        vdos, _ = pipeline.generate_vdos(pdb_path)
        logger.info("Computed real NMA VDOS from structure %s", pdb_path)
        return np.asarray(vdos, dtype=np.float64)

    logger.warning(
        "No structure provided (pdb_path=None): falling back to a sequence-length "
        "pseudo-VDOS that contains NO structural signal. Predictions will not be "
        "physically meaningful. Pass a PDB structure for real NMA-based VDOS."
    )
    n_residues = len(sequence)
    frequencies = np.sqrt(np.arange(1, min(n_residues, 100) + 1, dtype=np.float64)) * 15.0

    sg = SpectralGenerator(freq_min=0, freq_max=500, n_points=n_points)
    vdos = sg.generate_dos(frequencies, broadening=5.0)
    return vdos


def _resolve_checkpoint(checkpoint_path: Path) -> Path:
    """Validate a checkpoint path, raising a helpful error if missing.

    Lists any checkpoints actually present in the directory so the user knows
    what is available. Note: the ``best_model_epoch*.pt`` files bundled in the
    repo are toy artifacts from the training smoke test (a 2-parameter linear
    stub) and are NOT compatible with the inference model architectures — do
    not point ``--checkpoint`` at them.
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        return checkpoint_path

    parent = checkpoint_path.parent
    available = sorted(p.name for p in parent.glob("*.pt")) if parent.exists() else []
    hint = (
        f" Checkpoints found in {parent}/: {available}."
        " (Note: bundled best_model_epoch*.pt are toy smoke-test stubs, not"
        " trained inference models.)"
        if available
        else f" No .pt checkpoints found in {parent}/."
    )
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}. Train a model first"
        f" (Colab notebook or CLI) and point --checkpoint at it.{hint}"
    )


def predict_stability(
    sequence: str,
    pH: float = 7.0,
    checkpoint_path: str = "checkpoints/novozymes_best.pt",
    device: str = "cpu",
    pdb_path: str | None = None,
) -> dict:
    """Predict melting temperature (Tm) for a protein mutation.

    Args:
        sequence: Amino acid sequence of the mutant protein.
        pH: pH value for the assay condition.
        checkpoint_path: Path to a trained VibroStructuralModel checkpoint.
        device: Device for inference ('cpu' or 'cuda').
        pdb_path: Optional path to a PDB structure. If given, real C-alpha
            coordinates and real NMA-derived VDOS are used; otherwise a
            structure-free extended-chain graph and pseudo-VDOS are used
            (a warning is emitted, and results are not physically meaningful).

    Returns:
        Dictionary with 'predicted_tm' (float) and 'sequence_length' (int).
    """
    from .models.gnn import GraphConstruction
    from .models.multimodal import VibroStructuralModel

    checkpoint_path = _resolve_checkpoint(Path(checkpoint_path))

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

    # Build graph from sequence. With a structure, use real C-alpha coordinates;
    # otherwise fall back to an extended-chain placeholder (no 3D information).
    features = GraphConstruction.construct_residue_features(sequence)
    if pdb_path is not None:
        import prody as _pr

        _ca = _pr.parsePDB(pdb_path).select("name CA")
        coords = torch.tensor(_ca.getCoords(), dtype=torch.float32)
        if coords.shape[0] != len(sequence):
            logger.warning(
                "Structure has %d CA atoms but sequence length is %d; "
                "using structure coordinates and truncating features to match.",
                coords.shape[0],
                len(sequence),
            )
            features = features[: coords.shape[0]]
    else:
        logger.warning(
            "No structure provided: building an extended-chain graph "
            "(no real 3D topology). Pass pdb_path for a real structure."
        )
        coords = torch.stack(
            [
                torch.arange(len(sequence), dtype=torch.float32) * 3.8,
                torch.zeros(len(sequence)),
                torch.zeros(len(sequence)),
            ],
            dim=1,
        )
    graph = GraphConstruction.construct_ca_graph(coords, features, distance_cutoff=10.0)
    graph.batch = torch.zeros(coords.shape[0], dtype=torch.long)

    # Generate VDOS (real NMA if a structure was supplied)
    vdos = _generate_vdos_for_sequence(sequence, pdb_path=pdb_path)
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
    product_smiles: str | None = None,
    checkpoint_path: str = "checkpoints/vibropredict_best.pt",
    device: str = "cpu",
    pdb_path: str | None = None,
) -> dict:
    """Predict catalytic turnover (k_cat) for an enzyme-substrate pair.

    Args:
        sequence: Amino acid sequence of the enzyme.
        substrate_smiles: SMILES string for the substrate.
        product_smiles: Optional SMILES string for the product.
        checkpoint_path: Path to a trained VibroPredictHybrid checkpoint.
        device: Device for inference ('cpu' or 'cuda').
        pdb_path: Optional path to a PDB structure for real NMA-derived VDOS.
            Without it, a structure-free pseudo-VDOS is used (warned; not
            physically meaningful).

    Returns:
        Dictionary with 'predicted_log_kcat', 'predicted_kcat', and 'gate_weights'.
    """
    from vibropredict.models.vibropredict_hybrid import VibroPredictHybrid

    checkpoint_path = _resolve_checkpoint(Path(checkpoint_path))

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Reconstruct model
    model = VibroPredictHybrid(fusion_dim=512, dropout=0.0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Generate VDOS (real NMA if a structure was supplied)
    vdos = _generate_vdos_for_sequence(sequence, pdb_path=pdb_path)
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
        "predicted_kcat": round(10**log_kcat, 2),
        "gate_weights": {
            "sequence": round(float(gate_weights[0]), 3),
            "spectral": round(float(gate_weights[1]), 3),
            "chemical": round(float(gate_weights[2]), 3),
        },
        "sequence_length": len(sequence),
    }
