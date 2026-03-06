"""
Command-line utilities for running QDD pipelines.

Designed for reproducible, end-to-end runs:
- Train + predict Novozymes (submission CSV)
- Predict CAFA5 (prediction/submission CSV)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from .datasets import NovozymesDataset, CAFA5Dataset, create_dataloaders
from .models.multimodal import VibroStructuralModel
from .training import Trainer, MetricComputer
from .utils import batch_collate_function, get_device, set_seed

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_fasta_ids(fasta_path: Path) -> list[str]:
    ids: list[str] = []
    with fasta_path.open("r") as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
    return ids


def _write_novozymes_submission(out_path: Path, seq_ids: Iterable[str], preds: np.ndarray) -> None:
    df = pd.DataFrame({"seq_id": list(seq_ids), "tm": preds.astype(float)})
    df.to_csv(out_path, index=False)


def _write_cafa5_submission(out_path: Path, protein_ids: Iterable[str], go_terms_per_protein: list[list[str]]) -> None:
    rows = []
    for pid, terms in zip(protein_ids, go_terms_per_protein, strict=False):
        rows.append({"protein_id": pid, "go_terms": " ".join(terms) if terms else "GO:0005575"})
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_novozymes(
    data_dir: Path,
    out_path: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str | None,
) -> None:
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    structure_pdb = data_dir / "wildtype_structure_prediction_af2.pdb"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing required file: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing required file: {test_csv}")

    spectra_dir = data_dir.parent / "spectral"
    _ensure_dir(spectra_dir)

    set_seed(seed)
    resolved_device = device or get_device()

    # Train dataset
    train_dataset = NovozymesDataset(
        csv_file=str(train_csv),
        structure_file=str(structure_pdb),
        spectra_dir=str(spectra_dir),
        include_updates=True,
    )

    # Split
    n_total = len(train_dataset)
    n_train = int(0.8 * n_total)
    n_val = max(1, n_total - n_train)
    train_split, val_split = random_split(train_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=batch_collate_function,
    )
    val_loader = DataLoader(
        val_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=batch_collate_function,
    )

    # Model + trainer
    model = VibroStructuralModel(
        latent_dim=128,
        gnn_input_dim=24,
        fusion_type="bilinear",
        dropout=0.2,
        num_go_terms=100,  # unused for Novozymes
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    trainer = Trainer(model=model, optimizer=optimizer, device=resolved_device, checkpoint_dir=str(data_dir / "checkpoints"))
    loss_fn = torch.nn.MSELoss()

    if epochs > 0:
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            epochs=epochs,
            metric_fn=MetricComputer.spearman_correlation,
            early_stopping_patience=max(3, min(10, epochs // 2)),
            task="novozymes",
        )

    # Predict on test.csv
    df_test = pd.read_csv(test_csv)
    if "seq_id" not in df_test.columns or "protein_sequence" not in df_test.columns or "pH" not in df_test.columns:
        raise ValueError("Unexpected Novozymes test.csv format; expected columns: seq_id, protein_sequence, pH")

    test_dataset = NovozymesDataset(
        csv_file=str(test_csv),
        structure_file=str(structure_pdb),
        spectra_dir=str(spectra_dir),
        include_updates=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=batch_collate_function,
    )

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            graph = batch["graph"].to(trainer.device)
            spectra = batch["spectra"].to(trainer.device)
            global_features = batch.get("global_features")
            if global_features is not None:
                global_features = global_features.to(trainer.device)
            outputs = model(graph, spectra, global_features=global_features, task="novozymes").squeeze(-1)
            preds.append(outputs.detach().cpu().numpy())

    preds_arr = np.concatenate(preds, axis=0)
    _ensure_dir(out_path.parent)
    _write_novozymes_submission(out_path, df_test["seq_id"].tolist(), preds_arr)
    logger.info(f"Wrote Novozymes submission: {out_path}")


def run_cafa5(
    data_dir: Path,
    out_path: Path,
    top_k_terms: int,
) -> None:
    terms_csv = data_dir / "train_terms.csv"
    test_fasta = data_dir / "test_sequences.fasta"

    if not terms_csv.exists():
        raise FileNotFoundError(f"Missing required file: {terms_csv}")
    if not test_fasta.exists():
        raise FileNotFoundError(f"Missing required file: {test_fasta}")

    terms_df_raw = pd.read_csv(terms_csv)
    # Reuse CAFA5Dataset normalization by instantiating a tiny dataset object.
    dummy = CAFA5Dataset(
        sequences_fasta=str(test_fasta),
        terms_csv=str(terms_csv),
        spectra_dir=str(data_dir / "spectral"),
        structure_dir=str(data_dir / "structures"),
        go_terms_list=None,
    )
    go_terms = dummy.go_terms
    if not go_terms:
        raise ValueError("No GO terms found in train_terms.csv")

    # Baseline: predict the most frequent GO terms for every protein.
    # This is fast, deterministic, and produces a valid file even without training.
    term_counts = dummy.terms_df["go_id"].value_counts()
    top_terms = term_counts.head(max(1, int(top_k_terms))).index.tolist()

    protein_ids = _load_fasta_ids(test_fasta)
    predictions = [top_terms for _ in protein_ids]

    _ensure_dir(out_path.parent)
    _write_cafa5_submission(out_path, protein_ids, predictions)
    logger.info(f"Wrote CAFA5 predictions: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qdd", description="Quantum Data Decoder pipelines")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_novo = sub.add_parser("novozymes", help="Train + predict Novozymes; write submission CSV")
    p_novo.add_argument("--data-dir", type=Path, default=Path("./data/kaggle"))
    p_novo.add_argument("--out", type=Path, default=Path("./submissions/novozymes_submission.csv"))
    p_novo.add_argument("--epochs", type=int, default=5)
    p_novo.add_argument("--batch-size", type=int, default=16)
    p_novo.add_argument("--seed", type=int, default=42)
    p_novo.add_argument("--device", type=str, default=None, help="cpu or cuda")

    p_cafa = sub.add_parser("cafa5", help="Predict CAFA5; write predictions/submission CSV")
    p_cafa.add_argument("--data-dir", type=Path, default=Path("./data/cafa5"))
    p_cafa.add_argument("--out", type=Path, default=Path("./submissions/cafa5_predictions.csv"))
    p_cafa.add_argument("--top-k-terms", type=int, default=25)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "novozymes":
        run_novozymes(
            data_dir=args.data_dir,
            out_path=args.out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        return 0

    if args.cmd == "cafa5":
        run_cafa5(data_dir=args.data_dir, out_path=args.out, top_k_terms=args.top_k_terms)
        return 0

    raise AssertionError("Unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())

