#!/usr/bin/env python3
"""
Real-world test: use QDD's physics pipeline to distinguish a fragile
enzyme (IsPETase wild-type, Tm ~45°C) from a thermostable engineered
variant (LCC-ICCG, Tm ~94°C) — purely from their PDB structures.

This is the part of the pipeline that runs without ML (no torch / no
trained checkpoint required): NMA → vibrational entropy + VDOS.

Outputs:
    benchmarks/real_world/petase_vs_lcc_summary.json
    benchmarks/real_world/vdos_<name>.npy
    benchmarks/real_world/vdos_comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.nma_analysis import ANMAnalyzer

CASES = [
    {
        "name": "IsPETase_wildtype",
        "pdb": "data/pdb/6eqe.pdb",
        "Tm_measured_C": 45.0,
        "note": "Wild-type Ideonella sakaiensis PETase",
        "color": "#ef4444",
    },
    {
        "name": "LCC-ICCG_engineered",
        "pdb": "data/pdb/6ths.pdb",
        "Tm_measured_C": 94.0,
        "note": "Engineered LCC with stabilizing disulfide (Carbios-derived)",
        "color": "#22c55e",
    },
]

OUTPUT_DIR = Path("benchmarks/real_world")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = {}
vdos_traces: dict[str, np.ndarray] = {}
K = 100  # number of modes to keep
BROADENING = 5.0  # cm^-1

for case in CASES:
    name = case["name"]
    print(f"\n=== {name} (Tm={case['Tm_measured_C']}°C) ===")

    anm = ANMAnalyzer(case["pdb"], cutoff=15.0)
    freqs, _ = anm.compute_modes(k=K)
    positive = freqs[freqs > 1e-6]

    # Use the project's native VDOS computation
    vdos = anm.compute_vdos(k=K, broadening=BROADENING)
    vdos_traces[name] = vdos
    np.save(OUTPUT_DIR / f"vdos_{name}.npy", vdos)

    # Vibrational entropy (J/mol/K; = R * sum over modes) at 298.15 K
    entropy = anm.compute_vibrational_entropy(k=K, temperature=298.15)

    # Spectral bookkeeping
    # src/nma_analysis.py builds its VDOS grid as np.linspace(0, 500, 1000) cm^-1
    freq_grid = np.linspace(0.0, 500.0, len(vdos))
    vdos_integral = float(np.trapezoid(vdos, freq_grid))
    peak_freq = float(freq_grid[int(np.argmax(vdos))])
    low_freq_mask = freq_grid < 100.0
    low_freq_fraction = float(
        np.trapezoid(vdos[low_freq_mask], freq_grid[low_freq_mask]) / vdos_integral
    )
    mean_freq = float(np.trapezoid(vdos * freq_grid, freq_grid) / vdos_integral)

    results[name] = {
        "Tm_measured_C": case["Tm_measured_C"],
        "note": case["note"],
        "n_modes_used": int(len(positive)),
        "min_eigenvalue": float(positive.min()),
        "mean_eigenvalue": float(positive.mean()),
        "vibrational_entropy_J_mol_K": float(entropy),
        "vdos_peak_freq_cm1": peak_freq,
        "vdos_mean_freq_cm1": mean_freq,
        "vdos_low_freq_fraction_under_100cm1": low_freq_fraction,
    }
    print(f"  modes used:              {results[name]['n_modes_used']}")
    print(f"  vib. entropy (J/mol/K):  {results[name]['vibrational_entropy_J_mol_K']:.3f}")
    print(f"  VDOS peak (cm^-1):       {results[name]['vdos_peak_freq_cm1']:.2f}")
    print(f"  VDOS mean freq (cm^-1):  {results[name]['vdos_mean_freq_cm1']:.2f}")
    print(f"  <100cm^-1 fraction:      {results[name]['vdos_low_freq_fraction_under_100cm1']:.3f}")

# --- Save summary JSON ---
with open(OUTPUT_DIR / "petase_vs_lcc_summary.json", "w") as f:
    json.dump(results, f, indent=2)

# --- Plot comparison ---
freq_grid = np.linspace(0.0, 500.0, len(next(iter(vdos_traces.values()))))
fig, ax = plt.subplots(figsize=(8, 4.5))
for case in CASES:
    name = case["name"]
    v = vdos_traces[name]
    ax.plot(
        freq_grid,
        v / v.max(),
        label=f"{name}  (Tm≈{case['Tm_measured_C']}°C)",
        color=case["color"],
        linewidth=2,
    )
ax.set_xlabel("Frequency (cm⁻¹)")
ax.set_ylabel("Normalized VDOS")
ax.set_title("Vibrational density of states — fragile vs thermostable PET-degrading enzyme")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 500)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "vdos_comparison.png", dpi=140)
plt.close(fig)

# --- Verdict ---
ft = results["IsPETase_wildtype"]
lc = results["LCC-ICCG_engineered"]
entropy_delta = lc["vibrational_entropy_J_mol_K"] - ft["vibrational_entropy_J_mol_K"]
lowfreq_delta = (
    lc["vdos_low_freq_fraction_under_100cm1"] - ft["vdos_low_freq_fraction_under_100cm1"]
)
print("\n=== VERDICT ===")
print(f"LCC is {lc['Tm_measured_C'] - ft['Tm_measured_C']:+.1f}°C more thermostable than IsPETase.")
print(f"Δ vibrational entropy (LCC - WT): {entropy_delta:+.3f} J/mol/K")
print(f"Δ low-frequency VDOS fraction:   {lowfreq_delta:+.4f}")
print("\nInterpretation:")
print("- A MORE thermostable protein is typically STIFFER at physiological temps,")
print("  meaning its mode spectrum shifts to HIGHER frequencies.")
print("- Expected sign: LCC should have LOWER entropy.")
print("- If we see that, the physics pipeline is consistent with the measured Tm gap.")
