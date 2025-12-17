"""
Normal Mode Analysis (NMA) Module for Quantum Data Decoder

Implements:
- Anisotropic Network Model (ANM) calculations
- Vibrational frequency extraction
- Vibrational entropy computation
- Density of States (DOS) synthesis
"""

import logging
from typing import Tuple, Optional
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from pathlib import Path

def _require_prody():
    try:
        import prody as pr  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "ProDy is required for Normal Mode Analysis (NMA). Install it with `pip install prody`."
        ) from exc
    return pr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ANMAnalyzer:
    """
    Anisotropic Network Model analyzer for protein dynamics.
    
    ANM treats the protein as a network of coarse-grained C-alpha atoms
    connected by springs. This enables efficient computation of normal modes
    even for large proteins.
    
    Key parameters:
    - Cutoff distance: 15 Angstroms (default)
    - Force constant weighting: distance-dependent (r^-2)
    - Number of modes: typically 100-200
    """
    
    def __init__(self, structure, cutoff: float = 15.0, 
                 distance_weighted: bool = True):
        """
        Initialize ANM analyzer.
        
        Args:
            structure: ProDy AtomGroup or path to PDB file
            cutoff: Distance cutoff for C-alpha connections (Angstroms)
            distance_weighted: Use distance-dependent force constants
        """
        pr = _require_prody()

        self.cutoff = cutoff
        self.distance_weighted = distance_weighted
        
        # Load structure if path provided
        if isinstance(structure, str):
            self.structure = pr.parsePDB(structure)
            logger.info(f"Loaded structure: {Path(structure).name}")
        else:
            self.structure = structure
        
        # Extract C-alpha atoms
        self.ca_atoms = self.structure.select('ca')
        if self.ca_atoms is None:
            raise ValueError("No C-alpha atoms found in structure")
        
        self.n_atoms = self.ca_atoms.numAtoms()
        logger.info(f"Selected {self.n_atoms} C-alpha atoms for ANM")
        
        # Initialize ANM using ProDy
        self.anm = pr.ANM(f"ANM_{self.n_atoms}")
        self.anm.buildHessian(self.ca_atoms, cutoff=self.cutoff)
        
        # Store coordinates and masses
        self.coordinates = self.ca_atoms.getCoords()
        self.masses = self.ca_atoms.getMasses()
        
        self._eigenvalues = None
        self._eigenvectors = None
        self._frequencies = None
    
    def compute_modes(self, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the first k non-zero normal modes.
        
        Args:
            k: Number of modes to compute
            
        Returns:
            Tuple of (frequencies, eigenvectors)
            - frequencies: Shape (k,), in cm^-1
            - eigenvectors: Shape (3*n_atoms, k), columns are mode vectors
        """
        logger.info(f"Computing first {k} normal modes...")

        # ProDy requires an explicit mode calculation step after building the Hessian.
        # ANM has 6 rigid-body modes; request a few extra to ensure we can return k non-zero modes.
        n_skip = 6
        n_total = min(3 * self.n_atoms, n_skip + int(k))
        if n_total <= 0:
            raise ValueError("Cannot compute modes for an empty structure.")

        try:
            self.anm.calcModes(n_total)
        except Exception as exc:
            raise RuntimeError("ProDy failed to calculate ANM modes.") from exc

        eigvals = self.anm.getEigvals()
        eigvecs = self.anm.getEigvecs()
        if eigvals is None or eigvecs is None:
            raise RuntimeError("ProDy returned no eigenvalues/eigenvectors. Did mode calculation succeed?")

        eigvals = np.asarray(eigvals)
        eigvecs = np.asarray(eigvecs)

        # Filter rigid-body/near-zero modes robustly (can be >6 if graph is disconnected).
        zero_tol = 1e-8
        nonzero_idx = np.where(eigvals > zero_tol)[0]
        if nonzero_idx.size == 0:
            raise RuntimeError("All ANM eigenvalues are (near-)zero; cannot compute vibrational modes.")

        # Ensure lowest modes first.
        nonzero_idx = nonzero_idx[np.argsort(eigvals[nonzero_idx])]
        selected_idx = nonzero_idx[: min(k, nonzero_idx.size)]

        self._eigenvalues = eigvals[selected_idx]
        self._eigenvectors = eigvecs[:, selected_idx]

        # Convert eigenvalues (omega^2) to frequencies in cm^-1.
        conversion_factor = 1 / (2 * np.pi * 29979.2458)  # 1/c in fs/cm
        frequencies = np.sqrt(np.maximum(self._eigenvalues, 0)) * conversion_factor

        self._frequencies = frequencies
        if frequencies.size:
            logger.info(
                f"Mode frequencies range: {frequencies.min():.2f} - {frequencies.max():.2f} cm^-1"
            )

        return frequencies, self._eigenvectors
    
    def compute_vdos(self, k: int = 100, broadening: float = 5.0) -> np.ndarray:
        """
        Compute Vibrational Density of States (VDOS) with Lorentzian broadening.
        
        VDOS is the spectral distribution of vibrational modes, often used
        to compare with experimental Raman or neutron spectroscopy data.
        
        Args:
            k: Number of modes to use
            broadening: Lorentzian broadening factor (cm^-1)
            
        Returns:
            VDOS spectrum: Shape (1000,), representing 0-500 cm^-1 range
        """
        if self._frequencies is None:
            self.compute_modes(k=k)
        
        # Generate frequency axis
        freq_min, freq_max = 0, 500  # cm^-1
        freq_axis = np.linspace(freq_min, freq_max, 1000)
        vdos = np.zeros_like(freq_axis)
        
        # Add Lorentzian peaks for each mode
        for f_mode in self._frequencies:
            if f_mode > freq_max:
                break
            # Lorentzian lineshape: L(f) = (Gamma^2) / ((f - f0)^2 + Gamma^2)
            gamma = broadening
            vdos += (gamma**2) / ((freq_axis - f_mode)**2 + gamma**2)
        
        # Normalize
        vdos = vdos / np.max(vdos) if np.max(vdos) > 0 else vdos
        
        logger.info(f"Computed VDOS with {broadening} cm^-1 broadening")
        return vdos
    
    def compute_vibrational_entropy(self, k: int = 100, 
                                   temperature: float = 298.15) -> float:
        """
        Compute vibrational entropy (S_vib) from normal modes.
        
        In the quantum harmonic oscillator approximation:
        S_vib = k_B * sum_i [ x_i / (exp(x_i) - 1) - ln(1 - exp(-x_i)) ]
        where x_i = hbar * omega_i / (k_B * T)
        
        This thermodynamic quantity encodes the entropic contribution to
        protein stability and is particularly important for predicting
        the effect of mutations.
        
        Args:
            k: Number of modes
            temperature: Temperature in Kelvin
            
        Returns:
            Vibrational entropy in J/(mol*K)
        """
        if self._frequencies is None:
            self.compute_modes(k=k)
        
        # Physical constants
        kb = 1.380649e-23  # Boltzmann constant (J/K)
        hbar = 1.054571817e-34  # Reduced Planck constant (J*s)
        avogadro = 6.02214076e23  # Avogadro's number
        
        # Convert frequencies from cm^-1 to rad/s
        # 1 cm^-1 = 100 m^-1 = 2*pi*c / 100 rad/s
        c = 299792458  # speed of light (m/s)
        omega = self._frequencies * 100 * 2 * np.pi * c  # rad/s
        
        # Compute dimensionless variable x = hbar*omega / (k_B*T)
        x = (hbar * omega) / (kb * temperature)
        
        # Avoid numerical issues with very large x
        x = np.minimum(x, 100)
        
        # Compute entropy per mode
        exp_x = np.exp(x)
        s_vib_modes = x / (exp_x - 1) - np.log(1 - np.exp(-x))
        
        # Total entropy per mole
        s_vib_total = kb * avogadro * np.sum(s_vib_modes)
        
        logger.info(f"Vibrational entropy (T={temperature}K): {s_vib_total:.2f} J/(mol*K)")
        return s_vib_total
    
    def get_mode_collectivity(self, mode_idx: int = 0) -> float:
        """
        Compute collectivity of a mode (0-1).
        
        Collectivity measures how many atoms participate in a mode.
        High collectivity (close to 1) indicates a global/delocalized mode.
        Low collectivity (close to 0) indicates a local/localized mode.
        
        Args:
            mode_idx: Mode index (0 = lowest frequency non-rigid mode)
            
        Returns:
            Collectivity score (0-1)
        """
        if self._eigenvectors is None or self._eigenvectors.shape[1] <= mode_idx:
            self.compute_modes(k=mode_idx + 1)
        
        # Get eigenvector for this mode
        mode_vector = self._eigenvectors[:, mode_idx]
        
        # Reshape to (n_atoms, 3) and compute participation per atom
        mode_matrix = mode_vector.reshape(self.n_atoms, 3)  # (n_atoms, 3)
        participation = np.sum(mode_matrix**2, axis=1)  # (n_atoms,)
        participation = participation / np.sum(participation)  # normalize
        
        # Collectivity: 1/sum(p_i^2) normalized by N
        collectivity = (1 / (np.sum(participation**2))) / self.n_atoms
        
        return collectivity
    
    def get_residue_fluctuations(self, k: int = 100) -> np.ndarray:
        """
        Compute mean-square fluctuations for each residue from modes.
        
        This provides a measure of thermal mobility and can be compared
        to experimental B-factors.
        
        Args:
            k: Number of modes
            
        Returns:
            Fluctuations: Shape (n_atoms,), in Angstroms^2
        """
        if self._eigenvectors is None or self._eigenvalues is None or self._eigenvectors.shape[1] < k:
            self.compute_modes(k=k)
        
        # Mean-square fluctuation from mode decomposition
        # <u^2> = sum_i |v_i|^2 / lambda_i (in harmonic approximation)
        mode_matrix = self._eigenvectors[:, :k].reshape(self.n_atoms, 3, k)  # (n_atoms, 3, k)
        mode_energies = self._eigenvalues[:k]
        
        # Avoid division by zero
        mode_energies = np.maximum(mode_energies, 1e-6)
        
        fluctuations = np.sum(np.sum(mode_matrix**2, axis=1) / mode_energies, axis=1)
        
        logger.info(f"Mean-square fluctuation: {np.mean(fluctuations):.3f} Å²")
        return fluctuations


class GNMAnalyzer:
    """
    Gaussian Network Model analyzer (simplified coarse-grained model).
    
    GNM is even more coarse-grained than ANM and is useful for
    very large proteins or quick feasibility studies.
    """
    
    def __init__(self, structure, cutoff: float = 10.0):
        """Initialize GNM analyzer."""
        pr = _require_prody()

        self.cutoff = cutoff
        
        if isinstance(structure, str):
            self.structure = pr.parsePDB(structure)
        else:
            self.structure = structure
        
        self.ca_atoms = self.structure.select('ca')
        self.n_atoms = self.ca_atoms.numAtoms()
        
        # Initialize GNM
        self.gnm = pr.GNM(f"GNM_{self.n_atoms}")
        self.gnm.buildKirchhoff(self.ca_atoms, cutoff=self.cutoff)
        
        logger.info(f"Initialized GNM with {self.n_atoms} C-alpha atoms")
    
    def compute_modes(self, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GNM modes."""
        logger.info(f"Computing first {k} GNM modes...")
        n_skip = 1  # GNM has a single rigid-body mode
        n_total = min(self.n_atoms, n_skip + int(k))
        if n_total <= 0:
            raise ValueError("Cannot compute modes for an empty structure.")

        try:
            self.gnm.calcModes(n_total)
        except Exception as exc:
            raise RuntimeError("ProDy failed to calculate GNM modes.") from exc

        eigenvalues = self.gnm.getEigvals()
        eigenvectors = self.gnm.getEigvecs()
        if eigenvalues is None or eigenvectors is None:
            raise RuntimeError("ProDy returned no eigenvalues/eigenvectors for GNM.")

        eigenvalues = np.asarray(eigenvalues)
        eigenvectors = np.asarray(eigenvectors)

        zero_tol = 1e-8
        nonzero_idx = np.where(eigenvalues > zero_tol)[0]
        nonzero_idx = nonzero_idx[np.argsort(eigenvalues[nonzero_idx])]
        selected_idx = nonzero_idx[: min(k, nonzero_idx.size)]

        return eigenvalues[selected_idx], eigenvectors[:, selected_idx]


def compare_structures(pdb1_path: str, pdb2_path: str, k: int = 50) -> dict:
    """
    Compare vibrational properties of two structures (e.g., WT vs mutant).
    
    Useful for analyzing the effect of mutations on protein dynamics.
    
    Args:
        pdb1_path: Path to first structure (e.g., wild-type)
        pdb2_path: Path to second structure (e.g., mutant)
        k: Number of modes to analyze
        
    Returns:
        Dictionary with comparison metrics
    """
    logger.info(f"Comparing structures: {Path(pdb1_path).name} vs {Path(pdb2_path).name}")
    
    # Analyze both structures
    anm1 = ANMAnalyzer(pdb1_path)
    anm2 = ANMAnalyzer(pdb2_path)
    
    freq1, _ = anm1.compute_modes(k=k)
    freq2, _ = anm2.compute_modes(k=k)
    
    s_vib1 = anm1.compute_vibrational_entropy(k=k)
    s_vib2 = anm2.compute_vibrational_entropy(k=k)
    
    # Compute delta properties
    delta_s_vib = s_vib2 - s_vib1
    freq_shift = np.mean(freq2 - freq1)
    
    result = {
        'structure1': Path(pdb1_path).name,
        'structure2': Path(pdb2_path).name,
        'delta_entropy_j_mol_k': delta_s_vib,
        'entropy_shift_pct': (delta_s_vib / s_vib1) * 100,
        'frequency_shift_cm1': freq_shift,
        'vdos1': anm1.compute_vdos(k=k),
        'vdos2': anm2.compute_vdos(k=k),
    }
    
    logger.info(f"ΔS_vib = {delta_s_vib:.2f} J/(mol*K) "
               f"({(delta_s_vib/s_vib1)*100:+.1f}%)")
    
    return result


def main():
    """Demonstration of NMA analysis."""
    try:
        pr = _require_prody()
    except ImportError as exc:
        logger.error(str(exc))
        return

    logger.info("=" * 60)
    logger.info("Quantum Data Decoder: NMA Analysis Module")
    logger.info("=" * 60)
    
    # Download Ubiquitin structure for testing
    logger.info("\nDownloading Ubiquitin (1UBQ) for testing...")
    try:
        pr.fetchPDB('1UBQ', folder='./data/pdb')
    except Exception as exc:
        logger.error(f"Failed to fetch PDB 1UBQ (offline or network issue?): {exc}")
        return
    pdb_path = './data/pdb/1ubq.pdb'
    
    # Run ANM analysis
    logger.info("\n[Test 1] Anisotropic Network Model Analysis")
    anm = ANMAnalyzer(pdb_path, cutoff=15.0)
    freqs, modes = anm.compute_modes(k=100)
    
    # Compute VDOS
    vdos = anm.compute_vdos(k=100, broadening=5.0)
    
    # Compute entropy
    s_vib = anm.compute_vibrational_entropy(k=100, temperature=298.15)
    
    # Get collectivity of lowest mode
    collectivity = anm.get_mode_collectivity(mode_idx=0)
    logger.info(f"Lowest mode collectivity: {collectivity:.3f}")
    
    # Get fluctuations
    fluct = anm.get_residue_fluctuations(k=100)
    logger.info(f"Residue fluctuations: min={fluct.min():.3f}, max={fluct.max():.3f} Å²")
    
    logger.info("\n" + "=" * 60)
    logger.info("NMA analysis complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
