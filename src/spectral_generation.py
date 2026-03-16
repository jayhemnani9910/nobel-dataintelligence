"""
Spectral Generation Module for Quantum Data Decoder

Synthesizes continuous vibrational spectra from discrete normal modes
and provides utilities for spectral processing and feature extraction.
"""

import logging
from typing import Tuple, Optional
import numpy as np
from scipy.signal import convolve, find_peaks
from pathlib import Path

logger = logging.getLogger(__name__)


class SpectralGenerator:
    """
    Generate continuous vibrational spectra from discrete normal modes.
    
    Implements multiple spectral synthesis methods:
    - Simple Density of States (DOS) with Lorentzian broadening
    - Raman-weighted spectrum (mode-weighted by collective participation)
    - IR-active spectrum (intensity from dipole derivatives)
    """
    
    def __init__(self, freq_min: float = 0, freq_max: float = 500, 
                 n_points: int = 1000):
        """
        Initialize spectral generator.
        
        Args:
            freq_min: Minimum frequency (cm^-1)
            freq_max: Maximum frequency (cm^-1)
            n_points: Number of frequency points in output
        """
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.n_points = n_points
        self.freq_axis = np.linspace(freq_min, freq_max, n_points)
    
    def generate_dos(self, frequencies: np.ndarray, 
                     broadening: float = 5.0,
                     intensity_weighting: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate Density of States (DOS) spectrum.
        
        Creates a continuous spectrum by convolving discrete mode frequencies
        with a Lorentzian lineshape function.
        
        Args:
            frequencies: Array of mode frequencies (cm^-1)
            broadening: Lorentzian broadening parameter (cm^-1)
            intensity_weighting: Optional weights for each mode (e.g., collectivity)
            
        Returns:
            DOS spectrum: Shape (n_points,), normalized to [0, 1]
        """
        spectrum = np.zeros_like(self.freq_axis)
        
        if intensity_weighting is None:
            intensity_weighting = np.ones_like(frequencies)
        else:
            max_w = np.max(intensity_weighting)
            intensity_weighting = intensity_weighting / max_w if max_w > 0 else np.ones_like(frequencies)
        
        # Add Lorentzian peak for each mode
        gamma = broadening
        for freq, intensity in zip(frequencies, intensity_weighting):
            if freq < self.freq_min or freq > self.freq_max:
                continue
            
            # Lorentzian lineshape
            lorentzian = (gamma**2) / ((self.freq_axis - freq)**2 + gamma**2)
            spectrum += intensity * lorentzian
        
        # Normalize
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)
        
        logger.debug(f"Generated DOS spectrum with {len(frequencies)} modes "
                    f"and {broadening} cm^-1 broadening")
        
        return spectrum
    
    def generate_raman_spectrum(self, frequencies: np.ndarray, 
                               collectivities: np.ndarray,
                               broadening: float = 5.0) -> np.ndarray:
        """
        Generate Raman-weighted spectrum.
        
        In Raman scattering, spectral intensity is related to the polarizability
        derivative. For proteins, collective modes typically scatter more strongly.
        We approximate by weighting modes by their collectivity.
        
        Args:
            frequencies: Mode frequencies (cm^-1)
            collectivities: Mode collectivity scores (0-1)
            broadening: Lorentzian broadening
            
        Returns:
            Raman spectrum (arbitrary intensity units)
        """
        # Raman intensity typically scales as omega^4 for Rayleigh scattering
        # For our purposes, use collectivity-weighted spectrum
        return self.generate_dos(frequencies, broadening=broadening,
                               intensity_weighting=collectivities)
    
    def generate_ir_spectrum(self, frequencies: np.ndarray,
                            ir_activities: np.ndarray,
                            broadening: float = 5.0) -> np.ndarray:
        """
        Generate IR-active spectrum.
        
        IR absorption intensity depends on the change in dipole moment
        during vibration. We would need MD simulations to compute true
        IR activities; here we provide the framework.
        
        Args:
            frequencies: Mode frequencies
            ir_activities: IR activity for each mode (derivative of dipole)
            broadening: Lorentzian broadening
            
        Returns:
            IR spectrum
        """
        return self.generate_dos(frequencies, broadening=broadening,
                               intensity_weighting=ir_activities)
    
    def apply_instrumental_response(self, spectrum: np.ndarray,
                                   fwhm: float = 10.0) -> np.ndarray:
        """
        Apply instrumental broadening (Gaussian convolution).
        
        Real spectroscopy instruments have finite resolution.
        This simulates that effect.
        
        Args:
            spectrum: Input spectrum
            fwhm: Full-width at half-maximum of Gaussian (cm^-1)
            
        Returns:
            Convolved spectrum
        """
        if fwhm <= 0:
            denom = np.max(np.abs(spectrum))
            return spectrum / denom if denom > 0 else spectrum

        # Convert FWHM to sigma for Gaussian
        sigma_pix = fwhm / (self.freq_max - self.freq_min) * self.n_points
        if sigma_pix <= 0:
            denom = np.max(np.abs(spectrum))
            return spectrum / denom if denom > 0 else spectrum
        
        # Gaussian kernel
        kernel_size = int(4 * sigma_pix)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        x = np.arange(kernel_size) - kernel_size // 2
        gaussian = np.exp(-x**2 / (2 * sigma_pix**2))
        gaussian = gaussian / np.sum(gaussian)
        
        # Convolve
        convolved = convolve(spectrum, gaussian, mode='same')
        
        return convolved / np.max(convolved)
    
    def extract_spectral_features(self, spectrum: np.ndarray) -> dict:
        """
        Extract handcrafted features from spectrum for machine learning.
        
        Args:
            spectrum: Input spectrum
            
        Returns:
            Dictionary of scalar features
        """
        spectrum = np.asarray(spectrum)
        total_intensity = float(np.sum(spectrum))
        if not np.isfinite(total_intensity) or abs(total_intensity) < 1e-12:
            return {
                "integral": 0.0,
                "peak_height": 0.0,
                "peak_frequency": 0.0,
                "centroid": 0.0,
                "std_dev": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "num_peaks": 0,
            }

        features = {
            'integral': total_intensity,  # Total intensity
            'peak_height': float(np.max(spectrum)),  # Highest peak
            'peak_frequency': float(self.freq_axis[int(np.argmax(spectrum))]),  # Frequency of highest peak
            'centroid': float(np.sum(self.freq_axis * spectrum) / total_intensity),  # Centroid
            'std_dev': float(
                np.sqrt(np.sum(((self.freq_axis - np.mean(self.freq_axis))**2) * spectrum) / total_intensity)
            ),
            'skewness': float(self._compute_skewness(spectrum)),
            'kurtosis': float(self._compute_kurtosis(spectrum)),
        }
        
        # Count local maxima above 10% of the global maximum.
        threshold = 0.1 * float(np.max(spectrum))
        peaks, props = find_peaks(spectrum, height=threshold)
        features['num_peaks'] = int(len(peaks))
        
        return features
    
    def _compute_skewness(self, spectrum: np.ndarray) -> float:
        """Compute spectral skewness."""
        denom = np.sum(spectrum)
        if denom == 0:
            return 0
        mean = np.sum(self.freq_axis * spectrum) / denom
        m3 = np.sum(spectrum * (self.freq_axis - mean)**3)
        m2 = np.sum(spectrum * (self.freq_axis - mean)**2)
        skewness = m3 / (m2**1.5) if m2 > 0 else 0
        return skewness
    
    def _compute_kurtosis(self, spectrum: np.ndarray) -> float:
        """Compute spectral kurtosis."""
        denom = np.sum(spectrum)
        if denom == 0:
            return 0
        mean = np.sum(self.freq_axis * spectrum) / denom
        m4 = np.sum(spectrum * (self.freq_axis - mean)**4)
        m2 = np.sum(spectrum * (self.freq_axis - mean)**2)
        kurtosis = m4 / (m2**2) - 3 if m2 > 0 else 0
        return kurtosis
    
    def compute_spectral_correlation(self, spectrum1: np.ndarray,
                                    spectrum2: np.ndarray) -> float:
        """
        Compute correlation between two spectra (for structure comparison).
        
        Args:
            spectrum1: First spectrum
            spectrum2: Second spectrum
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Normalize
        denom1 = np.linalg.norm(spectrum1)
        denom2 = np.linalg.norm(spectrum2)
        if denom1 == 0 or denom2 == 0:
            return 0.0
        spec1_norm = spectrum1 / denom1
        spec2_norm = spectrum2 / denom2
        
        correlation = np.dot(spec1_norm, spec2_norm)
        return correlation


class DeltaSpectralFeatures:
    """
    Compute delta (change) features between two structures.
    
    Useful for mutation analysis where we compare WT vs mutant.
    """
    
    def __init__(self, spectral_generator: SpectralGenerator):
        self.sg = spectral_generator
    
    def compute_delta_features(self, spectrum_wt: np.ndarray,
                              spectrum_mut: np.ndarray,
                              freq_vib_wt: float,
                              freq_vib_mut: float) -> dict:
        """
        Compute delta features for stability prediction.
        
        Args:
            spectrum_wt: Wild-type spectrum
            spectrum_mut: Mutant spectrum
            freq_vib_wt: Wild-type vibrational entropy
            freq_vib_mut: Mutant vibrational entropy
            
        Returns:
            Dictionary of delta features
        """
        # Spectral differences
        delta_spectrum = spectrum_mut - spectrum_wt
        
        features_wt = self.sg.extract_spectral_features(spectrum_wt)
        features_mut = self.sg.extract_spectral_features(spectrum_mut)
        
        delta_features = {
            'delta_integral': features_mut['integral'] - features_wt['integral'],
            'delta_peak_height': features_mut['peak_height'] - features_wt['peak_height'],
            'delta_centroid': features_mut['centroid'] - features_wt['centroid'],
            'delta_entropy_j_mol_k': freq_vib_mut - freq_vib_wt,
            'spectral_correlation': self.sg.compute_spectral_correlation(spectrum_wt, spectrum_mut),
            'spectral_l2_norm': np.linalg.norm(delta_spectrum),
        }
        
        return delta_features


