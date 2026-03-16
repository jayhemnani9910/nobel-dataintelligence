"""Tests for spectral generation module."""

import unittest
import numpy as np

from src.spectral_generation import SpectralGenerator, DeltaSpectralFeatures


class TestSpectralGeneratorDOS(unittest.TestCase):
    """Test Density of States generation."""

    def setUp(self):
        self.sg = SpectralGenerator(freq_min=0, freq_max=500, n_points=1000)

    def test_single_mode(self):
        freqs = np.array([250.0])
        dos = self.sg.generate_dos(freqs, broadening=5.0)
        self.assertEqual(dos.shape, (1000,))
        # Peak should be near the mode frequency
        peak_idx = np.argmax(dos)
        peak_freq = self.sg.freq_axis[peak_idx]
        self.assertAlmostEqual(peak_freq, 250.0, delta=5.0)

    def test_output_normalized(self):
        freqs = np.array([100.0, 200.0, 300.0])
        dos = self.sg.generate_dos(freqs, broadening=5.0)
        self.assertAlmostEqual(np.max(dos), 1.0, places=5)

    def test_empty_frequencies(self):
        dos = self.sg.generate_dos(np.array([]), broadening=5.0)
        np.testing.assert_array_equal(dos, np.zeros(1000))

    def test_out_of_range_frequencies_ignored(self):
        freqs = np.array([600.0, 700.0])  # Above freq_max=500
        dos = self.sg.generate_dos(freqs, broadening=5.0)
        np.testing.assert_array_equal(dos, np.zeros(1000))


class TestSpectralFeatures(unittest.TestCase):
    """Test spectral feature extraction."""

    def setUp(self):
        self.sg = SpectralGenerator(freq_min=0, freq_max=500, n_points=1000)

    def test_zero_spectrum(self):
        features = self.sg.extract_spectral_features(np.zeros(1000))
        self.assertEqual(features["integral"], 0.0)
        self.assertEqual(features["peak_height"], 0.0)
        self.assertEqual(features["num_peaks"], 0)

    def test_single_peak_spectrum(self):
        spectrum = np.zeros(1000)
        spectrum[500] = 10.0  # Single peak at index 500
        features = self.sg.extract_spectral_features(spectrum)
        self.assertGreater(features["peak_height"], 0)
        self.assertEqual(features["num_peaks"], 1)

    def test_feature_keys(self):
        spectrum = np.random.rand(1000)
        features = self.sg.extract_spectral_features(spectrum)
        expected_keys = {"integral", "peak_height", "peak_frequency", "centroid",
                         "std_dev", "skewness", "kurtosis", "num_peaks"}
        self.assertEqual(set(features.keys()), expected_keys)


class TestSpectralCorrelation(unittest.TestCase):
    """Test spectral correlation computation."""

    def setUp(self):
        self.sg = SpectralGenerator(freq_min=0, freq_max=500, n_points=1000)

    def test_identical_spectra(self):
        spectrum = np.random.rand(1000)
        corr = self.sg.compute_spectral_correlation(spectrum, spectrum)
        self.assertAlmostEqual(corr, 1.0, places=5)

    def test_orthogonal_spectra(self):
        s1 = np.zeros(1000)
        s2 = np.zeros(1000)
        s1[:500] = 1.0
        s2[500:] = 1.0
        corr = self.sg.compute_spectral_correlation(s1, s2)
        self.assertAlmostEqual(corr, 0.0, places=5)

    def test_zero_spectrum(self):
        corr = self.sg.compute_spectral_correlation(np.zeros(1000), np.ones(1000))
        self.assertEqual(corr, 0.0)


class TestDeltaSpectralFeatures(unittest.TestCase):
    """Test delta feature computation."""

    def setUp(self):
        self.sg = SpectralGenerator(freq_min=0, freq_max=500, n_points=1000)
        self.dsf = DeltaSpectralFeatures(self.sg)

    def test_identical_spectra_zero_delta(self):
        spectrum = np.random.rand(1000)
        delta = self.dsf.compute_delta_features(spectrum, spectrum, 1.0, 1.0)
        self.assertAlmostEqual(delta["delta_integral"], 0.0, places=5)
        self.assertAlmostEqual(delta["delta_entropy_j_mol_k"], 0.0, places=5)
        self.assertAlmostEqual(delta["spectral_l2_norm"], 0.0, places=5)
        self.assertAlmostEqual(delta["spectral_correlation"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
