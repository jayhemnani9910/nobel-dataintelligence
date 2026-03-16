"""Tests for shared utility functions."""

import unittest
import tempfile
from pathlib import Path

import numpy as np

from src.utils import (
    parse_fasta,
    encode_sequence,
    decode_sequence,
    normalize_spectrum,
    compute_sequence_properties,
)


class TestParseFasta(unittest.TestCase):
    """Test FASTA parsing utility."""

    def _write_fasta(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_single_sequence(self):
        path = self._write_fasta(">prot_1\nACDEFG\n")
        result = parse_fasta(path)
        self.assertEqual(result, {"prot_1": "ACDEFG"})

    def test_multi_line_sequence(self):
        path = self._write_fasta(">prot_1\nACDE\nFGHI\nKLMN\n")
        result = parse_fasta(path)
        self.assertEqual(result, {"prot_1": "ACDEFGHIKLMN"})

    def test_multiple_sequences(self):
        path = self._write_fasta(">a\nAAA\n>b\nBBB\n>c\nCCC\n")
        result = parse_fasta(path)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["a"], "AAA")
        self.assertEqual(result["c"], "CCC")

    def test_empty_file(self):
        path = self._write_fasta("")
        result = parse_fasta(path)
        self.assertEqual(result, {})

    def test_header_with_description(self):
        path = self._write_fasta(">prot_1 some description here\nACDEFG\n")
        result = parse_fasta(path)
        self.assertIn("prot_1", result)
        self.assertEqual(result["prot_1"], "ACDEFG")


class TestSequenceEncoding(unittest.TestCase):
    """Test amino acid encoding and decoding."""

    def test_round_trip(self):
        seq = "ACDEFGHIKLMNPQRSTVWY"
        encoded = encode_sequence(seq)
        decoded = decode_sequence(encoded)
        self.assertEqual(decoded, seq)

    def test_unknown_amino_acid(self):
        seq = "AJ"  # J is not a standard amino acid
        encoded = encode_sequence(seq)
        # J should map to X (index 20)
        self.assertEqual(encoded[1, 20], 1.0)

    def test_encoding_shape(self):
        seq = "ACDEF"
        encoded = encode_sequence(seq)
        self.assertEqual(encoded.shape, (5, 22))

    def test_one_hot_sums_to_one(self):
        seq = "ACDEF"
        encoded = encode_sequence(seq)
        row_sums = encoded.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(5))


class TestNormalizeSpectrum(unittest.TestCase):
    """Test spectrum normalization methods."""

    def test_max_normalization(self):
        spectrum = np.array([1.0, 3.0, 2.0])
        result = normalize_spectrum(spectrum, method="max")
        self.assertAlmostEqual(np.max(np.abs(result)), 1.0)

    def test_l2_normalization(self):
        spectrum = np.array([3.0, 4.0])
        result = normalize_spectrum(spectrum, method="l2")
        self.assertAlmostEqual(np.linalg.norm(result), 1.0)

    def test_zscore_normalization(self):
        spectrum = np.random.randn(100)
        result = normalize_spectrum(spectrum, method="zscore")
        self.assertAlmostEqual(np.mean(result), 0.0, places=5)
        self.assertAlmostEqual(np.std(result), 1.0, places=5)

    def test_integral_normalization(self):
        spectrum = np.array([2.0, 3.0, 5.0])
        result = normalize_spectrum(spectrum, method="integral")
        self.assertAlmostEqual(np.sum(result), 1.0)

    def test_zero_spectrum_returns_unchanged(self):
        spectrum = np.zeros(10)
        for method in ("max", "l2", "integral", "zscore"):
            result = normalize_spectrum(spectrum, method=method)
            np.testing.assert_array_equal(result, spectrum)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            normalize_spectrum(np.ones(10), method="invalid")


class TestComputeSequenceProperties(unittest.TestCase):
    """Test physicochemical property computation."""

    def test_basic_properties(self):
        props = compute_sequence_properties("ACDE")
        self.assertEqual(props["length"], 4)
        self.assertIn("mean_hydrophobicity", props)
        self.assertIn("charge_positive", props)

    def test_empty_sequence(self):
        props = compute_sequence_properties("")
        self.assertEqual(props["length"], 0)

    def test_single_amino_acid(self):
        props = compute_sequence_properties("K")
        self.assertEqual(props["length"], 1)
        self.assertEqual(props["charge_positive"], 1)


if __name__ == "__main__":
    unittest.main()
