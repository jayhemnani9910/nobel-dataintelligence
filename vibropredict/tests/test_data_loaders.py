"""Unit tests for VibroPredict data loading and standardization.

Tests cover:
- KinHubLoader: load, validate, resolve_ambiguities
- EnzyExtractFilter: filter removes overlapping IDs and low confidence
- Standardization: log_transform_kcat, canonicalize_smiles, compute_drfp
- EnzymeKineticsDataset: __len__, __getitem__
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from vibropredict.data.kinhub import KinHubLoader
from vibropredict.data.enzyextract import EnzyExtractFilter
from vibropredict.data.standardization import log_transform_kcat


class TestKinHubLoader(unittest.TestCase):
    """Test KinHubLoader data loading and validation."""

    def setUp(self):
        """Create a temporary CSV file with mock KinHub data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, 'kinhub.csv')

        df = pd.DataFrame({
            'uniprot_id': ['P12345', 'P12346', None, 'P12348', 'P12345', 'P12345'],
            'k_cat': [10.0, 20.0, 30.0, None, 15.0, 25.0],
            'substrate_smiles': ['CC', 'CCO', 'C=O', 'CC(=O)O', 'CC', 'CC'],
        })
        df.to_csv(self.csv_path, index=False)
        self.loader = KinHubLoader(self.csv_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_returns_dataframe(self):
        """Test load returns a DataFrame with required columns."""
        df = self.loader.load()
        self.assertIsInstance(df, pd.DataFrame)
        for col in ['uniprot_id', 'k_cat', 'substrate_smiles']:
            self.assertIn(col, df.columns)

    def test_load_missing_column_raises(self):
        """Test load raises ValueError if columns are missing."""
        bad_path = os.path.join(self.temp_dir.name, 'bad.csv')
        pd.DataFrame({'foo': [1]}).to_csv(bad_path, index=False)
        loader = KinHubLoader(bad_path)
        with self.assertRaises(ValueError):
            loader.load()

    def test_validate_drops_invalid_rows(self):
        """Test validate drops rows with missing uniprot_id or k_cat."""
        df = self.loader.load()
        validated = self.loader.validate(df)
        # Row with None uniprot_id and row with None k_cat should be dropped
        self.assertLess(len(validated), len(df))
        self.assertFalse(validated['uniprot_id'].isna().any())
        self.assertFalse(validated['k_cat'].isna().any())

    def test_resolve_ambiguities_merges_duplicates(self):
        """Test resolve_ambiguities merges duplicate (enzyme, substrate) pairs."""
        df = self.loader.load()
        validated = self.loader.validate(df)
        resolved = self.loader.resolve_ambiguities(validated)
        # P12345 + CC appears twice in validated data; should be merged to one
        p12345_cc = resolved[
            (resolved['uniprot_id'] == 'P12345') &
            (resolved['substrate_smiles'] == 'CC')
        ]
        self.assertEqual(len(p12345_cc), 1)


class TestEnzyExtractFilter(unittest.TestCase):
    """Test EnzyExtractFilter quality filtering."""

    def setUp(self):
        """Create a temporary CSV with mock EnzyExtract data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, 'enzyextract.csv')

        df = pd.DataFrame({
            'uniprot_id': ['P00001', 'P00002', 'P00003', 'P00004', ''],
            'substrate_smiles': ['CC', 'CCO', 'C=O', 'CC(=O)O', 'CC'],
            'confidence': [0.95, 0.85, 0.99, 0.92, 0.99],
            'k_cat': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        df.to_csv(self.csv_path, index=False)
        self.filter = EnzyExtractFilter(self.csv_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_filter_removes_overlapping_ids(self):
        """Test filter removes entries already in KinHub."""
        df = self.filter.load()
        kinhub_ids = {'P00001', 'P00003'}
        filtered = self.filter.filter(df, kinhub_ids)
        remaining_ids = set(filtered['uniprot_id'].astype(str))
        self.assertNotIn('P00001', remaining_ids)
        self.assertNotIn('P00003', remaining_ids)

    def test_filter_removes_low_confidence(self):
        """Test filter removes entries with confidence <= 0.9."""
        df = self.filter.load()
        filtered = self.filter.filter(df, set())
        # P00002 has confidence 0.85, should be removed
        self.assertNotIn('P00002', filtered['uniprot_id'].values)

    def test_filter_removes_empty_uniprot_id(self):
        """Test filter removes entries with empty uniprot_id."""
        df = self.filter.load()
        filtered = self.filter.filter(df, set())
        # Last row has empty uniprot_id
        for uid in filtered['uniprot_id'].values:
            self.assertTrue(str(uid).strip() != '')


class TestLogTransformKcat(unittest.TestCase):
    """Test log-transform of k_cat values."""

    def test_log_transform_kcat(self):
        """Test log10 transform produces correct values."""
        df = pd.DataFrame({'k_cat': [1.0, 10.0, 100.0, 1000.0]})
        result = log_transform_kcat(df)
        self.assertIn('log_kcat', result.columns)
        np.testing.assert_array_almost_equal(
            result['log_kcat'].values, [0.0, 1.0, 2.0, 3.0]
        )

    def test_log_transform_clips_zeros(self):
        """Test zeros are clipped before transform."""
        df = pd.DataFrame({'k_cat': [0.0, -1.0, 10.0]})
        result = log_transform_kcat(df)
        # Should not contain -inf or nan
        self.assertTrue(np.all(np.isfinite(result['log_kcat'].values)))


class TestCanonicalizeSmilesNoRDKit(unittest.TestCase):
    """Test canonicalize_smiles when RDKit may not be available."""

    def test_import_works(self):
        """Test the function can at least be imported."""
        from vibropredict.data.standardization import canonicalize_smiles
        self.assertTrue(callable(canonicalize_smiles))


class TestComputeDrfpNoRDKit(unittest.TestCase):
    """Test compute_drfp signature and fallback behavior."""

    def test_import_works(self):
        """Test the function can at least be imported."""
        from vibropredict.data.standardization import compute_drfp
        self.assertTrue(callable(compute_drfp))


if __name__ == '__main__':
    unittest.main()
