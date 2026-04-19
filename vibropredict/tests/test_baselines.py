"""
Tests for the baseline evaluation harness.

Verifies:
- BaselineModel ABC contract is enforced.
- Stub (mean_predictor) baseline works end-to-end.
- Registry discovers registered baselines.
- CatPred is registered (even if unavailable).
"""

from __future__ import annotations

import pytest

from vibropredict.evaluation.baselines.base import (
    _BASELINE_REGISTRY,
    BaselineModel,
    get_baseline,
    list_baselines,
    register_baseline,
)


class TestBaselineABC:
    """Verify the ABC contract."""

    def test_cannot_instantiate_abc(self):
        """BaselineModel should not be instantiable directly."""
        with pytest.raises(TypeError):
            BaselineModel()

    def test_subclass_must_implement_predict(self):
        """Subclass without predict() raises TypeError."""

        class BadBaseline(BaselineModel):
            pass

        with pytest.raises(TypeError):
            BadBaseline()

    def test_valid_subclass(self):
        """Subclass with predict() can be instantiated."""

        class GoodBaseline(BaselineModel):
            def predict(self, sequence, smiles):
                return 0.0, {}

        model = GoodBaseline()
        log_kcat, meta = model.predict("ACDEF", "CC")
        assert log_kcat == 0.0
        assert isinstance(meta, dict)


class TestRegistry:
    """Verify the decorator-based registry."""

    def test_register_and_get(self):
        """register_baseline + get_baseline round-trip works."""

        @register_baseline("_test_model")
        class TestModel(BaselineModel):
            def predict(self, sequence, smiles):
                return 42.0, {"test": True}

        model = get_baseline("_test_model")
        assert isinstance(model, BaselineModel)
        log_kcat, meta = model.predict("ACDEF", "CC")
        assert log_kcat == 42.0

        # Cleanup
        _BASELINE_REGISTRY.pop("_test_model", None)

    def test_get_unknown_raises(self):
        """get_baseline with unknown name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown baseline"):
            get_baseline("nonexistent_model_xyz")

    def test_register_non_subclass_raises(self):
        """Registering a non-BaselineModel class raises TypeError."""
        with pytest.raises(TypeError, match="must subclass BaselineModel"):

            @register_baseline("_bad")
            class NotABaseline:
                pass

    def test_list_baselines(self):
        """list_baselines returns a sorted list of registered names."""
        names = list_baselines()
        assert isinstance(names, list)
        assert names == sorted(names)


class TestStubBaseline:
    """Verify the mean_predictor stub baseline."""

    def test_stub_registered(self):
        """mean_predictor should be in the registry."""
        assert "mean_predictor" in _BASELINE_REGISTRY

    def test_stub_predict(self):
        """mean_predictor returns its configured mean."""
        model = get_baseline("mean_predictor")
        log_kcat, meta = model.predict("ACDEF", "CC(=O)O")
        assert log_kcat == 1.5
        assert meta["method"] == "mean_predictor"

    def test_stub_custom_mean(self):
        """mean_predictor accepts a custom mean_value."""
        model = get_baseline("mean_predictor", mean_value=3.14)
        log_kcat, _ = model.predict("ACDEF", "CC")
        assert log_kcat == 3.14

    def test_stub_batch_predict(self):
        """predict_batch should work with the default loop implementation."""
        model = get_baseline("mean_predictor")
        results = model.predict_batch(
            sequences=["ACDEF", "GHIKL"],
            smiles_list=["CC", "OO"],
        )
        assert len(results) == 2
        assert all(r[0] == 1.5 for r in results)


class TestCatPredBaseline:
    """Verify CatPred baseline registration and behavior."""

    def test_catpred_registered(self):
        """catpred should be in the registry."""
        assert "catpred" in _BASELINE_REGISTRY

    def test_catpred_instantiation(self):
        """CatPred baseline should instantiate without errors."""
        model = get_baseline("catpred")
        assert isinstance(model, BaselineModel)
        assert model.name == "CatPred"

    def test_catpred_unavailable_raises_runtime_error(self):
        """predict() raises RuntimeError when CatPred is not installed."""
        model = get_baseline("catpred")
        model._available = "unavailable"  # Force unavailable state
        with pytest.raises(RuntimeError, match="CatPred is not available"):
            model.predict("ACDEF", "CC")
