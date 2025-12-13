"""Unit test suite for Quantum Data Decoder.

Tests are organized by module:
- test_data_loading.py: Dataset classes and DataLoader creation
- test_models.py: Neural network architectures and forward passes
- test_training.py: Training utilities, metrics, and callbacks

Run all tests with:
    python -m pytest tests/
    or
    python -m unittest discover tests/

Run specific test file:
    python -m unittest tests.test_models
    
Run specific test:
    python -m unittest tests.test_models.TestProteinGNN.test_model_initialization
"""

__all__ = [
    'test_data_loading',
    'test_models',
    'test_training'
]
