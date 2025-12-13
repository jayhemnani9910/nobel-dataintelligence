"""
Quantum Data Decoder - Vibrational Analysis for Protein Function Prediction

A comprehensive framework for predicting protein stability and function
through multimodal deep learning on structural and spectral data.
"""

__version__ = "0.1.0"
__author__ = "Quantum Data Decoder Team"

from . import data_acquisition
from . import nma_analysis
from . import spectral_generation
from . import utils
from . import datasets
from . import training
from . import models

__all__ = [
    'data_acquisition',
    'nma_analysis',
    'spectral_generation',
    'utils',
    'datasets',
    'training',
    'models',
]
