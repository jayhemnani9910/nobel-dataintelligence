"""
Model architectures for Quantum Data Decoder
"""

from .gnn import ProteinGNN, GraphConstruction
from .cnn import SpectralCNN, SpectralFeatureExtractor, MultiScaleSpectralCNN
from .multimodal import VibroStructuralModel, VibroStructuralFusion
from .losses import (
    MarginRankingLossCustom,
    SpearmanCorrelationLoss,
    FocalLoss,
    ContrastiveLoss,
    WeightedBCELoss,
    CombinedLoss,
)

__all__ = [
    'ProteinGNN',
    'GraphConstruction',
    'SpectralCNN',
    'SpectralFeatureExtractor',
    'MultiScaleSpectralCNN',
    'VibroStructuralModel',
    'VibroStructuralFusion',
    'MarginRankingLossCustom',
    'SpearmanCorrelationLoss',
    'FocalLoss',
    'ContrastiveLoss',
    'WeightedBCELoss',
    'CombinedLoss',
]
