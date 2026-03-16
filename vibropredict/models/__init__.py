"""
VibroPredict model architectures.

Provides multimodal encoders and fusion modules for enzyme kinetics
prediction from protein sequences, vibrational spectra, and chemical
representations.
"""

from vibropredict.models.sequence_encoder import ProtT5Encoder
from vibropredict.models.chemical_encoder import ChemicalEncoder
from vibropredict.models.fusion import TriModalFusion
from vibropredict.models.vibropredict_hybrid import VibroPredictHybrid

__all__ = [
    "ProtT5Encoder",
    "ChemicalEncoder",
    "TriModalFusion",
    "VibroPredictHybrid",
]
