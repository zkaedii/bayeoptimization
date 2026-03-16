"""Evidential deep learning for uncertainty-aware classification and regression.

This package provides Dirichlet-based (classification) and Normal-Inverse-Gamma
(regression) evidential models with full uncertainty decomposition, multimodal
fusion, and open-set recognition.
"""

from .classifier import EvidentialClassifier
from .fusion import FusionPhase, MultimodalFusion
from .openset import OpenSetRecognition
from .regressor import EvidentialRegressor

__all__: list[str] = [
    "EvidentialClassifier",
    "EvidentialRegressor",
    "FusionPhase",
    "MultimodalFusion",
    "OpenSetRecognition",
]
