"""Active learning and distribution drift detection components.

Exports:
    ActiveLearning: Uncertainty-diversity balanced batch selection.
    DriftDetector: KL-divergence based distribution drift detection.
    DriftReport: Dataclass summarising drift detector state.
"""

from src.active.learner import ActiveLearning
from src.active.drift import DriftDetector, DriftReport

__all__: list[str] = ["ActiveLearning", "DriftDetector", "DriftReport"]
