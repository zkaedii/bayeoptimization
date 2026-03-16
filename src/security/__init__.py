"""Security hardening: adversarial robustness and confusion-matrix defenses.

Exports:
    AdversarialRobustness: FGSM attack generation and Hamiltonian defense.
    RobustnessReport: Dataclass for adversarial evaluation results.
    ConfusionMatrixDefense: Threshold calibration under FP/FN budgets.
    BudgetViolationError: Raised when confusion-matrix budgets are exceeded.
    FalseNegativeHardening: Cost-sensitive threshold optimisation.
    HardeningResult: Dataclass for hardening optimisation results.
"""

from src.security.adversarial import AdversarialRobustness, RobustnessReport
from src.security.confusion_defense import ConfusionMatrixDefense, BudgetViolationError
from src.security.fn_hardening import FalseNegativeHardening, HardeningResult

__all__: list[str] = [
    "AdversarialRobustness",
    "RobustnessReport",
    "ConfusionMatrixDefense",
    "BudgetViolationError",
    "FalseNegativeHardening",
    "HardeningResult",
]
