"""Confusion-matrix aware threshold calibration under FP/FN budgets.

Finds a classification threshold that jointly respects false-positive and
false-negative rate budgets, with safety-critical priority given to the
false-negative constraint.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_N_THRESHOLDS: int = 1000


class BudgetViolationError(Exception):
    """Raised when an audit detects that FP or FN budgets are exceeded.

    Example:
        >>> raise BudgetViolationError("FN rate 0.05 exceeds budget 0.01")
        Traceback (most recent call last):
        ...
        src.security.confusion_defense.BudgetViolationError: FN rate 0.05 exceeds budget 0.01
    """


class ConfusionMatrixDefense:
    """Threshold calibration respecting false-positive and false-negative budgets.

    During :meth:`calibrate`, a grid of 1000 candidate thresholds is
    evaluated.  The threshold that satisfies *both* budgets is selected.
    If no threshold satisfies both, the FN budget is prioritised (safety-
    critical) and a warning is logged.

    Args:
        fp_budget: Maximum tolerated false-positive rate.
        fn_budget: Maximum tolerated false-negative rate.

    Returns:
        A ``ConfusionMatrixDefense`` instance.

    Example:
        >>> import numpy as np
        >>> cmd = ConfusionMatrixDefense(fp_budget=0.05, fn_budget=0.01)
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
        >>> threshold = cmd.calibrate(y_true, y_prob)
        >>> 0.0 <= threshold <= 1.0
        True
    """

    def __init__(self, fp_budget: float = 0.05, fn_budget: float = 0.01) -> None:
        self._fp_budget: float = fp_budget
        self._fn_budget: float = fn_budget
        self._threshold: float | None = None
        logger.info(
            "ConfusionMatrixDefense initialised (fp_budget=%.3f, fn_budget=%.3f)",
            fp_budget,
            fn_budget,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find the optimal threshold satisfying FP and FN budgets.

        Performs a grid search over 1000 linearly spaced thresholds in
        ``[0, 1]``.  If no threshold satisfies both budgets, the FN budget
        is prioritised and the best threshold under that constraint is
        returned.

        Args:
            y_true: Binary ground-truth labels of shape ``(N,)``.
            y_prob: Predicted probabilities of shape ``(N,)``.

        Returns:
            The calibrated threshold.

        Example:
            >>> cmd = ConfusionMatrixDefense(fp_budget=0.10, fn_budget=0.05)
            >>> y_true = np.array([0, 0, 0, 1, 1, 1])
            >>> y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
            >>> t = cmd.calibrate(y_true, y_prob)
            >>> isinstance(t, float)
            True
        """
        thresholds: np.ndarray = np.linspace(0.0, 1.0, _N_THRESHOLDS)
        n_pos: int = int(np.sum(y_true == 1))
        n_neg: int = int(np.sum(y_true == 0))

        best_both: float | None = None
        best_both_score: float = np.inf

        best_fn_only: float | None = None
        best_fn_only_score: float = np.inf

        for t in thresholds:
            y_pred: np.ndarray = (y_prob >= t).astype(int)

            fp_rate: float = float(np.sum((y_pred == 1) & (y_true == 0))) / n_neg if n_neg > 0 else 0.0
            fn_rate: float = float(np.sum((y_pred == 0) & (y_true == 1))) / n_pos if n_pos > 0 else 0.0

            fn_ok: bool = fn_rate <= self._fn_budget
            fp_ok: bool = fp_rate <= self._fp_budget

            # Combined cost: sum of rates (lower is better)
            cost: float = fp_rate + fn_rate

            if fn_ok and fp_ok and cost < best_both_score:
                best_both = float(t)
                best_both_score = cost

            if fn_ok and cost < best_fn_only_score:
                best_fn_only = float(t)
                best_fn_only_score = cost

        if best_both is not None:
            self._threshold = best_both
            logger.info("Calibrated threshold=%.4f (both budgets met).", self._threshold)
        elif best_fn_only is not None:
            self._threshold = best_fn_only
            logger.warning(
                "No threshold satisfies both budgets. Prioritising FN budget. "
                "Threshold=%.4f",
                self._threshold,
            )
        else:
            # Fallback: pick threshold minimising FN rate
            fn_rates: list[float] = []
            for t in thresholds:
                y_pred = (y_prob >= t).astype(int)
                fn_r: float = float(np.sum((y_pred == 0) & (y_true == 1))) / n_pos if n_pos > 0 else 0.0
                fn_rates.append(fn_r)
            self._threshold = float(thresholds[int(np.argmin(fn_rates))])
            logger.warning(
                "FN budget infeasible. Using lowest-FN threshold=%.4f",
                self._threshold,
            )

        return self._threshold

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply the calibrated threshold to produce binary predictions.

        Args:
            y_prob: Predicted probabilities of shape ``(N,)``.

        Returns:
            Binary predictions of shape ``(N,)``.

        Example:
            >>> cmd = ConfusionMatrixDefense()
            >>> y_true = np.array([0, 1, 1])
            >>> y_prob = np.array([0.2, 0.7, 0.9])
            >>> _ = cmd.calibrate(y_true, y_prob)
            >>> preds = cmd.predict(np.array([0.3, 0.8]))
            >>> preds.dtype
            dtype('int64')
        """
        if self._threshold is None:
            raise RuntimeError("Call calibrate() before predict().")
        return (y_prob >= self._threshold).astype(int)

    def audit(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        """Audit predictions against the configured FP/FN budgets.

        Args:
            y_true: Binary ground-truth labels of shape ``(N,)``.
            y_pred: Binary predictions of shape ``(N,)``.

        Returns:
            A dictionary with keys ``fp_rate``, ``fn_rate``,
            ``fp_compliant``, ``fn_compliant``, and ``threshold``.

        Raises:
            BudgetViolationError: If either the FP or FN budget is exceeded.

        Example:
            >>> cmd = ConfusionMatrixDefense(fp_budget=0.5, fn_budget=0.5)
            >>> y_true = np.array([0, 0, 1, 1])
            >>> y_prob = np.array([0.1, 0.4, 0.6, 0.9])
            >>> _ = cmd.calibrate(y_true, y_prob)
            >>> preds = cmd.predict(y_prob)
            >>> result = cmd.audit(y_true, preds)
            >>> result["fp_compliant"]
            True
        """
        if self._threshold is None:
            raise RuntimeError("Call calibrate() before audit().")

        n_pos: int = int(np.sum(y_true == 1))
        n_neg: int = int(np.sum(y_true == 0))

        fp_rate: float = float(np.sum((y_pred == 1) & (y_true == 0))) / n_neg if n_neg > 0 else 0.0
        fn_rate: float = float(np.sum((y_pred == 0) & (y_true == 1))) / n_pos if n_pos > 0 else 0.0

        fp_compliant: bool = fp_rate <= self._fp_budget
        fn_compliant: bool = fn_rate <= self._fn_budget

        result: dict[str, Any] = {
            "fp_rate": fp_rate,
            "fn_rate": fn_rate,
            "fp_compliant": fp_compliant,
            "fn_compliant": fn_compliant,
            "threshold": self._threshold,
        }

        if not fp_compliant or not fn_compliant:
            violations: list[str] = []
            if not fp_compliant:
                violations.append(f"FP rate {fp_rate:.4f} exceeds budget {self._fp_budget:.4f}")
            if not fn_compliant:
                violations.append(f"FN rate {fn_rate:.4f} exceeds budget {self._fn_budget:.4f}")
            msg: str = "; ".join(violations)
            logger.error("Budget violation: %s", msg)
            raise BudgetViolationError(msg)

        logger.info("Audit passed: fp_rate=%.4f, fn_rate=%.4f", fp_rate, fn_rate)
        return result
