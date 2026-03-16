"""Cost-sensitive threshold optimisation to minimise weighted FN/FP costs.

Searches over a fine grid of thresholds to find the one that minimises
a user-defined cost function weighting false negatives and false positives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_N_THRESHOLDS: int = 1000


@dataclass
class HardeningResult:
    """Result of cost-sensitive threshold optimisation.

    Args:
        optimal_threshold: Threshold minimising the cost function.
        min_cost: Minimum cost value achieved.
        fn_rate_at_opt: False-negative rate at the optimal threshold.
        fp_rate_at_opt: False-positive rate at the optimal threshold.
        cost_curve: List of ``(threshold, cost)`` tuples across the grid.

    Returns:
        A ``HardeningResult`` dataclass instance.

    Example:
        >>> hr = HardeningResult(0.4, 0.12, 0.02, 0.10, [(0.0, 1.0), (0.5, 0.12)])
        >>> hr.optimal_threshold
        0.4
    """

    optimal_threshold: float
    min_cost: float
    fn_rate_at_opt: float
    fp_rate_at_opt: float
    cost_curve: list[tuple[float, float]]


class FalseNegativeHardening:
    """Cost-sensitive threshold optimiser for binary classifiers.

    Minimises ``C(t) = fn_cost * FN_rate(t) + fp_cost * FP_rate(t)`` over
    a grid of 1000 thresholds.

    Args:
        fn_cost: Cost multiplier for false negatives.
        fp_cost: Cost multiplier for false positives.

    Returns:
        A ``FalseNegativeHardening`` instance.

    Example:
        >>> import numpy as np
        >>> fnh = FalseNegativeHardening(fn_cost=10.0, fp_cost=1.0)
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
        >>> result = fnh.optimize(y_true, y_prob)
        >>> 0.0 <= result.optimal_threshold <= 1.0
        True
    """

    def __init__(self, fn_cost: float = 1.0, fp_cost: float = 1.0) -> None:
        self._fn_cost: float = fn_cost
        self._fp_cost: float = fp_cost
        self._optimal_threshold: float | None = None
        logger.info(
            "FalseNegativeHardening initialised (fn_cost=%.3f, fp_cost=%.3f)",
            fn_cost,
            fp_cost,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, y_true: np.ndarray, y_prob: np.ndarray) -> HardeningResult:
        """Find the threshold minimising the weighted FN/FP cost.

        Evaluates ``C(t) = fn_cost * FN_rate(t) + fp_cost * FP_rate(t)``
        over 1000 linearly spaced thresholds in ``[0, 1]``.

        Args:
            y_true: Binary ground-truth labels of shape ``(N,)``.
            y_prob: Predicted probabilities of shape ``(N,)``.

        Returns:
            A :class:`HardeningResult` with the optimal threshold, cost,
            rates, and the full cost curve.

        Example:
            >>> fnh = FalseNegativeHardening(fn_cost=5.0, fp_cost=1.0)
            >>> y_true = np.array([0, 0, 0, 1, 1, 1])
            >>> y_prob = np.array([0.1, 0.2, 0.6, 0.5, 0.8, 0.9])
            >>> result = fnh.optimize(y_true, y_prob)
            >>> result.min_cost <= 5.0  # bounded by worst case
            True
        """
        thresholds: np.ndarray = np.linspace(0.0, 1.0, _N_THRESHOLDS)
        n_pos: int = int(np.sum(y_true == 1))
        n_neg: int = int(np.sum(y_true == 0))

        cost_curve: list[tuple[float, float]] = []
        best_threshold: float = 0.5
        best_cost: float = np.inf
        best_fn_rate: float = 0.0
        best_fp_rate: float = 0.0

        for t in thresholds:
            y_pred: np.ndarray = (y_prob >= t).astype(int)

            fn_rate: float = float(np.sum((y_pred == 0) & (y_true == 1))) / n_pos if n_pos > 0 else 0.0
            fp_rate: float = float(np.sum((y_pred == 1) & (y_true == 0))) / n_neg if n_neg > 0 else 0.0

            cost: float = self._fn_cost * fn_rate + self._fp_cost * fp_rate
            cost_curve.append((float(t), cost))

            if cost < best_cost:
                best_cost = cost
                best_threshold = float(t)
                best_fn_rate = fn_rate
                best_fp_rate = fp_rate

        self._optimal_threshold = best_threshold
        logger.info(
            "Optimal threshold=%.4f, cost=%.4f (fn_rate=%.4f, fp_rate=%.4f)",
            best_threshold,
            best_cost,
            best_fn_rate,
            best_fp_rate,
        )

        return HardeningResult(
            optimal_threshold=best_threshold,
            min_cost=best_cost,
            fn_rate_at_opt=best_fn_rate,
            fp_rate_at_opt=best_fp_rate,
            cost_curve=cost_curve,
        )

    def apply(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply the optimised threshold to produce binary predictions.

        Args:
            y_prob: Predicted probabilities of shape ``(N,)``.

        Returns:
            Binary predictions of shape ``(N,)`` as integers.

        Example:
            >>> fnh = FalseNegativeHardening()
            >>> y_true = np.array([0, 1])
            >>> y_prob = np.array([0.3, 0.7])
            >>> _ = fnh.optimize(y_true, y_prob)
            >>> preds = fnh.apply(np.array([0.2, 0.8]))
            >>> preds.dtype
            dtype('int64')
        """
        if self._optimal_threshold is None:
            raise RuntimeError("Call optimize() before apply().")
        return (y_prob >= self._optimal_threshold).astype(int)
