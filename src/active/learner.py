"""Active learning with uncertainty-diversity batch selection.

This module provides a greedy sequential batch selection strategy that
balances model uncertainty with geometric diversity among the selected
points.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ActiveLearning:
    """Greedy batch-mode active learner balancing uncertainty and diversity.

    The scoring function for each candidate *x* is:

        score(x) = w_uncertainty * uncertainty(x)
                   + w_diversity  * diversity(x, selected)

    where *diversity* is the minimum Euclidean distance from *x* to every
    point already selected in the current batch.  The first point is chosen
    purely by uncertainty; subsequent picks use the combined score.

    Args:
        w_uncertainty: Weight applied to the uncertainty component.
        w_diversity: Weight applied to the diversity component.

    Returns:
        An ``ActiveLearning`` instance ready for batch selection.

    Example:
        >>> import numpy as np
        >>> al = ActiveLearning(w_uncertainty=0.6, w_diversity=0.4)
        >>> pool = np.random.randn(50, 3)
        >>> uncert = np.random.rand(50)
        >>> result = al.select_batch(pool, uncert, k=5)
        >>> len(result["selected_indices"])
        5
    """

    def __init__(self, w_uncertainty: float = 0.6, w_diversity: float = 0.4) -> None:
        if w_uncertainty < 0 or w_diversity < 0:
            raise ValueError("Weights must be non-negative.")
        self._w_uncertainty: float = w_uncertainty
        self._w_diversity: float = w_diversity
        self._query_history: list[dict[str, Any]] = []
        self._round: int = 0
        self._last_selection: dict[str, Any] | None = None
        logger.info(
            "ActiveLearning initialised with w_uncertainty=%.3f, w_diversity=%.3f",
            w_uncertainty,
            w_diversity,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_batch(
        self, pool: np.ndarray, uncertainties: np.ndarray, k: int
    ) -> dict[str, Any]:
        """Select a batch of *k* points from the unlabelled pool.

        Uses greedy sequential selection: the first point is selected by
        highest uncertainty alone; each subsequent point maximises a
        weighted combination of uncertainty and minimum Euclidean distance
        to already-selected points.

        Args:
            pool: Feature matrix of shape ``(N, D)``.
            uncertainties: Uncertainty scores of shape ``(N,)``.
            k: Number of points to select.

        Returns:
            A dictionary with keys ``selected_indices`` (list of int) and
            ``scores`` (list of float).

        Example:
            >>> al = ActiveLearning()
            >>> pool = np.array([[0, 0], [1, 1], [2, 2], [10, 10]])
            >>> uncert = np.array([0.1, 0.9, 0.5, 0.8])
            >>> result = al.select_batch(pool, uncert, k=2)
            >>> sorted(result["selected_indices"])  # doctest: +SKIP
            [1, 3]
        """
        n_pool: int = pool.shape[0]
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k > n_pool:
            raise ValueError(
                f"k={k} exceeds pool size {n_pool}."
            )
        if uncertainties.shape[0] != n_pool:
            raise ValueError(
                "uncertainties length must match pool size."
            )

        # Normalise uncertainties to [0, 1] for fair weighting.
        u_min: float = float(uncertainties.min())
        u_max: float = float(uncertainties.max())
        if u_max - u_min > 0:
            norm_uncert: np.ndarray = (uncertainties - u_min) / (u_max - u_min)
        else:
            norm_uncert = np.ones(n_pool, dtype=np.float64)

        selected_indices: list[int] = []
        selected_scores: list[float] = []
        remaining: set[int] = set(range(n_pool))

        # --- First pick: highest uncertainty ---
        first_idx: int = int(np.argmax(norm_uncert))
        selected_indices.append(first_idx)
        selected_scores.append(float(norm_uncert[first_idx]))
        remaining.discard(first_idx)

        # --- Subsequent picks ---
        for _ in range(1, k):
            best_idx: int = -1
            best_score: float = -np.inf
            remaining_list: list[int] = list(remaining)

            # Compute diversity: min Euclidean distance to selected set
            selected_points: np.ndarray = pool[selected_indices]  # (S, D)
            candidate_points: np.ndarray = pool[remaining_list]   # (R, D)

            # Pairwise distances: (R, S)
            diffs: np.ndarray = (
                candidate_points[:, np.newaxis, :] - selected_points[np.newaxis, :, :]
            )
            dists: np.ndarray = np.sqrt((diffs ** 2).sum(axis=2))  # (R, S)
            min_dists: np.ndarray = dists.min(axis=1)  # (R,)

            # Normalise diversity scores
            d_min: float = float(min_dists.min())
            d_max: float = float(min_dists.max())
            if d_max - d_min > 0:
                norm_div: np.ndarray = (min_dists - d_min) / (d_max - d_min)
            else:
                norm_div = np.ones(len(remaining_list), dtype=np.float64)

            # Compute combined scores
            uncert_vals: np.ndarray = norm_uncert[remaining_list]
            scores: np.ndarray = (
                self._w_uncertainty * uncert_vals + self._w_diversity * norm_div
            )

            winner: int = int(np.argmax(scores))
            best_idx = remaining_list[winner]
            best_score = float(scores[winner])

            selected_indices.append(best_idx)
            selected_scores.append(best_score)
            remaining.discard(best_idx)

        self._last_selection = {
            "selected_indices": selected_indices,
            "scores": selected_scores,
        }
        logger.info("Selected batch of %d from pool of %d.", k, n_pool)
        return {
            "selected_indices": selected_indices,
            "scores": selected_scores,
        }

    def step(self, new_labeled_indices: list[int]) -> None:
        """Record the latest selection round in query history.

        Should be called after the oracle has labelled the selected points.

        Args:
            new_labeled_indices: The indices that were labelled in this round.

        Returns:
            None

        Example:
            >>> al = ActiveLearning()
            >>> pool = np.random.randn(20, 2)
            >>> uncert = np.random.rand(20)
            >>> result = al.select_batch(pool, uncert, k=3)
            >>> al.step(result["selected_indices"])
            >>> len(al.get_query_history())
            1
        """
        self._round += 1
        scores: list[float] = (
            self._last_selection["scores"]
            if self._last_selection is not None
            else []
        )
        record: dict[str, Any] = {
            "round": self._round,
            "selected_indices": list(new_labeled_indices),
            "scores": scores,
        }
        self._query_history.append(record)
        logger.info("Recorded query round %d with %d selections.", self._round, len(new_labeled_indices))

    def get_query_history(self) -> list[dict[str, Any]]:
        """Return the full query history across all rounds.

        Returns:
            A list of dictionaries, each containing ``round``,
            ``selected_indices``, and ``scores``.

        Example:
            >>> al = ActiveLearning()
            >>> al.get_query_history()
            []
        """
        return list(self._query_history)
