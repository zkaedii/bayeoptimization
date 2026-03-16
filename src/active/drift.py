"""Distribution drift detection via sliding-window KL divergence.

Monitors incoming data against a frozen reference distribution and
classifies drift severity into three phases: STABLE, DRIFT, CRITICAL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_EPSILON: float = 1e-10


@dataclass
class DriftReport:
    """Summary produced by :meth:`DriftDetector.update`.

    Args:
        phase: Current drift phase — ``"STABLE"``, ``"DRIFT"``, or ``"CRITICAL"``.
        kl_score: Average KL divergence across feature dimensions.
        samples_seen: Total number of samples ingested so far.
        window_kl_history: List of KL scores from all update calls.
        triggered_at: Sample count at which CRITICAL was first triggered,
            or ``None`` if never triggered.

    Returns:
        A ``DriftReport`` dataclass instance.

    Example:
        >>> report = DriftReport("STABLE", 0.02, 500, [0.01, 0.02], None)
        >>> report.phase
        'STABLE'
    """

    phase: str
    kl_score: float
    samples_seen: int
    window_kl_history: list[float]
    triggered_at: int | None


class DriftDetector:
    """Three-phase drift detector using binned KL divergence.

    Maintains a frozen *reference window* and a sliding *current window*.
    On each :meth:`update` call the KL divergence between the two
    distributions is computed per feature dimension and averaged.

    Phase transitions:

    * **STABLE** — KL < ``threshold_drift``
    * **DRIFT**  — ``threshold_drift`` <= KL < ``threshold_critical``
    * **CRITICAL** — KL >= ``threshold_critical``; sets :attr:`retrain_signal`

    Args:
        window_size: Number of samples kept in each window.
        threshold_drift: KL threshold to enter DRIFT phase.
        threshold_critical: KL threshold to enter CRITICAL phase.
        n_bins: Number of histogram bins per feature dimension.

    Returns:
        A ``DriftDetector`` instance.

    Example:
        >>> import numpy as np
        >>> dd = DriftDetector(window_size=200)
        >>> dd.set_reference(np.random.randn(200, 3))
        >>> report = dd.update(np.random.randn(50, 3))
        >>> report.phase
        'STABLE'
    """

    def __init__(
        self,
        window_size: int = 500,
        threshold_drift: float = 0.1,
        threshold_critical: float = 0.3,
        n_bins: int = 10,
    ) -> None:
        self._window_size: int = window_size
        self._threshold_drift: float = threshold_drift
        self._threshold_critical: float = threshold_critical
        self._n_bins: int = n_bins

        self._reference_window: np.ndarray | None = None
        self._current_window: np.ndarray | None = None
        self._samples_seen: int = 0
        self._phase: str = "STABLE"
        self._retrain_signal: bool = False
        self._triggered_at: int | None = None
        self._kl_history: list[float] = []

        logger.info(
            "DriftDetector initialised (window=%d, drift=%.3f, critical=%.3f, bins=%d)",
            window_size,
            threshold_drift,
            threshold_critical,
            n_bins,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_reference(self, data: np.ndarray) -> None:
        """Freeze *data* as the reference distribution.

        Args:
            data: Array of shape ``(N, D)`` representing baseline data.

        Returns:
            None

        Example:
            >>> dd = DriftDetector()
            >>> dd.set_reference(np.random.randn(500, 4))
        """
        self._reference_window = data[-self._window_size :].copy()
        self._current_window = None
        self._samples_seen = 0
        self._phase = "STABLE"
        self._retrain_signal = False
        self._triggered_at = None
        self._kl_history = []
        logger.info(
            "Reference window set with %d samples, %d features.",
            self._reference_window.shape[0],
            self._reference_window.shape[1],
        )

    def update(self, x_batch: np.ndarray) -> DriftReport:
        """Ingest a batch of new observations and compute drift.

        Args:
            x_batch: New data of shape ``(B, D)``.

        Returns:
            A :class:`DriftReport` summarising the detector state.

        Example:
            >>> dd = DriftDetector(window_size=100)
            >>> dd.set_reference(np.random.randn(100, 2))
            >>> report = dd.update(np.random.randn(30, 2))
            >>> isinstance(report.kl_score, float)
            True
        """
        if self._reference_window is None:
            raise RuntimeError("Call set_reference() before update().")

        # Append to current window (sliding)
        if self._current_window is None:
            self._current_window = x_batch.copy()
        else:
            self._current_window = np.concatenate(
                [self._current_window, x_batch], axis=0
            )

        # Trim to window size (keep most recent)
        if self._current_window.shape[0] > self._window_size:
            self._current_window = self._current_window[-self._window_size :]

        self._samples_seen += x_batch.shape[0]

        # Compute KL divergence
        kl: float = self._compute_kl(self._reference_window, self._current_window)
        self._kl_history.append(kl)

        # Phase transition
        if kl >= self._threshold_critical:
            self._phase = "CRITICAL"
            self._retrain_signal = True
            if self._triggered_at is None:
                self._triggered_at = self._samples_seen
            logger.warning("CRITICAL drift detected: KL=%.4f at sample %d", kl, self._samples_seen)
        elif kl >= self._threshold_drift:
            self._phase = "DRIFT"
            logger.info("DRIFT detected: KL=%.4f", kl)
        else:
            self._phase = "STABLE"

        return DriftReport(
            phase=self._phase,
            kl_score=kl,
            samples_seen=self._samples_seen,
            window_kl_history=list(self._kl_history),
            triggered_at=self._triggered_at,
        )

    @property
    def retrain_signal(self) -> bool:
        """Whether the detector recommends retraining.

        Returns:
            ``True`` when CRITICAL drift has been detected and not yet
            acknowledged.

        Example:
            >>> dd = DriftDetector(threshold_critical=0.0)
            >>> dd.set_reference(np.zeros((100, 1)))
            >>> _ = dd.update(np.ones((100, 1)))
            >>> dd.retrain_signal
            True
        """
        return self._retrain_signal

    def acknowledge_retrain(self) -> None:
        """Reset the retrain signal and shift the reference window.

        Copies the current sliding window into the reference window so
        future comparisons are relative to the new baseline.

        Returns:
            None

        Example:
            >>> dd = DriftDetector(threshold_critical=0.0)
            >>> dd.set_reference(np.zeros((100, 1)))
            >>> _ = dd.update(np.ones((100, 1)))
            >>> dd.acknowledge_retrain()
            >>> dd.retrain_signal
            False
        """
        self._retrain_signal = False
        if self._current_window is not None:
            self._reference_window = self._current_window.copy()
            logger.info("Reference window shifted to current window after retrain acknowledgement.")
        self._phase = "STABLE"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_kl(self, ref: np.ndarray, cur: np.ndarray) -> float:
        """Compute mean KL divergence across feature dimensions.

        For each feature column the values of *ref* and *cur* are binned
        into ``n_bins`` bins using shared bin edges.  A small epsilon is
        added to prevent ``log(0)``.

        Args:
            ref: Reference data of shape ``(N_ref, D)``.
            cur: Current data of shape ``(N_cur, D)``.

        Returns:
            The mean KL divergence across all feature dimensions.
        """
        n_features: int = ref.shape[1]
        kl_sum: float = 0.0

        for d in range(n_features):
            ref_col: np.ndarray = ref[:, d]
            cur_col: np.ndarray = cur[:, d]

            # Shared bin edges covering both windows
            combined_min: float = float(min(ref_col.min(), cur_col.min()))
            combined_max: float = float(max(ref_col.max(), cur_col.max()))

            edges: np.ndarray = np.linspace(combined_min, combined_max, self._n_bins + 1)

            ref_counts: np.ndarray = np.histogram(ref_col, bins=edges)[0].astype(np.float64)
            cur_counts: np.ndarray = np.histogram(cur_col, bins=edges)[0].astype(np.float64)

            # Normalise to probability distributions
            ref_prob: np.ndarray = ref_counts / ref_counts.sum() + _EPSILON
            cur_prob: np.ndarray = cur_counts / cur_counts.sum() + _EPSILON

            # Re-normalise after epsilon addition
            ref_prob = ref_prob / ref_prob.sum()
            cur_prob = cur_prob / cur_prob.sum()

            # KL(ref || cur)
            kl_d: float = float(np.sum(ref_prob * np.log(ref_prob / cur_prob)))
            kl_sum += kl_d

        mean_kl: float = kl_sum / n_features
        return mean_kl
