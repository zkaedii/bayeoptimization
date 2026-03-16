"""Open-set recognition via evidential uncertainty thresholding."""

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from .classifier import EvidentialClassifier

logger = logging.getLogger(__name__)


class OpenSetRecognition:
    """Reject unknown classes when evidential uncertainty exceeds a threshold.

    Wraps an :class:`EvidentialClassifier` and uses its vacuity score to
    decide whether an input belongs to a known class or should be rejected
    as ``UNKNOWN``.  The rejection threshold can be set manually or
    calibrated from a validation set to achieve a target false-positive rate.

    Args:
        n_classes: Number of known classes (K >= 2).
        threshold: Vacuity value at or above which a sample is rejected.

    Returns:
        An ``OpenSetRecognition`` instance.

    Example:
        >>> osr = OpenSetRecognition(n_classes=3, threshold=0.6)
        >>> pred = osr.predict_open(np.array([10.0, 1.0, 0.5]))
        >>> pred["label"] != "UNKNOWN"
        True
    """

    def __init__(self, n_classes: int, threshold: float = 0.5) -> None:
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        self.n_classes: int = n_classes
        self.threshold: float = threshold
        self._classifier: EvidentialClassifier = EvidentialClassifier(n_classes=n_classes)

        # Tracking statistics
        self._total_seen: int = 0
        self._total_rejected: int = 0
        self._class_seen: dict[int, int] = defaultdict(int)
        self._class_rejected: dict[int, int] = defaultdict(int)

        logger.debug(
            "OpenSetRecognition initialised: K=%d, threshold=%.4f",
            n_classes,
            threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_open(self, evidence: np.ndarray) -> dict[str, Any]:
        """Predict with open-set rejection.

        If the vacuity of the evidential prediction meets or exceeds the
        rejection threshold the sample is labelled ``"UNKNOWN"`` with
        ``predicted_class = -1``.

        Args:
            evidence: 1-D array of shape ``(K,)`` with non-negative evidence.

        Returns:
            Dictionary with keys ``predicted_class``, ``label``, ``probs``,
            ``uncertainty``, ``epistemic``, ``aleatoric``, ``rejected``.

        Example:
            >>> osr = OpenSetRecognition(n_classes=3, threshold=0.5)
            >>> pred = osr.predict_open(np.array([0.0, 0.0, 0.0]))
            >>> pred["rejected"]
            True
        """
        result = self._classifier.predict(evidence)
        vacuity = result["uncertainty"]
        rejected = bool(vacuity >= self.threshold)

        predicted_class: int = -1 if rejected else result["predicted_class"]
        label: str = "UNKNOWN" if rejected else str(predicted_class)

        # Update tracking stats — always track by would-be predicted class
        would_be_class: int = result["predicted_class"]
        self._total_seen += 1
        self._class_seen[would_be_class] += 1
        if rejected:
            self._total_rejected += 1
            self._class_rejected[would_be_class] = (
                self._class_rejected.get(would_be_class, 0) + 1
            )

        logger.debug(
            "predict_open: class=%d, rejected=%s, vacuity=%.4f",
            predicted_class,
            rejected,
            vacuity,
        )

        return {
            "predicted_class": predicted_class,
            "label": label,
            "probs": result["probs"],
            "uncertainty": vacuity,
            "epistemic": result["epistemic"],
            "aleatoric": result["aleatoric"],
            "rejected": rejected,
        }

    def calibrate(
        self,
        evidences: list[np.ndarray],
        labels: list[int],
        fpr_target: float = 0.05,
    ) -> float:
        """Calibrate the rejection threshold from validation data.

        Finds the threshold that achieves approximately the target
        false-positive rate (fraction of *known*-class samples incorrectly
        rejected).

        Args:
            evidences: List of evidence vectors for known-class samples.
            labels: Corresponding ground-truth class indices.
            fpr_target: Desired false-positive rate on known classes
                (i.e. fraction of known samples rejected).

        Returns:
            The calibrated threshold (also stored in ``self.threshold``).

        Example:
            >>> osr = OpenSetRecognition(n_classes=2)
            >>> evs = [np.array([10.0, 0.5])] * 20 + [np.array([0.1, 0.1])] * 5
            >>> labs = [0] * 20 + [1] * 5
            >>> thr = osr.calibrate(evs, labs, fpr_target=0.05)
            >>> thr > 0
            True
        """
        if len(evidences) != len(labels):
            raise ValueError(
                f"evidences and labels must have same length, "
                f"got {len(evidences)} and {len(labels)}"
            )
        if not evidences:
            raise ValueError("Cannot calibrate with empty validation set.")

        # Collect vacuity scores for all known-class samples
        uncertainties: list[float] = []
        for ev in evidences:
            fwd = self._classifier.forward(ev)
            uncertainties.append(fwd["vacuity"])

        uncertainties_sorted = np.sort(uncertainties)

        # Find threshold at (1 - fpr_target) quantile
        # We want at most fpr_target fraction of known samples to be rejected,
        # i.e. to have vacuity >= threshold.  So threshold is the
        # (1 - fpr_target) quantile of the uncertainty distribution.
        quantile_index = int(np.ceil((1.0 - fpr_target) * len(uncertainties_sorted))) - 1
        quantile_index = max(0, min(quantile_index, len(uncertainties_sorted) - 1))
        calibrated_threshold = float(uncertainties_sorted[quantile_index])

        self.threshold = calibrated_threshold
        logger.info(
            "Calibrated threshold to %.4f for fpr_target=%.4f on %d samples",
            calibrated_threshold,
            fpr_target,
            len(evidences),
        )
        return calibrated_threshold

    def get_rejection_stats(self) -> dict[str, Any]:
        """Return cumulative rejection statistics.

        Provides total counts and per-class rejection rates useful for
        monitoring distribution drift.

        Args:
            (none)

        Returns:
            Dictionary with keys ``total_seen``, ``total_rejected``,
            ``rejection_rate``, ``per_class_rates``.

        Example:
            >>> osr = OpenSetRecognition(n_classes=2, threshold=0.9)
            >>> _ = osr.predict_open(np.array([10.0, 0.5]))
            >>> stats = osr.get_rejection_stats()
            >>> stats["total_seen"]
            1
        """
        rejection_rate = (
            self._total_rejected / self._total_seen
            if self._total_seen > 0
            else 0.0
        )

        per_class_rates: dict[int, float] = {}
        all_classes = set(self._class_seen.keys()) | set(self._class_rejected.keys())
        for cls in sorted(all_classes):
            seen = self._class_seen.get(cls, 0)
            rejected = self._class_rejected.get(cls, 0)
            per_class_rates[cls] = rejected / seen if seen > 0 else 0.0

        return {
            "total_seen": self._total_seen,
            "total_rejected": self._total_rejected,
            "rejection_rate": rejection_rate,
            "per_class_rates": per_class_rates,
        }
