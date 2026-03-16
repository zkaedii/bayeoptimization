"""Dirichlet-based evidential classification with uncertainty decomposition."""

import logging
from typing import Any

import numpy as np
from scipy.special import digamma, gammaln

logger = logging.getLogger(__name__)


class EvidentialClassifier:
    """Dirichlet-based uncertainty quantification for classification.

    Maps non-negative evidence values to Dirichlet concentration parameters,
    then decomposes predictive uncertainty into epistemic (vacuity) and
    aleatoric (data-inherent) components.

    Args:
        n_classes: Number of target classes (K >= 2).
        kl_weight: Base weight for the KL-divergence regulariser.
        kl_annealing: Whether to linearly anneal the KL weight from 0 to
            ``kl_weight`` over ``anneal_steps`` training steps.
        anneal_steps: Number of steps over which the KL weight is annealed.

    Returns:
        An ``EvidentialClassifier`` instance ready for forward / loss calls.

    Example:
        >>> clf = EvidentialClassifier(n_classes=3)
        >>> result = clf.predict(np.array([10.0, 1.0, 0.5]))
        >>> result["predicted_class"]
        0
    """

    def __init__(
        self,
        n_classes: int,
        kl_weight: float = 0.001,
        kl_annealing: bool = True,
        anneal_steps: int = 100,
    ) -> None:
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        self.n_classes: int = n_classes
        self.kl_weight: float = kl_weight
        self.kl_annealing: bool = kl_annealing
        self.anneal_steps: int = anneal_steps
        logger.debug(
            "EvidentialClassifier initialised with K=%d, kl_weight=%.4f, "
            "annealing=%s over %d steps",
            n_classes,
            kl_weight,
            kl_annealing,
            anneal_steps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, evidence: np.ndarray) -> dict[str, Any]:
        """Compute Dirichlet parameters and uncertainty from raw evidence.

        Args:
            evidence: 1-D array of shape ``(K,)`` with non-negative evidence
                values.  Negative entries are clamped to 0.

        Returns:
            Dictionary with keys ``alpha``, ``dirichlet_strength``,
            ``probs``, ``vacuity``, ``epistemic``, ``aleatoric``.

        Example:
            >>> clf = EvidentialClassifier(n_classes=3)
            >>> out = clf.forward(np.array([5.0, 2.0, 1.0]))
            >>> out["probs"].shape
            (3,)
        """
        evidence = self._validate_evidence(evidence)
        alpha = evidence + 1.0
        strength = float(np.sum(alpha))
        probs = alpha / strength

        vacuity = float(self.n_classes / strength)
        epistemic = 1.0 - float(np.max(alpha)) / strength
        aleatoric = float(-np.sum(probs * np.log(probs + 1e-10)))

        return {
            "alpha": alpha,
            "dirichlet_strength": strength,
            "probs": probs,
            "vacuity": vacuity,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
        }

    def predict(self, evidence: np.ndarray) -> dict[str, Any]:
        """Return a full prediction dictionary including the top class.

        Args:
            evidence: 1-D array of shape ``(K,)`` with non-negative evidence.

        Returns:
            Dictionary with keys ``predicted_class``, ``probs``,
            ``uncertainty`` (vacuity), ``epistemic``, ``aleatoric``.

        Example:
            >>> clf = EvidentialClassifier(n_classes=4)
            >>> pred = clf.predict(np.array([0.0, 0.0, 12.0, 0.0]))
            >>> pred["predicted_class"]
            2
        """
        fwd = self.forward(evidence)
        return {
            "predicted_class": int(np.argmax(fwd["probs"])),
            "probs": fwd["probs"],
            "uncertainty": fwd["vacuity"],
            "epistemic": fwd["epistemic"],
            "aleatoric": fwd["aleatoric"],
        }

    def compute_loss(
        self,
        evidence: np.ndarray,
        target: int,
        step: int = 0,
    ) -> dict[str, float]:
        """Compute the evidential classification loss.

        The loss is composed of a negative-log-likelihood term and a
        KL-divergence regulariser between the predicted Dirichlet and a
        uniform prior Dir(1).

        Args:
            evidence: 1-D array of shape ``(K,)`` with non-negative evidence.
            target: Ground-truth class index in ``[0, K)``.
            step: Current training step (used for KL annealing).

        Returns:
            Dictionary with keys ``total_loss``, ``nll_loss``, ``kl_loss``,
            ``annealing_factor``.

        Example:
            >>> clf = EvidentialClassifier(n_classes=3)
            >>> loss = clf.compute_loss(np.array([5.0, 1.0, 0.5]), target=0, step=50)
            >>> loss["total_loss"] > 0
            True
        """
        evidence = self._validate_evidence(evidence)
        alpha = evidence + 1.0
        strength = float(np.sum(alpha))

        # NLL: log(S) - log(alpha_target)
        nll = float(np.log(strength) - np.log(alpha[target]))

        # KL(Dir(alpha) || Dir(1))
        kl = self._kl_dirichlet(alpha)

        # Annealing factor
        if self.kl_annealing:
            lam = min(1.0, step / max(self.anneal_steps, 1))
        else:
            lam = 1.0

        total = nll + lam * self.kl_weight * kl

        logger.debug(
            "Loss step=%d: nll=%.4f  kl=%.4f  lambda=%.4f  total=%.4f",
            step,
            nll,
            kl,
            lam,
            total,
        )

        return {
            "total_loss": total,
            "nll_loss": nll,
            "kl_loss": kl,
            "annealing_factor": lam,
        }

    def is_high_uncertainty(
        self,
        evidence: np.ndarray,
        threshold: float = 0.5,
    ) -> bool:
        """Check whether the vacuity exceeds a given threshold.

        Args:
            evidence: 1-D array of shape ``(K,)`` with non-negative evidence.
            threshold: Vacuity value above which the prediction is deemed
                highly uncertain.

        Returns:
            ``True`` if vacuity >= *threshold*, ``False`` otherwise.

        Example:
            >>> clf = EvidentialClassifier(n_classes=3)
            >>> clf.is_high_uncertainty(np.array([0.0, 0.0, 0.0]), threshold=0.5)
            True
        """
        fwd = self.forward(evidence)
        return bool(fwd["vacuity"] >= threshold)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_evidence(self, evidence: np.ndarray) -> np.ndarray:
        """Validate and clamp evidence to non-negative values.

        Args:
            evidence: Raw evidence array.

        Returns:
            Clamped copy of shape ``(K,)``.

        Example:
            >>> clf = EvidentialClassifier(n_classes=2)
            >>> clf._validate_evidence(np.array([-1.0, 3.0]))
            array([0., 3.])
        """
        evidence = np.asarray(evidence, dtype=np.float64)
        if evidence.ndim != 1 or evidence.shape[0] != self.n_classes:
            raise ValueError(
                f"Evidence must be 1-D with length {self.n_classes}, "
                f"got shape {evidence.shape}"
            )
        return np.maximum(evidence, 0.0)

    def _kl_dirichlet(self, alpha: np.ndarray) -> float:
        """KL divergence KL(Dir(alpha) || Dir(1)).

        Uses the analytic formula for two Dirichlet distributions.

        Args:
            alpha: Concentration parameters of shape ``(K,)``.

        Returns:
            Non-negative KL divergence value.

        Example:
            >>> clf = EvidentialClassifier(n_classes=3)
            >>> clf._kl_dirichlet(np.array([1.0, 1.0, 1.0]))
            0.0
        """
        k = len(alpha)
        ones = np.ones(k, dtype=np.float64)
        sum_alpha = np.sum(alpha)
        sum_ones = float(k)

        kl = float(
            gammaln(sum_alpha)
            - gammaln(sum_ones)
            - np.sum(gammaln(alpha))
            + np.sum(gammaln(ones))
            + np.sum((alpha - ones) * (digamma(alpha) - digamma(sum_alpha)))
        )
        return max(kl, 0.0)
