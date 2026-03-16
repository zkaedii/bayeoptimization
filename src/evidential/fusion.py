"""Product-of-Experts multimodal evidence fusion for Dirichlet models."""

import logging
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FusionPhase(str, Enum):
    """Phase label assigned to a fused prediction.

    Attributes:
        CONSENSUS: All modalities agree on the top class with low uncertainty.
        CONFLICT: At least two modalities disagree on the top class.
        UNCERTAIN: Fused uncertainty exceeds the uncertain threshold.
    """

    CONSENSUS = "CONSENSUS"
    CONFLICT = "CONFLICT"
    UNCERTAIN = "UNCERTAIN"


class MultimodalFusion:
    """Product-of-Experts evidence combination across modalities.

    Fuses Dirichlet evidence vectors from multiple modalities using the
    Dirichlet product-of-experts rule and classifies the fusion outcome
    into one of three phases: CONSENSUS, CONFLICT, or UNCERTAIN.

    Args:
        n_classes: Number of target classes (K >= 2).
        threshold_consensus: Maximum total uncertainty for a CONSENSUS verdict.
        threshold_uncertain: Minimum total uncertainty for an UNCERTAIN verdict.

    Returns:
        A ``MultimodalFusion`` instance.

    Example:
        >>> fuser = MultimodalFusion(n_classes=3)
        >>> result = fuser.fuse([np.array([10.0, 1.0, 0.5]),
        ...                      np.array([8.0, 0.5, 0.2])])
        >>> result["phase"]
        'CONSENSUS'
    """

    def __init__(
        self,
        n_classes: int,
        threshold_consensus: float = 0.3,
        threshold_uncertain: float = 0.7,
    ) -> None:
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        if threshold_consensus >= threshold_uncertain:
            raise ValueError(
                f"threshold_consensus ({threshold_consensus}) must be less than "
                f"threshold_uncertain ({threshold_uncertain})"
            )
        self.n_classes: int = n_classes
        self.threshold_consensus: float = threshold_consensus
        self.threshold_uncertain: float = threshold_uncertain
        logger.debug(
            "MultimodalFusion initialised: K=%d, consensus<%.2f, uncertain>=%.2f",
            n_classes,
            threshold_consensus,
            threshold_uncertain,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        evidence_list: list[np.ndarray | None],
    ) -> dict[str, Any]:
        """Fuse evidence vectors from multiple modalities.

        Missing modalities (``None`` entries) are silently skipped.  If no
        valid evidence is supplied a uniform-uncertainty result is returned.

        Args:
            evidence_list: List of 1-D evidence arrays of shape ``(K,)`` or
                ``None`` for missing modalities.

        Returns:
            Dictionary with keys:

            - ``fused_probs`` (np.ndarray): class probabilities from fused alpha.
            - ``fused_uncertainty`` (float): vacuity K / S of fused Dirichlet.
            - ``phase`` (str): one of ``CONSENSUS``, ``CONFLICT``, ``UNCERTAIN``.
            - ``modality_agreements`` (list[int]): top-class per valid modality.

        Example:
            >>> fuser = MultimodalFusion(n_classes=2)
            >>> out = fuser.fuse([np.array([5.0, 0.1]), None, np.array([6.0, 0.3])])
            >>> out["phase"]
            'CONSENSUS'
        """
        # Collect valid evidence, clamp negatives, compute per-modality alphas
        alpha_list: list[np.ndarray] = []
        modality_top_classes: list[int] = []

        for idx, ev in enumerate(evidence_list):
            if ev is None:
                logger.debug("Modality %d is missing, skipping.", idx)
                continue
            ev = np.asarray(ev, dtype=np.float64)
            if ev.ndim != 1 or ev.shape[0] != self.n_classes:
                raise ValueError(
                    f"Evidence at index {idx} must be 1-D with length "
                    f"{self.n_classes}, got shape {ev.shape}"
                )
            ev = np.maximum(ev, 0.0)
            alpha_i = ev + 1.0
            alpha_list.append(alpha_i)
            modality_top_classes.append(int(np.argmax(alpha_i)))

        if not alpha_list:
            logger.warning("No valid evidence vectors supplied; returning uniform.")
            uniform_probs = np.ones(self.n_classes, dtype=np.float64) / self.n_classes
            return {
                "fused_probs": uniform_probs,
                "fused_uncertainty": 1.0,
                "phase": FusionPhase.UNCERTAIN.value,
                "modality_agreements": [],
            }

        # Fuse via product-of-experts
        fused_alpha = self._product_of_experts(alpha_list)
        fused_strength = float(np.sum(fused_alpha))
        fused_probs = fused_alpha / fused_strength
        fused_uncertainty = float(self.n_classes / fused_strength)

        # Phase classification
        phase = self._classify_phase(
            modality_top_classes,
            fused_uncertainty,
        )

        logger.debug(
            "Fusion result: phase=%s, uncertainty=%.4f, top_classes=%s",
            phase,
            fused_uncertainty,
            modality_top_classes,
        )

        return {
            "fused_probs": fused_probs,
            "fused_uncertainty": fused_uncertainty,
            "phase": phase,
            "modality_agreements": modality_top_classes,
        }

    # ------------------------------------------------------------------
    # Product-of-experts
    # ------------------------------------------------------------------

    def _product_of_experts(
        self,
        alpha_list: list[np.ndarray],
    ) -> np.ndarray:
        """Combine Dirichlet experts via the product-of-experts rule.

        The product rule for Dirichlet distributions yields:
        ``alpha_fused = sum(alpha_i) - (N - 1) * ones``.

        Fused alphas are clamped to >= 1 to avoid degenerate distributions.

        Args:
            alpha_list: List of 1-D concentration-parameter arrays.

        Returns:
            Fused concentration parameters of shape ``(K,)``.

        Example:
            >>> fuser = MultimodalFusion(n_classes=3)
            >>> a = fuser._product_of_experts([np.array([3.0, 1.0, 1.0]),
            ...                                np.array([2.0, 1.0, 1.0])])
            >>> a[0] > a[1]
            True
        """
        n_experts = len(alpha_list)
        summed = np.sum(np.stack(alpha_list, axis=0), axis=0)
        fused = summed - (n_experts - 1) * np.ones(self.n_classes, dtype=np.float64)
        fused = np.maximum(fused, 1.0)
        return fused

    # ------------------------------------------------------------------
    # Phase classification
    # ------------------------------------------------------------------

    def _classify_phase(
        self,
        modality_top_classes: list[int],
        fused_uncertainty: float,
    ) -> str:
        """Determine the fusion phase from modality agreement and uncertainty.

        Args:
            modality_top_classes: Top-predicted class index per modality.
            fused_uncertainty: Vacuity of the fused Dirichlet.

        Returns:
            One of ``"CONSENSUS"``, ``"CONFLICT"``, ``"UNCERTAIN"``.

        Example:
            >>> fuser = MultimodalFusion(n_classes=3)
            >>> fuser._classify_phase([0, 0, 0], 0.1)
            'CONSENSUS'
        """
        # Check for conflict: at least two modalities disagree
        unique_classes = set(modality_top_classes)
        has_conflict = len(unique_classes) > 1

        if has_conflict:
            return FusionPhase.CONFLICT.value

        if fused_uncertainty >= self.threshold_uncertain:
            return FusionPhase.UNCERTAIN.value

        if fused_uncertainty < self.threshold_consensus:
            return FusionPhase.CONSENSUS.value

        # Between consensus and uncertain thresholds, not conflicting
        # Default to UNCERTAIN as the prediction is not confidently agreed upon
        return FusionPhase.UNCERTAIN.value
