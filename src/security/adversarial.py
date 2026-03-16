"""Adversarial robustness evaluation with FGSM attacks and Hamiltonian defense.

Provides tools to generate FGSM adversarial examples, apply a
Hamiltonian-inspired smoothing defense, and evaluate model robustness
across multiple perturbation strengths.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

_DEFAULT_EPSILONS: list[float] = [0.01, 0.05, 0.1, 0.2, 0.3]


@dataclass
class RobustnessReport:
    """Results of an adversarial robustness evaluation.

    Args:
        clean_accuracy: Model accuracy on unperturbed inputs.
        adversarial_results: Mapping from epsilon to adversarial accuracy.
        defense_results: Mapping from epsilon to defended accuracy.
        per_sample: Per-sample detail dicts (clean pred, adv pred, etc.).

    Returns:
        A ``RobustnessReport`` dataclass instance.

    Example:
        >>> rr = RobustnessReport(0.95, {0.1: 0.7}, {0.1: 0.85}, [])
        >>> rr.clean_accuracy
        0.95
    """

    clean_accuracy: float
    adversarial_results: dict[float, float]
    defense_results: dict[float, float]
    per_sample: list[dict[str, Any]]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid function.

    Args:
        z: Input array.

    Returns:
        Sigmoid of each element.

    Example:
        >>> float(np.round(_sigmoid(np.array([0.0]))[0], 1))
        0.5
    """
    return 1.0 / (1.0 + np.exp(-z))


class AdversarialRobustness:
    """FGSM attack generation and Hamiltonian smoothing defense.

    Args:
        epsilon: Default perturbation magnitude for FGSM.
        clip_min: Minimum value for clipping adversarial examples.
        clip_max: Maximum value for clipping adversarial examples.
        defense_eta: Step-size parameter for Hamiltonian defense.
        defense_gamma: Scaling factor inside the sigmoid gate.
        defense_sigma: Standard deviation for Gaussian smoothing.

    Returns:
        An ``AdversarialRobustness`` instance.

    Example:
        >>> ar = AdversarialRobustness(epsilon=0.1)
        >>> x = np.array([[0.5, 0.5]])
        >>> grad = np.array([[1.0, -1.0]])
        >>> x_adv = ar.fgsm_attack(x, grad)
        >>> x_adv.shape
        (1, 2)
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        defense_eta: float = 0.1,
        defense_gamma: float = 0.95,
        defense_sigma: float = 0.5,
    ) -> None:
        self._epsilon: float = epsilon
        self._clip_min: float = clip_min
        self._clip_max: float = clip_max
        self._defense_eta: float = defense_eta
        self._defense_gamma: float = defense_gamma
        self._defense_sigma: float = defense_sigma
        logger.info(
            "AdversarialRobustness initialised (eps=%.3f, sigma=%.3f)",
            epsilon,
            defense_sigma,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fgsm_attack(
        self,
        x: np.ndarray,
        gradient: np.ndarray,
        epsilon: float | None = None,
    ) -> np.ndarray:
        """Generate adversarial examples using the Fast Gradient Sign Method.

        Computes ``x_adv = clip(x + epsilon * sign(gradient))``.

        Args:
            x: Clean inputs of shape ``(N, D)``.
            gradient: Loss gradients w.r.t. *x*, same shape as *x*.
            epsilon: Perturbation magnitude. Uses default if ``None``.

        Returns:
            Adversarial examples clipped to ``[clip_min, clip_max]``.

        Example:
            >>> ar = AdversarialRobustness(clip_min=0.0, clip_max=1.0)
            >>> x = np.full((2, 3), 0.5)
            >>> grad = np.ones((2, 3))
            >>> x_adv = ar.fgsm_attack(x, grad, epsilon=0.1)
            >>> np.allclose(x_adv, 0.6)
            True
        """
        eps: float = epsilon if epsilon is not None else self._epsilon
        perturbation: np.ndarray = eps * np.sign(gradient)
        x_adv: np.ndarray = x + perturbation
        x_adv = np.clip(x_adv, self._clip_min, self._clip_max)
        return x_adv

    def hamiltonian_defense(self, x: np.ndarray) -> np.ndarray:
        """Apply Hamiltonian-inspired smoothing defense.

        The defense is:

            smooth = gaussian_filter1d(x, sigma)
            H(x) = x + eta * smooth * sigmoid(gamma * smooth)

        Args:
            x: Input array of shape ``(N, D)``.

        Returns:
            Defended array of the same shape.

        Example:
            >>> ar = AdversarialRobustness(defense_eta=0.1, defense_sigma=0.5)
            >>> x = np.random.rand(5, 10)
            >>> x_def = ar.hamiltonian_defense(x)
            >>> x_def.shape
            (5, 10)
        """
        smoothed: np.ndarray = gaussian_filter1d(
            x, sigma=self._defense_sigma, axis=-1
        )
        gate: np.ndarray = _sigmoid(self._defense_gamma * smoothed)
        x_defended: np.ndarray = x + self._defense_eta * smoothed * gate
        return x_defended

    def evaluate(
        self,
        x_clean: np.ndarray,
        y_true: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        gradient_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        epsilon_list: list[float] | None = None,
    ) -> RobustnessReport:
        """Evaluate model robustness across multiple perturbation strengths.

        For each epsilon value, FGSM adversarial examples are generated,
        the Hamiltonian defense is applied, and accuracies are recorded.

        Args:
            x_clean: Clean input data of shape ``(N, D)``.
            y_true: Ground-truth labels of shape ``(N,)``.
            predict_fn: ``predict_fn(x) -> predictions`` of shape ``(N,)``.
            gradient_fn: ``gradient_fn(x, y) -> gradients`` same shape as *x*.
            epsilon_list: List of epsilon values to test. Uses default if
                ``None``.

        Returns:
            A :class:`RobustnessReport` with clean, adversarial, and
            defended accuracies.

        Example:
            >>> ar = AdversarialRobustness()
            >>> x = np.random.rand(20, 5)
            >>> y = np.random.randint(0, 2, 20)
            >>> pred = lambda x: np.ones(x.shape[0], dtype=int)
            >>> grad = lambda x, y: np.random.randn(*x.shape)
            >>> report = ar.evaluate(x, y, pred, grad, epsilon_list=[0.1])
            >>> 0.1 in report.adversarial_results
            True
        """
        epsilons: list[float] = epsilon_list if epsilon_list is not None else list(_DEFAULT_EPSILONS)

        clean_preds: np.ndarray = predict_fn(x_clean)
        clean_acc: float = float(np.mean(clean_preds == y_true))

        adversarial_results: dict[float, float] = {}
        defense_results: dict[float, float] = {}
        per_sample: list[dict[str, Any]] = []

        gradients: np.ndarray = gradient_fn(x_clean, y_true)

        for eps in epsilons:
            x_adv: np.ndarray = self.fgsm_attack(x_clean, gradients, epsilon=eps)
            adv_preds: np.ndarray = predict_fn(x_adv)
            adv_acc: float = float(np.mean(adv_preds == y_true))
            adversarial_results[eps] = adv_acc

            x_defended: np.ndarray = self.hamiltonian_defense(x_adv)
            def_preds: np.ndarray = predict_fn(x_defended)
            def_acc: float = float(np.mean(def_preds == y_true))
            defense_results[eps] = def_acc

            logger.info(
                "eps=%.3f  adv_acc=%.4f  def_acc=%.4f", eps, adv_acc, def_acc
            )

        # Build per-sample details for the default epsilon
        default_eps: float = self._epsilon
        x_adv_default: np.ndarray = self.fgsm_attack(x_clean, gradients, epsilon=default_eps)
        adv_preds_default: np.ndarray = predict_fn(x_adv_default)
        x_def_default: np.ndarray = self.hamiltonian_defense(x_adv_default)
        def_preds_default: np.ndarray = predict_fn(x_def_default)

        for i in range(x_clean.shape[0]):
            per_sample.append(
                {
                    "index": i,
                    "true_label": int(y_true[i]),
                    "clean_pred": int(clean_preds[i]),
                    "adv_pred": int(adv_preds_default[i]),
                    "defended_pred": int(def_preds_default[i]),
                    "epsilon": default_eps,
                }
            )

        return RobustnessReport(
            clean_accuracy=clean_acc,
            adversarial_results=adversarial_results,
            defense_results=defense_results,
            per_sample=per_sample,
        )
