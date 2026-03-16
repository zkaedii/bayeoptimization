"""Normal-Inverse-Gamma evidential regression with uncertainty decomposition."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _softplus(x: float) -> float:
    """Numerically stable softplus: log(1 + exp(x)).

    Args:
        x: Scalar input value.

    Returns:
        softplus(x), always positive.

    Example:
        >>> round(_softplus(0.0), 4)
        0.6931
    """
    if x > 20.0:
        return x
    return float(np.log1p(np.exp(x)))


class EvidentialRegressor:
    """Normal-Inverse-Gamma (NIG) evidential regression.

    The network predicts four parameters ``(gamma, log_nu, log_alpha,
    log_beta)`` which parameterise a NIG posterior.  From this posterior the
    aleatoric and epistemic uncertainties can be decomposed analytically.

    Args:
        kl_weight: Regularisation weight for the evidence regulariser
            ``lambda * |y - gamma| * (2*nu + alpha)``.

    Returns:
        An ``EvidentialRegressor`` instance.

    Example:
        >>> reg = EvidentialRegressor(kl_weight=0.01)
        >>> pred = reg.predict(np.array([2.5, 0.0, 1.0, 0.5]))
        >>> "mean" in pred and "epistemic" in pred
        True
    """

    def __init__(self, kl_weight: float = 0.001) -> None:
        self.kl_weight: float = kl_weight
        logger.debug("EvidentialRegressor initialised with kl_weight=%.4f", kl_weight)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, outputs: np.ndarray) -> dict[str, float]:
        """Parse raw network outputs into NIG parameters and uncertainties.

        Args:
            outputs: 1-D array of length 4 containing
                ``(gamma, log_nu, log_alpha, log_beta)``.

        Returns:
            Dictionary with keys ``gamma``, ``nu``, ``alpha``, ``beta``,
            ``aleatoric``, ``epistemic``, ``total``.

        Example:
            >>> reg = EvidentialRegressor()
            >>> fwd = reg.forward(np.array([1.0, 0.0, 2.0, 0.5]))
            >>> fwd["gamma"]
            1.0
        """
        gamma, nu, alpha, beta = self._parse_outputs(outputs)

        aleatoric = beta / (alpha - 1.0)
        epistemic = beta / (nu * (alpha - 1.0))
        total = aleatoric + epistemic

        return {
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total": total,
        }

    def predict(self, outputs: np.ndarray) -> dict[str, float]:
        """Return a user-facing prediction dictionary.

        Args:
            outputs: 1-D array of length 4 containing
                ``(gamma, log_nu, log_alpha, log_beta)``.

        Returns:
            Dictionary with keys ``mean``, ``aleatoric``, ``epistemic``,
            ``total``.

        Example:
            >>> reg = EvidentialRegressor()
            >>> pred = reg.predict(np.array([3.0, 1.0, 2.0, 0.5]))
            >>> isinstance(pred["mean"], float)
            True
        """
        fwd = self.forward(outputs)
        return {
            "mean": fwd["gamma"],
            "aleatoric": fwd["aleatoric"],
            "epistemic": fwd["epistemic"],
            "total": fwd["total"],
        }

    def compute_loss(
        self,
        outputs: np.ndarray,
        target: float,
    ) -> dict[str, float]:
        """Compute the NIG evidential regression loss.

        The loss is the negative log-likelihood of the NIG distribution plus
        a regulariser that penalises evidence on incorrect predictions.

        Args:
            outputs: 1-D array of length 4 containing
                ``(gamma, log_nu, log_alpha, log_beta)``.
            target: Ground-truth scalar target value.

        Returns:
            Dictionary with keys ``total_loss``, ``nig_loss``, ``reg_loss``.

        Example:
            >>> reg = EvidentialRegressor()
            >>> loss = reg.compute_loss(np.array([2.0, 0.0, 1.5, 0.3]), target=2.1)
            >>> loss["total_loss"] > 0
            True
        """
        gamma, nu, alpha, beta = self._parse_outputs(outputs)

        omega = 2.0 * beta * (1.0 + nu)
        residual = (target - gamma) ** 2

        # NIG NLL
        nig_loss = (
            0.5 * np.log(np.pi / nu)
            - alpha * np.log(omega)
            + (alpha + 0.5) * np.log(residual * nu + omega)
        )
        # Add log-gamma normalisation terms for completeness
        nig_loss += float(
            np.log(np.exp(float(self._log_gamma(alpha))) /
                   np.exp(float(self._log_gamma(alpha + 0.5))) + 1e-30)
        )
        # Simplified: use gammaln directly
        nig_loss_val = float(
            0.5 * np.log(np.pi / nu)
            - alpha * np.log(omega)
            + (alpha + 0.5) * np.log(residual * nu + omega)
            + self._log_gamma(alpha)
            - self._log_gamma(alpha + 0.5)
        )

        # Regulariser: penalise evidence for wrong predictions
        reg_loss = float(self.kl_weight * np.abs(target - gamma) * (2.0 * nu + alpha))

        total = nig_loss_val + reg_loss

        logger.debug(
            "Regression loss: nig=%.4f  reg=%.4f  total=%.4f",
            nig_loss_val,
            reg_loss,
            total,
        )

        return {
            "total_loss": total,
            "nig_loss": nig_loss_val,
            "reg_loss": reg_loss,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_outputs(self, outputs: np.ndarray) -> tuple[float, float, float, float]:
        """Parse and transform raw network outputs into NIG parameters.

        Args:
            outputs: 1-D array of length 4.

        Returns:
            Tuple ``(gamma, nu, alpha, beta)`` with constraints enforced.

        Example:
            >>> reg = EvidentialRegressor()
            >>> g, n, a, b = reg._parse_outputs(np.array([0.0, 0.0, 0.0, 0.0]))
            >>> n > 0 and a > 1 and b > 0
            True
        """
        outputs = np.asarray(outputs, dtype=np.float64)
        if outputs.ndim != 1 or outputs.shape[0] != 4:
            raise ValueError(
                f"Outputs must be 1-D with length 4, got shape {outputs.shape}"
            )
        gamma = float(outputs[0])
        nu = _softplus(float(outputs[1]))
        alpha = _softplus(float(outputs[2])) + 1.0
        beta = _softplus(float(outputs[3]))

        # Guard against degenerate values
        nu = max(nu, 1e-6)
        beta = max(beta, 1e-6)

        return gamma, nu, alpha, beta

    @staticmethod
    def _log_gamma(x: float) -> float:
        """Compute log-gamma, delegating to scipy.special.gammaln.

        Args:
            x: Positive scalar.

        Returns:
            ``log(Gamma(x))``.

        Example:
            >>> import math
            >>> abs(EvidentialRegressor._log_gamma(1.0) - 0.0) < 1e-10
            True
        """
        from scipy.special import gammaln as _gammaln

        return float(_gammaln(x))
