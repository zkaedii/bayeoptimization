"""ZKAEDI PRIME Bayesian Optimization with recursive Hamiltonian dynamics.

Extends :class:`BayesianOptimizer` by evolving a Hamiltonian field over the
acquisition-function landscape, using sigmoid momentum coupling and Box-Muller
noise to balance exploration and exploitation in a physics-inspired manner.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize

from src.optimization.bayesian import BayesianOptimizer

logger: logging.Logger = logging.getLogger(__name__)


class ZkaediPrimeBO(BayesianOptimizer):
    """Bayesian optimizer augmented with recursive Hamiltonian dynamics.

    The PRIME field *H* lives in acquisition-function space and is evolved via
    the coupled Hamiltonian equation::

        H_t = H_base + eta * H_{t-1} * sigmoid(gamma * H_{t-1})
              + sigma_prime * bm_noise * (1 + beta * |H_{t-1}|)

    After ``t_max`` evolution steps the final field modulates the acquisition
    function, and the modulated landscape is optimised to propose the next
    candidate.

    Args:
        *args: Positional arguments forwarded to :class:`BayesianOptimizer`.
        eta: Momentum coupling strength for the Hamiltonian recursion.
        gamma: Sigmoid steepness controlling nonlinear momentum response.
        beta: Noise amplification factor proportional to field magnitude.
        sigma_prime: Base noise scale (Box-Muller generated).
        t_max: Number of Hamiltonian evolution steps per :meth:`suggest` call.
        **kwargs: Keyword arguments forwarded to :class:`BayesianOptimizer`.

    Returns:
        A configured ZkaediPrimeBO instance.

    Example:
        >>> opt = ZkaediPrimeBO(bounds=[(-5.0, 5.0)], n_warmup=3)
        >>> x = opt.suggest()
        >>> opt.update(x, float(x[0] ** 2))
    """

    def __init__(
        self,
        *args: Any,
        eta: float = 0.4,
        gamma: float = 0.3,
        beta: float = 0.1,
        sigma_prime: float = 0.05,
        t_max: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eta: float = eta
        self.gamma: float = gamma
        self.beta: float = beta
        self.sigma_prime: float = sigma_prime
        self.t_max: int = t_max

        self._H: np.ndarray = np.zeros(self.ndim)
        self._phase: str = "EXPLORING"
        self._step_count: int = 0
        self._variance: float = 0.0

        logger.info(
            "ZkaediPrimeBO created: eta=%.3f, gamma=%.3f, beta=%.3f, "
            "sigma_prime=%.3f, t_max=%d",
            eta,
            gamma,
            beta,
            sigma_prime,
            t_max,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self) -> np.ndarray:
        """Suggest the next candidate using Hamiltonian-modulated acquisition.

        During warmup, delegates to the parent :meth:`BayesianOptimizer.suggest`.
        After warmup, the GP posterior mean initialises *H_base*, the
        Hamiltonian is evolved for ``t_max`` steps, and the resulting field
        modulates the acquisition function. The modulated landscape is then
        optimised via multi-start L-BFGS-B.

        Returns:
            A 1-D numpy array of shape ``(ndim,)`` — the proposed candidate.

        Example:
            >>> opt = ZkaediPrimeBO(bounds=[(0.0, 1.0)], n_warmup=2, t_max=10)
            >>> x = opt.suggest()
            >>> x.shape
            (1,)
        """
        if len(self.y_obs) < self.n_warmup:
            return super().suggest()

        if not self._gp_fitted:
            self._fit_gp()

        # Compute H_base from GP posterior mean at the current best location
        best_idx: int = int(np.argmin(self.y_obs))
        best_x: np.ndarray = self.X_obs[best_idx]
        mu_base: np.ndarray
        mu_base, _ = self.gp.predict(np.atleast_2d(best_x), return_std=True)
        h_base: np.ndarray = np.full(self.ndim, float(mu_base[0]))

        # Evolve Hamiltonian field
        h_evolved: np.ndarray = self._evolve_hamiltonian(h_base)
        self._H = h_evolved

        # Update phase tracking
        self._variance = float(np.var(h_evolved))
        if self._variance < 0.1:
            self._phase = "CONVERGING"
        elif self._variance > 2.0:
            self._phase = "BIFURCATING"
        else:
            self._phase = "EXPLORING"

        logger.debug(
            "Hamiltonian phase: %s (variance=%.4f)", self._phase, self._variance
        )

        # Optimise modulated acquisition
        lower_arr: np.ndarray = np.array([b[0] for b in self.bounds])
        upper_arr: np.ndarray = np.array([b[1] for b in self.bounds])

        best_candidate: Optional[np.ndarray] = None
        best_acq: float = np.inf

        for _ in range(self._n_restarts):
            x0: np.ndarray = np.random.uniform(lower_arr, upper_arr)
            result = minimize(
                fun=lambda x: -self._modulated_acquisition(x, h_evolved),
                x0=x0,
                bounds=self.bounds,
                method="L-BFGS-B",
            )
            if result.fun < best_acq:
                best_acq = result.fun
                best_candidate = result.x

        if best_candidate is None:
            logger.warning(
                "All L-BFGS-B restarts failed in PRIME suggest; "
                "falling back to random sample"
            )
            return np.random.uniform(lower_arr, upper_arr)

        logger.debug(
            "PRIME suggested x with modulated acquisition value %.6f", -best_acq
        )
        return best_candidate

    def get_field_state(self) -> dict[str, Any]:
        """Return the current state of the Hamiltonian field.

        Returns:
            A dictionary containing:
            - ``H``: Current Hamiltonian field vector (ndarray).
            - ``phase``: One of ``"CONVERGING"``, ``"EXPLORING"``, ``"BIFURCATING"``.
            - ``step_count``: Total number of evolution steps executed so far.
            - ``variance``: Variance of the current field vector.

        Example:
            >>> opt = ZkaediPrimeBO(bounds=[(0.0, 1.0)])
            >>> state = opt.get_field_state()
            >>> sorted(state.keys())
            ['H', 'phase', 'step_count', 'variance']
        """
        return {
            "H": self._H.copy(),
            "phase": self._phase,
            "step_count": self._step_count,
            "variance": self._variance,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evolve_hamiltonian(self, H_base: np.ndarray) -> np.ndarray:
        """Evolve the Hamiltonian field for ``t_max`` steps.

        Applies the recursion::

            sigmoid = 1 / (1 + exp(-gamma * H_prev))
            H_t = H_base + eta * H_prev * sigmoid
                  + sigma_prime * bm_noise * (1 + beta * |H_prev|)

        where ``bm_noise`` is generated via the Box-Muller transform (never
        ``np.random.normal``).

        Args:
            H_base: Base Hamiltonian field derived from the GP posterior mean,
                shape ``(ndim,)``.

        Returns:
            Evolved Hamiltonian field of shape ``(ndim,)``.

        Example:
            >>> opt = ZkaediPrimeBO(bounds=[(0.0, 1.0)], t_max=5)
            >>> H_out = opt._evolve_hamiltonian(np.array([0.5]))
            >>> H_out.shape
            (1,)
        """
        H_prev: np.ndarray = self._H.copy()
        H_t: np.ndarray = H_base.copy()

        for step in range(self.t_max):
            sigmoid: np.ndarray = 1.0 / (1.0 + np.exp(-self.gamma * H_prev))
            noise: np.ndarray = self._box_muller_noise(H_base.shape)
            H_t = (
                H_base
                + self.eta * H_prev * sigmoid
                + self.sigma_prime * noise * (1.0 + self.beta * np.abs(H_prev))
            )
            H_prev = H_t

        self._step_count += self.t_max
        return H_t

    def _box_muller_noise(self, shape: tuple[int, ...]) -> np.ndarray:
        """Generate standard-normal noise using the Box-Muller transform.

        This method is used instead of ``np.random.normal`` in all PRIME
        components, as mandated by project conventions.

        The transform converts pairs of uniform samples into independent
        standard-normal samples::

            z = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)

        Args:
            shape: Desired output shape for the noise array.

        Returns:
            Array of standard-normal samples with the requested shape.

        Example:
            >>> opt = ZkaediPrimeBO(bounds=[(0.0, 1.0)])
            >>> noise = opt._box_muller_noise((100,))
            >>> noise.shape
            (100,)
        """
        n_elements: int = int(np.prod(shape))
        u1: np.ndarray = np.random.uniform(size=n_elements)
        u2: np.ndarray = np.random.uniform(size=n_elements)

        # Clamp u1 away from zero to avoid log(0)
        u1 = np.clip(u1, 1e-10, 1.0)

        z: np.ndarray = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        return z.reshape(shape)

    def _modulated_acquisition(
        self, x: np.ndarray, H_field: np.ndarray
    ) -> float:
        """Evaluate the Hamiltonian-modulated acquisition function.

        Combines the standard acquisition value with the evolved Hamiltonian
        field. The field acts as a spatially-varying exploration bonus that
        biases the search toward dynamically interesting regions.

        The modulation is computed as::

            modulated = acq_base * (1 + tanh(dot(H_field, x_norm)))

        where ``x_norm`` is the candidate normalised to [0, 1] within bounds.

        Args:
            x: Candidate point, shape ``(ndim,)``.
            H_field: Evolved Hamiltonian field, shape ``(ndim,)``.

        Returns:
            Modulated acquisition value.

        Example:
            >>> opt = ZkaediPrimeBO(bounds=[(0.0, 1.0)], n_warmup=1)
            >>> opt.update(np.array([0.5]), 0.25)
            >>> val = opt._modulated_acquisition(np.array([0.3]), np.array([0.1]))
            >>> isinstance(val, float)
            True
        """
        acq_base: float = self._acquisition(x)

        # Normalise x to [0, 1] within bounds
        lower: np.ndarray = np.array([b[0] for b in self.bounds])
        upper: np.ndarray = np.array([b[1] for b in self.bounds])
        x_norm: np.ndarray = (np.asarray(x) - lower) / (upper - lower + 1e-12)

        # Hamiltonian modulation via tanh coupling
        coupling: float = float(np.tanh(np.dot(H_field, x_norm)))
        modulated: float = acq_base * (1.0 + coupling)

        return modulated
