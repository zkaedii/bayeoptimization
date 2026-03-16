"""Bayesian optimization with Gaussian Process surrogate and acquisition functions.

Provides a GP-based optimizer supporting RBF, Matern52, and Matern32 kernels,
with Expected Improvement (EI), Probability of Improvement (PI), and Upper
Confidence Bound (UCB) acquisition strategies. Handles 1D through 20D input spaces.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Matern, RBF

logger: logging.Logger = logging.getLogger(__name__)

_KERNEL_MAP: dict[str, Kernel] = {
    "rbf": RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
    "matern52": Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5),
    "matern32": Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=1.5),
}

_VALID_ACQUISITIONS: set[str] = {"EI", "PI", "UCB"}


class BayesianOptimizer:
    """Gaussian Process surrogate model with acquisition function suite.

    Uses scikit-learn's GaussianProcessRegressor as the GP backend. During an
    initial warmup phase, candidates are drawn uniformly at random from the
    bounds. After warmup, the GP is fitted and new candidates are proposed by
    maximising the selected acquisition function via multi-start L-BFGS-B.

    Args:
        bounds: Search-space bounds as a list of (lower, upper) tuples, one per
            dimension. Supports 1-D through 20-D.
        kernel: GP kernel name — one of ``"rbf"``, ``"matern52"``, ``"matern32"``.
        acquisition: Acquisition function — one of ``"EI"``, ``"PI"``, ``"UCB"``.
        n_warmup: Number of random evaluations before GP fitting begins.
        xi: Exploration parameter for EI and PI acquisition functions.
        kappa: Exploration parameter for UCB acquisition function.
        noise_alpha: Regularisation noise added to the GP diagonal.

    Returns:
        A configured BayesianOptimizer instance.

    Example:
        >>> opt = BayesianOptimizer(bounds=[(-5.0, 5.0)], kernel="matern52")
        >>> x = opt.suggest()
        >>> opt.update(x, float(x[0] ** 2))
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        kernel: str = "matern52",
        acquisition: str = "EI",
        n_warmup: int = 10,
        xi: float = 0.01,
        kappa: float = 2.576,
        noise_alpha: float = 1e-6,
    ) -> None:
        if not bounds:
            raise ValueError("bounds must contain at least one (lower, upper) tuple")
        if len(bounds) > 20:
            raise ValueError("Maximum supported dimensionality is 20")
        for lo, hi in bounds:
            if lo >= hi:
                raise ValueError(f"Invalid bound ({lo}, {hi}): lower must be < upper")

        kernel_key: str = kernel.lower()
        if kernel_key not in _KERNEL_MAP:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Choose from: {list(_KERNEL_MAP.keys())}"
            )

        if acquisition not in _VALID_ACQUISITIONS:
            raise ValueError(
                f"Unknown acquisition '{acquisition}'. Choose from: {_VALID_ACQUISITIONS}"
            )

        self.bounds: list[tuple[float, float]] = bounds
        self.ndim: int = len(bounds)
        self.acquisition_name: str = acquisition
        self.n_warmup: int = n_warmup
        self.xi: float = xi
        self.kappa: float = kappa
        self.noise_alpha: float = noise_alpha

        self.X_obs: np.ndarray = np.empty((0, self.ndim))
        self.y_obs: np.ndarray = np.empty(0)

        selected_kernel: Kernel = _KERNEL_MAP[kernel_key]
        self.gp: GaussianProcessRegressor = GaussianProcessRegressor(
            kernel=selected_kernel,
            alpha=noise_alpha,
            n_restarts_optimizer=5,
            normalize_y=True,
        )
        self._gp_fitted: bool = False
        self._n_restarts: int = 10

        logger.info(
            "BayesianOptimizer created: ndim=%d, kernel=%s, acq=%s, warmup=%d",
            self.ndim,
            kernel,
            acquisition,
            n_warmup,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self) -> np.ndarray:
        """Suggest the next candidate point to evaluate.

        During warmup (fewer than ``n_warmup`` observations), returns a
        uniformly random point. Afterwards, maximises the acquisition function
        using multi-start L-BFGS-B with 10 random restarts.

        Returns:
            A 1-D numpy array of shape ``(ndim,)`` representing the candidate.

        Example:
            >>> opt = BayesianOptimizer(bounds=[(0.0, 1.0), (0.0, 1.0)])
            >>> candidate = opt.suggest()
            >>> candidate.shape
            (2,)
        """
        if len(self.y_obs) < self.n_warmup:
            lower: np.ndarray = np.array([b[0] for b in self.bounds])
            upper: np.ndarray = np.array([b[1] for b in self.bounds])
            candidate: np.ndarray = np.random.uniform(lower, upper)
            logger.debug("Warmup sample %d/%d", len(self.y_obs) + 1, self.n_warmup)
            return candidate

        if not self._gp_fitted:
            self._fit_gp()

        scipy_bounds: list[tuple[float, float]] = self.bounds
        lower_arr: np.ndarray = np.array([b[0] for b in self.bounds])
        upper_arr: np.ndarray = np.array([b[1] for b in self.bounds])

        best_x: Optional[np.ndarray] = None
        best_acq: float = np.inf  # we minimise negative acquisition

        for _ in range(self._n_restarts):
            x0: np.ndarray = np.random.uniform(lower_arr, upper_arr)
            result = minimize(
                fun=lambda x: -self._acquisition(x),
                x0=x0,
                bounds=scipy_bounds,
                method="L-BFGS-B",
            )
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x

        if best_x is None:
            logger.warning("All L-BFGS-B restarts failed; returning random sample")
            return np.random.uniform(lower_arr, upper_arr)

        logger.debug("Suggested x with acquisition value %.6f", -best_acq)
        return best_x

    def update(self, x: np.ndarray, y: float) -> None:
        """Record a new observation and refit the GP if warmup is complete.

        Args:
            x: Input point, shape ``(ndim,)``.
            y: Observed objective value at *x*.

        Returns:
            None

        Example:
            >>> opt = BayesianOptimizer(bounds=[(0.0, 1.0)])
            >>> opt.update(np.array([0.5]), 0.25)
            >>> len(opt.y_obs)
            1
        """
        x_2d: np.ndarray = np.atleast_2d(x)
        if x_2d.shape[1] != self.ndim:
            raise ValueError(
                f"Expected x with {self.ndim} dimensions, got {x_2d.shape[1]}"
            )
        self.X_obs = np.vstack([self.X_obs, x_2d])
        self.y_obs = np.append(self.y_obs, y)
        logger.debug("Observation %d recorded: y=%.6f", len(self.y_obs), y)

        if len(self.y_obs) >= self.n_warmup:
            self._fit_gp()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_gp(self) -> None:
        """Fit the Gaussian Process to all collected observations.

        Called automatically by :meth:`update` once the warmup phase ends, and
        before each :meth:`suggest` if the GP has not yet been fitted.

        Returns:
            None

        Example:
            >>> opt = BayesianOptimizer(bounds=[(0.0, 1.0)], n_warmup=1)
            >>> opt.update(np.array([0.5]), 0.25)
            >>> opt._gp_fitted
            True
        """
        if len(self.y_obs) == 0:
            logger.warning("Cannot fit GP with zero observations")
            return
        self.gp.fit(self.X_obs, self.y_obs)
        self._gp_fitted = True
        logger.info("GP fitted with %d observations", len(self.y_obs))

    def _acquisition(self, x: np.ndarray) -> float:
        """Evaluate the selected acquisition function at a single point.

        Delegates to :meth:`_ei`, :meth:`_pi`, or :meth:`_ucb` based on the
        ``acquisition`` parameter chosen at construction time.

        Args:
            x: Candidate point, shape ``(ndim,)``.

        Returns:
            Acquisition value (higher is better for maximisation).

        Example:
            >>> opt = BayesianOptimizer(bounds=[(0.0, 1.0)], n_warmup=1)
            >>> opt.update(np.array([0.5]), 0.25)
            >>> val = opt._acquisition(np.array([0.3]))
            >>> isinstance(val, float)
            True
        """
        x_2d: np.ndarray = np.atleast_2d(x)
        mu_arr: np.ndarray
        sigma_arr: np.ndarray
        mu_arr, sigma_arr = self.gp.predict(x_2d, return_std=True)
        mu: float = float(mu_arr[0])
        sigma: float = float(sigma_arr[0])

        if self.acquisition_name == "EI":
            return self._ei(mu, sigma)
        if self.acquisition_name == "PI":
            return self._pi(mu, sigma)
        return self._ucb(mu, sigma)

    def _ei(self, mu: float, sigma: float) -> float:
        """Expected Improvement acquisition function (for minimisation).

        Computes ``EI(x) = (y_best - mu - xi) * Phi(Z) + sigma * phi(Z)``
        where ``Z = (y_best - mu - xi) / sigma``.

        Args:
            mu: GP posterior mean at the candidate point.
            sigma: GP posterior standard deviation at the candidate point.

        Returns:
            Expected improvement value (non-negative).

        Example:
            >>> opt = BayesianOptimizer(bounds=[(0.0, 1.0)], n_warmup=1)
            >>> opt.update(np.array([0.5]), 0.25)
            >>> ei_val = opt._ei(0.3, 0.1)
            >>> ei_val >= 0.0
            True
        """
        if sigma <= 0.0:
            return 0.0
        y_best: float = float(np.min(self.y_obs))
        z: float = (y_best - mu - self.xi) / sigma
        ei: float = (y_best - mu - self.xi) * float(norm.cdf(z)) + sigma * float(
            norm.pdf(z)
        )
        return max(ei, 0.0)

    def _pi(self, mu: float, sigma: float) -> float:
        """Probability of Improvement acquisition function (for minimisation).

        Computes ``PI(x) = Phi((y_best - mu - xi) / sigma)``.

        Args:
            mu: GP posterior mean at the candidate point.
            sigma: GP posterior standard deviation at the candidate point.

        Returns:
            Probability of improvement in [0, 1].

        Example:
            >>> opt = BayesianOptimizer(bounds=[(0.0, 1.0)], n_warmup=1)
            >>> opt.update(np.array([0.5]), 0.25)
            >>> 0.0 <= opt._pi(0.3, 0.1) <= 1.0
            True
        """
        if sigma <= 0.0:
            return 0.0
        y_best: float = float(np.min(self.y_obs))
        z: float = (y_best - mu - self.xi) / sigma
        return float(norm.cdf(z))

    def _ucb(self, mu: float, sigma: float) -> float:
        """Upper Confidence Bound acquisition function (for minimisation).

        Computes ``UCB(x) = -(mu - kappa * sigma)`` so that maximising this
        value corresponds to minimising the objective with an exploration bonus.

        Args:
            mu: GP posterior mean at the candidate point.
            sigma: GP posterior standard deviation at the candidate point.

        Returns:
            Negative lower confidence bound (higher means more promising).

        Example:
            >>> opt = BayesianOptimizer(bounds=[(0.0, 1.0)], n_warmup=1)
            >>> opt.update(np.array([0.5]), 0.25)
            >>> isinstance(opt._ucb(0.3, 0.1), float)
            True
        """
        return -(mu - self.kappa * sigma)
