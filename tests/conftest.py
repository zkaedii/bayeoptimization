"""Shared fixtures for the optimization test suite."""

from __future__ import annotations

import numpy as np
import pytest

from src.optimization.bayesian import BayesianOptimizer
from src.optimization.prime_bo import ZkaediPrimeBO


def ackley_2d(x: np.ndarray) -> float:
    """Evaluate the 2-D Ackley function (global minimum = 0 at origin).

    Args:
        x: Input array of shape (2,).

    Returns:
        Ackley function value.
    """
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    return float(
        -a * np.exp(-b * np.sqrt(sum_sq / d))
        - np.exp(sum_cos / d)
        + a
        + np.e
    )


@pytest.fixture()
def bo_1d() -> BayesianOptimizer:
    """A 1-D BayesianOptimizer with small warmup for fast tests."""
    return BayesianOptimizer(
        bounds=[(-5.0, 5.0)],
        kernel="rbf",
        acquisition="EI",
        n_warmup=3,
    )


@pytest.fixture()
def bo_2d() -> BayesianOptimizer:
    """A 2-D BayesianOptimizer with default settings."""
    return BayesianOptimizer(
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        kernel="matern52",
        acquisition="EI",
        n_warmup=5,
    )


@pytest.fixture()
def prime_1d() -> ZkaediPrimeBO:
    """A 1-D ZkaediPrimeBO with small warmup and few evolution steps."""
    return ZkaediPrimeBO(
        bounds=[(-5.0, 5.0)],
        kernel="matern52",
        acquisition="EI",
        n_warmup=3,
        t_max=10,
    )


@pytest.fixture()
def prime_2d() -> ZkaediPrimeBO:
    """A 2-D ZkaediPrimeBO for convergence and phase tests."""
    return ZkaediPrimeBO(
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        kernel="matern52",
        acquisition="EI",
        n_warmup=5,
        t_max=10,
    )


@pytest.fixture()
def fitted_bo_1d() -> BayesianOptimizer:
    """A 1-D BayesianOptimizer that has completed warmup and is GP-fitted."""
    rng = np.random.RandomState(42)
    opt = BayesianOptimizer(
        bounds=[(-5.0, 5.0)],
        kernel="rbf",
        acquisition="EI",
        n_warmup=3,
    )
    for _ in range(3):
        x = rng.uniform(-5.0, 5.0, size=(1,))
        opt.update(x, float(x[0] ** 2))
    return opt


@pytest.fixture()
def fitted_prime_1d() -> ZkaediPrimeBO:
    """A 1-D ZkaediPrimeBO that has completed warmup and is GP-fitted."""
    rng = np.random.RandomState(42)
    opt = ZkaediPrimeBO(
        bounds=[(-5.0, 5.0)],
        kernel="matern52",
        acquisition="EI",
        n_warmup=3,
        t_max=5,
    )
    for _ in range(3):
        x = rng.uniform(-5.0, 5.0, size=(1,))
        opt.update(x, float(x[0] ** 2))
    return opt
