"""Optimization module for the ZKAEDI platform.

Exports the core Bayesian optimization classes:
- BayesianOptimizer: GP-based surrogate model with acquisition function suite.
- ZkaediPrimeBO: Extends BayesianOptimizer with recursive Hamiltonian dynamics.
"""

from src.optimization.bayesian import BayesianOptimizer
from src.optimization.prime_bo import ZkaediPrimeBO

__all__: list[str] = ["BayesianOptimizer", "ZkaediPrimeBO"]
