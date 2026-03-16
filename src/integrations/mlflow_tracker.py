"""MLflow tracking integration for Bayesian optimization and evidential experiments.

Provides a high-level wrapper around the MLflow tracking API with
convenience methods for logging BO iterations, PRIME field state, and
model artifacts.  Supports context-manager usage for automatic run
lifecycle management.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Full MLflow integration for BO and evidential experiments.

    Wraps the MLflow tracking client to provide structured logging helpers
    for Gaussian-process hyperparameters, acquisition values, and PRIME
    Hamiltonian field state.  Supports context-manager usage so that runs
    are automatically ended on scope exit.

    Params:
        tracking_uri: MLflow tracking server URI.  Read from the
            ``MLFLOW_TRACKING_URI`` environment variable when *None*.
        artifact_location: Optional root artifact store location.

    Returns:
        An ``MLflowTracker`` instance.

    Example::

        with MLflowTracker(tracking_uri="http://localhost:5000") as tracker:
            exp_id = tracker.start_experiment("my_experiment")
            run_id = tracker.start_run("run-001", tags={"team": "research"})
            tracker.log_params({"lr": 0.01})
            tracker.log_metrics({"loss": 0.5}, step=1)
            tracker.end_run()
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ) -> None:
        """Initialise the MLflow tracker.

        Params:
            tracking_uri: MLflow tracking server URI.
            artifact_location: Root artifact store location.

        Returns:
            None

        Example::

            tracker = MLflowTracker(tracking_uri="http://mlflow:5000")
        """
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        self._tracking_uri: Optional[str] = tracking_uri
        self._artifact_location: Optional[str] = artifact_location
        self._client: MlflowClient = MlflowClient()
        self._experiment_id: Optional[str] = None
        self._run_id: Optional[str] = None
        logger.info(
            "MLflowTracker initialised (tracking_uri=%s)",
            tracking_uri or "env/default",
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> MLflowTracker:
        """Enter the context manager.

        Returns:
            The ``MLflowTracker`` instance.

        Example::

            with MLflowTracker() as tracker:
                tracker.start_experiment("exp")
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the context manager, ending any active run.

        Params:
            exc_type: Exception type, if raised inside the block.
            exc_val: Exception value.
            exc_tb: Traceback.

        Returns:
            None

        Example::

            with MLflowTracker() as tracker:
                tracker.start_run("r1")
            # run is ended automatically here
        """
        if self._run_id is not None:
            status = "FAILED" if exc_type is not None else "FINISHED"
            self.end_run(status=status)

    # ------------------------------------------------------------------
    # Experiment / run lifecycle
    # ------------------------------------------------------------------
    def start_experiment(self, name: str) -> str:
        """Create or retrieve an MLflow experiment by *name*.

        Params:
            name: Experiment name.

        Returns:
            The experiment ID as a string.

        Example::

            tracker = MLflowTracker()
            exp_id = tracker.start_experiment("bayesopt_v2")
        """
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is not None:
            self._experiment_id = experiment.experiment_id
        else:
            self._experiment_id = mlflow.create_experiment(
                name, artifact_location=self._artifact_location
            )
        mlflow.set_experiment(name)
        logger.info(
            "Experiment '%s' ready (id=%s)", name, self._experiment_id
        )
        return self._experiment_id

    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new MLflow run inside the current experiment.

        Params:
            run_name: Human-readable run name.
            tags: Optional key-value tags to attach to the run.

        Returns:
            The run ID as a string.

        Example::

            tracker = MLflowTracker()
            tracker.start_experiment("exp")
            run_id = tracker.start_run("run-42", tags={"variant": "A"})
        """
        active_run = mlflow.start_run(
            run_name=run_name,
            experiment_id=self._experiment_id,
            tags=tags,
        )
        self._run_id = active_run.info.run_id
        logger.info("Run started: name=%s id=%s", run_name, self._run_id)
        return self._run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run.

        Params:
            status: Final run status — ``"FINISHED"``, ``"FAILED"``, or
                ``"KILLED"``.

        Returns:
            None

        Example::

            tracker = MLflowTracker()
            tracker.start_experiment("exp")
            tracker.start_run("r")
            tracker.end_run(status="FINISHED")
        """
        if self._run_id is None:
            logger.warning("end_run called but no active run exists")
            return
        mlflow.end_run(status=status)
        logger.info("Run ended: id=%s status=%s", self._run_id, status)
        self._run_id = None

    # ------------------------------------------------------------------
    # Generic logging
    # ------------------------------------------------------------------
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a batch of parameters to the active run.

        Params:
            params: Mapping of parameter names to values.

        Raises:
            RuntimeError: If no active run exists.

        Returns:
            None

        Example::

            tracker.log_params({"learning_rate": 0.01, "n_iter": 100})
        """
        if self._run_id is None:
            raise RuntimeError("Call start_run() before log_params().")
        mlflow.log_params(params)
        logger.debug("Logged %d params", len(params))

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log a batch of metrics at a given step.

        Params:
            metrics: Mapping of metric names to numeric values.
            step: Integer step / iteration index.

        Returns:
            None

        Example::

            tracker.log_metrics({"rmse": 0.12, "r2": 0.95}, step=10)
        """
        if self._run_id is None:
            raise RuntimeError("Call start_run() before log_metrics().")
        mlflow.log_metrics(metrics, step=step)
        logger.debug("Logged %d metrics at step %d", len(metrics), step)

    # ------------------------------------------------------------------
    # BO-specific logging
    # ------------------------------------------------------------------
    def log_bo_step(
        self,
        step: int,
        x_candidate: Union[List[float], Sequence[float]],
        y_observed: float,
        acquisition_value: float,
        gp_hyperparams: Dict[str, float],
    ) -> None:
        """Log a single Bayesian-optimisation iteration.

        Records the candidate point, observed value, acquisition function
        output, and Gaussian-process hyperparameters.

        Params:
            step: Iteration index.
            x_candidate: The candidate input proposed by the acquisition fn.
            y_observed: The objective value observed at *x_candidate*.
            acquisition_value: Acquisition function value at *x_candidate*.
            gp_hyperparams: GP hyperparameters dict with keys
                ``length_scale``, ``amplitude``, ``noise_alpha``.

        Returns:
            None

        Example::

            tracker.log_bo_step(
                step=5,
                x_candidate=[0.3, 0.7],
                y_observed=1.23,
                acquisition_value=0.45,
                gp_hyperparams={
                    "length_scale": 0.5,
                    "amplitude": 1.0,
                    "noise_alpha": 0.01,
                },
            )
        """
        metrics: Dict[str, float] = {
            "y_observed": y_observed,
            "acquisition_value": acquisition_value,
            "gp_length_scale": gp_hyperparams.get("length_scale", 0.0),
            "gp_amplitude": gp_hyperparams.get("amplitude", 0.0),
            "gp_noise_alpha": gp_hyperparams.get("noise_alpha", 0.0),
        }
        if self._run_id is None:
            raise RuntimeError("Call start_run() before log_bo_step().")
        mlflow.log_metrics(metrics, step=step)

        # Store candidate as a tag (tags are upsert-safe, unlike params).
        mlflow.set_tag(f"x_candidate_step_{step}", str(list(x_candidate)))
        logger.debug("Logged BO step %d", step)

    # ------------------------------------------------------------------
    # PRIME-specific logging
    # ------------------------------------------------------------------
    def log_prime_state(
        self,
        step: int,
        phase: str,
        variance: float,
        eta: float,
        gamma: float,
        beta: float,
    ) -> None:
        """Log PRIME Hamiltonian field state at a given step.

        Params:
            step: Iteration index.
            phase: Current PRIME phase label.
            variance: Field variance.
            eta: Eta coupling constant.
            gamma: Gamma damping factor.
            beta: Beta inverse temperature.

        Returns:
            None

        Example::

            tracker.log_prime_state(
                step=10,
                phase="exploration",
                variance=0.42,
                eta=0.1,
                gamma=0.9,
                beta=1.5,
            )
        """
        if self._run_id is None:
            raise RuntimeError("Call start_run() before log_prime_state().")
        metrics: Dict[str, float] = {
            "prime_variance": variance,
            "prime_eta": eta,
            "prime_gamma": gamma,
            "prime_beta": beta,
        }
        mlflow.log_metrics(metrics, step=step)
        mlflow.set_tag("prime_phase", phase)
        logger.debug("Logged PRIME state at step %d (phase=%s)", step, phase)

    # ------------------------------------------------------------------
    # Model artifacts
    # ------------------------------------------------------------------
    def log_model(self, model: Any, artifact_path: str) -> None:
        """Log a trained model as an MLflow artifact.

        Attempts to detect whether *model* is a scikit-learn or PyTorch model
        and delegates to the appropriate ``mlflow.<flavour>.log_model`` call.

        Params:
            model: A trained model instance (sklearn or torch).
            artifact_path: Sub-path inside the run's artifact directory.

        Returns:
            None

        Example::

            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor().fit(X, y)
            tracker.log_model(gp, "gp_surrogate")
        """
        model_type: str = type(model).__module__

        if "sklearn" in model_type:
            mlflow.sklearn.log_model(model, artifact_path)
            logger.info("Logged sklearn model to '%s'", artifact_path)
        elif "torch" in model_type:
            mlflow.pytorch.log_model(model, artifact_path)
            logger.info("Logged PyTorch model to '%s'", artifact_path)
        else:
            # Fallback: pickle via the generic pyfunc flavour
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
            )
            logger.info(
                "Logged model (type=%s) via pyfunc to '%s'",
                type(model).__name__,
                artifact_path,
            )
