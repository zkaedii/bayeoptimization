"""Comprehensive tests for the active learning and drift detection module.

Covers ActiveLearning (greedy diversity batch selection, step, query history)
and DriftDetector (set_reference, update, drift phases, retrain signal,
window rotation) with edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.active.learner import ActiveLearning
from src.active.drift import DriftDetector, DriftReport


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def al() -> ActiveLearning:
    """Default ActiveLearning instance."""
    return ActiveLearning(w_uncertainty=0.6, w_diversity=0.4)


@pytest.fixture
def pool_2d() -> np.ndarray:
    """A small 2-D pool with well-separated points."""
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
        [5.0, 5.0],
    ])


@pytest.fixture
def dd() -> DriftDetector:
    """Default DriftDetector instance."""
    return DriftDetector(window_size=200, threshold_drift=0.1, threshold_critical=0.3, n_bins=10)


# ======================================================================
# ActiveLearning — select_batch
# ======================================================================

class TestSelectBatch:
    """Tests for ActiveLearning.select_batch with greedy diversity."""

    def test_returns_correct_number_of_indices(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.random.rand(pool_2d.shape[0])
        result = al.select_batch(pool_2d, uncert, k=3)
        assert len(result["selected_indices"]) == 3
        assert len(result["scores"]) == 3

    def test_selected_indices_are_unique(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.random.rand(pool_2d.shape[0])
        result = al.select_batch(pool_2d, uncert, k=5)
        assert len(set(result["selected_indices"])) == 5

    def test_first_pick_is_highest_uncertainty(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.6])
        result = al.select_batch(pool_2d, uncert, k=1)
        assert result["selected_indices"][0] == 3  # highest uncertainty

    def test_batch_diversity_min_pairwise_distance_positive(self, al: ActiveLearning) -> None:
        """Selected batch points must have min pairwise distance > 0."""
        rng = np.random.default_rng(42)
        pool = rng.standard_normal((100, 5))
        uncert = rng.random(100)
        result = al.select_batch(pool, uncert, k=10)

        selected_points = pool[result["selected_indices"]]
        n = selected_points.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(selected_points[i] - selected_points[j])
                assert dist > 0, f"Points {i} and {j} overlap."

    def test_greedy_diversity_spreads_points(self, al: ActiveLearning) -> None:
        """With two tight clusters, greedy diversity should pick from both."""
        cluster_a = np.zeros((5, 2))
        cluster_b = np.full((5, 2), 100.0)
        pool = np.vstack([cluster_a, cluster_b])
        # Uniform uncertainty so diversity dominates subsequent picks
        uncert = np.ones(10)
        result = al.select_batch(pool, uncert, k=2)
        idxs = result["selected_indices"]
        # One from each cluster
        from_a = any(i < 5 for i in idxs)
        from_b = any(i >= 5 for i in idxs)
        assert from_a and from_b

    def test_k_equals_pool_size(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.random.rand(pool_2d.shape[0])
        result = al.select_batch(pool_2d, uncert, k=pool_2d.shape[0])
        assert sorted(result["selected_indices"]) == list(range(pool_2d.shape[0]))

    def test_k_equals_one(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        result = al.select_batch(pool_2d, uncert, k=1)
        assert result["selected_indices"] == [3]
        assert len(result["scores"]) == 1

    def test_uniform_uncertainties(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        """When all uncertainties are equal, normalisation falls back to ones."""
        uncert = np.full(pool_2d.shape[0], 0.5)
        result = al.select_batch(pool_2d, uncert, k=3)
        assert len(result["selected_indices"]) == 3

    def test_scores_are_floats(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.random.rand(pool_2d.shape[0])
        result = al.select_batch(pool_2d, uncert, k=4)
        for s in result["scores"]:
            assert isinstance(s, float)

    def test_scores_keys(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.random.rand(pool_2d.shape[0])
        result = al.select_batch(pool_2d, uncert, k=2)
        assert "selected_indices" in result
        assert "scores" in result


class TestSelectBatchEdgeCases:
    """Edge cases and error handling for select_batch."""

    def test_k_zero_raises(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.random.rand(pool_2d.shape[0])
        with pytest.raises(ValueError, match="k must be a positive integer"):
            al.select_batch(pool_2d, uncert, k=0)

    def test_k_negative_raises(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.random.rand(pool_2d.shape[0])
        with pytest.raises(ValueError, match="k must be a positive integer"):
            al.select_batch(pool_2d, uncert, k=-1)

    def test_k_exceeds_pool_raises(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.random.rand(pool_2d.shape[0])
        with pytest.raises(ValueError, match="exceeds pool size"):
            al.select_batch(pool_2d, uncert, k=pool_2d.shape[0] + 1)

    def test_mismatched_uncertainty_length_raises(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.random.rand(pool_2d.shape[0] + 3)
        with pytest.raises(ValueError, match="uncertainties length must match pool size"):
            al.select_batch(pool_2d, uncert, k=1)

    def test_single_sample_pool(self, al: ActiveLearning) -> None:
        pool = np.array([[1.0, 2.0]])
        uncert = np.array([0.9])
        result = al.select_batch(pool, uncert, k=1)
        assert result["selected_indices"] == [0]

    def test_negative_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="Weights must be non-negative"):
            ActiveLearning(w_uncertainty=-1.0, w_diversity=0.5)

    def test_negative_diversity_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="Weights must be non-negative"):
            ActiveLearning(w_uncertainty=0.5, w_diversity=-0.1)

    def test_zero_weights_allowed(self) -> None:
        al = ActiveLearning(w_uncertainty=0.0, w_diversity=0.0)
        pool = np.array([[1.0], [2.0], [3.0]])
        uncert = np.array([0.1, 0.5, 0.9])
        result = al.select_batch(pool, uncert, k=2)
        assert len(result["selected_indices"]) == 2

    def test_two_samples_pool(self, al: ActiveLearning) -> None:
        pool = np.array([[0.0, 0.0], [100.0, 100.0]])
        uncert = np.array([0.5, 0.5])
        result = al.select_batch(pool, uncert, k=2)
        assert sorted(result["selected_indices"]) == [0, 1]

    def test_identical_points_in_pool(self, al: ActiveLearning) -> None:
        """Pool with duplicate points should still return k indices."""
        pool = np.ones((5, 3))
        uncert = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        result = al.select_batch(pool, uncert, k=3)
        assert len(result["selected_indices"]) == 3
        assert len(set(result["selected_indices"])) == 3


# ======================================================================
# ActiveLearning — step and query history
# ======================================================================

class TestStepAndHistory:
    """Tests for ActiveLearning.step and get_query_history."""

    def test_initial_history_empty(self, al: ActiveLearning) -> None:
        assert al.get_query_history() == []

    def test_step_records_round(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.random.rand(pool_2d.shape[0])
        result = al.select_batch(pool_2d, uncert, k=2)
        al.step(result["selected_indices"])
        history = al.get_query_history()
        assert len(history) == 1
        assert history[0]["round"] == 1
        assert history[0]["selected_indices"] == result["selected_indices"]

    def test_multiple_steps(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.random.rand(pool_2d.shape[0])
        for i in range(3):
            result = al.select_batch(pool_2d, uncert, k=2)
            al.step(result["selected_indices"])
        history = al.get_query_history()
        assert len(history) == 3
        assert [h["round"] for h in history] == [1, 2, 3]

    def test_step_stores_scores(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        uncert = np.random.rand(pool_2d.shape[0])
        result = al.select_batch(pool_2d, uncert, k=3)
        al.step(result["selected_indices"])
        history = al.get_query_history()
        assert history[0]["scores"] == result["scores"]

    def test_step_without_prior_select(self, al: ActiveLearning) -> None:
        """step() called before select_batch should produce empty scores."""
        al.step([0, 1, 2])
        history = al.get_query_history()
        assert history[0]["scores"] == []

    def test_history_is_copy(self, al: ActiveLearning, pool_2d: np.ndarray) -> None:
        """get_query_history returns a copy, not the internal list."""
        uncert = np.random.rand(pool_2d.shape[0])
        result = al.select_batch(pool_2d, uncert, k=2)
        al.step(result["selected_indices"])
        history1 = al.get_query_history()
        history2 = al.get_query_history()
        assert history1 is not history2
        assert history1 == history2


# ======================================================================
# DriftReport dataclass
# ======================================================================

class TestDriftReport:
    """Tests for the DriftReport dataclass."""

    def test_fields(self) -> None:
        report = DriftReport(
            phase="STABLE",
            kl_score=0.05,
            samples_seen=100,
            window_kl_history=[0.02, 0.05],
            triggered_at=None,
        )
        assert report.phase == "STABLE"
        assert report.kl_score == 0.05
        assert report.samples_seen == 100
        assert report.window_kl_history == [0.02, 0.05]
        assert report.triggered_at is None

    def test_triggered_at_set(self) -> None:
        report = DriftReport("CRITICAL", 0.5, 300, [0.1, 0.5], 300)
        assert report.triggered_at == 300

    def test_equality(self) -> None:
        r1 = DriftReport("DRIFT", 0.15, 200, [0.15], None)
        r2 = DriftReport("DRIFT", 0.15, 200, [0.15], None)
        assert r1 == r2

    def test_repr(self) -> None:
        report = DriftReport("STABLE", 0.01, 50, [], None)
        r = repr(report)
        assert "STABLE" in r
        assert "0.01" in r


# ======================================================================
# DriftDetector — set_reference and update
# ======================================================================

class TestDriftDetectorSetReference:
    """Tests for DriftDetector.set_reference."""

    def test_set_reference_resets_state(self, dd: DriftDetector) -> None:
        ref = np.random.randn(200, 3)
        dd.set_reference(ref)
        assert dd.retrain_signal is False

    def test_set_reference_trims_to_window(self) -> None:
        det = DriftDetector(window_size=50)
        data = np.random.randn(100, 2)
        det.set_reference(data)
        # Internal reference should be last 50 rows
        assert det._reference_window.shape[0] == 50
        np.testing.assert_array_equal(det._reference_window, data[-50:])

    def test_set_reference_smaller_than_window(self) -> None:
        det = DriftDetector(window_size=200)
        data = np.random.randn(50, 2)
        det.set_reference(data)
        assert det._reference_window.shape[0] == 50

    def test_update_without_reference_raises(self, dd: DriftDetector) -> None:
        with pytest.raises(RuntimeError, match="Call set_reference"):
            dd.update(np.random.randn(10, 3))


class TestDriftDetectorUpdate:
    """Tests for DriftDetector.update."""

    def test_stable_phase_on_same_distribution(self) -> None:
        det = DriftDetector(window_size=1000, threshold_drift=0.1, threshold_critical=0.3, n_bins=10)
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((1000, 4))
        det.set_reference(ref)
        batch = rng.standard_normal((500, 4))
        report = det.update(batch)
        assert report.phase == "STABLE"
        assert report.kl_score < det._threshold_drift

    def test_samples_seen_increments(self, dd: DriftDetector) -> None:
        dd.set_reference(np.random.randn(200, 2))
        dd.update(np.random.randn(30, 2))
        report = dd.update(np.random.randn(20, 2))
        assert report.samples_seen == 50

    def test_kl_history_grows(self, dd: DriftDetector) -> None:
        dd.set_reference(np.random.randn(200, 2))
        dd.update(np.random.randn(10, 2))
        dd.update(np.random.randn(10, 2))
        report = dd.update(np.random.randn(10, 2))
        assert len(report.window_kl_history) == 3

    def test_current_window_trimmed(self) -> None:
        det = DriftDetector(window_size=50)
        det.set_reference(np.random.randn(50, 2))
        # Send more than window_size
        det.update(np.random.randn(60, 2))
        assert det._current_window.shape[0] == 50

    def test_report_is_drift_report_instance(self, dd: DriftDetector) -> None:
        dd.set_reference(np.random.randn(200, 2))
        report = dd.update(np.random.randn(20, 2))
        assert isinstance(report, DriftReport)


# ======================================================================
# DriftDetector — phase transitions
# ======================================================================

class TestDriftPhases:
    """Tests for STABLE, DRIFT, and CRITICAL phase detection."""

    def test_drift_phase(self) -> None:
        """Feeding data from a shifted distribution should trigger DRIFT."""
        det = DriftDetector(
            window_size=500,
            threshold_drift=0.05,
            threshold_critical=5.0,
            n_bins=10,
        )
        rng = np.random.default_rng(7)
        ref = rng.standard_normal((500, 3))
        det.set_reference(ref)
        # Shift mean enough to exceed drift but not critical
        shifted = rng.standard_normal((500, 3)) + 1.0
        report = det.update(shifted)
        assert report.phase in ("DRIFT", "CRITICAL")
        assert report.kl_score >= det._threshold_drift

    def test_critical_phase(self) -> None:
        """Large distribution shift should trigger CRITICAL."""
        det = DriftDetector(
            window_size=300,
            threshold_drift=0.05,
            threshold_critical=0.3,
            n_bins=10,
        )
        rng = np.random.default_rng(11)
        ref = rng.standard_normal((300, 2))
        det.set_reference(ref)
        # Very large shift
        shifted = rng.standard_normal((300, 2)) + 10.0
        report = det.update(shifted)
        assert report.phase == "CRITICAL"
        assert report.kl_score >= det._threshold_critical

    def test_stable_returns_to_stable(self) -> None:
        """After set_reference reset, same-distribution data is STABLE."""
        det = DriftDetector(window_size=1000, threshold_drift=0.1, threshold_critical=5.0)
        rng = np.random.default_rng(1)
        ref = rng.standard_normal((1000, 2))
        det.set_reference(ref)
        # First: drifted
        det.update(rng.standard_normal((1000, 2)) + 3.0)
        # Reset reference and send similar data
        det.set_reference(ref)
        report = det.update(rng.standard_normal((1000, 2)))
        assert report.phase == "STABLE"

    def test_critical_sets_triggered_at(self) -> None:
        det = DriftDetector(window_size=100, threshold_drift=0.01, threshold_critical=0.1)
        rng = np.random.default_rng(3)
        det.set_reference(rng.standard_normal((100, 2)))
        det.update(rng.standard_normal((100, 2)) + 10.0)
        assert det._triggered_at is not None

    def test_triggered_at_only_set_once(self) -> None:
        det = DriftDetector(window_size=100, threshold_drift=0.01, threshold_critical=0.1)
        rng = np.random.default_rng(5)
        det.set_reference(rng.standard_normal((100, 2)))
        r1 = det.update(rng.standard_normal((50, 2)) + 10.0)
        first_trigger = r1.triggered_at
        r2 = det.update(rng.standard_normal((50, 2)) + 10.0)
        assert r2.triggered_at == first_trigger


# ======================================================================
# DriftDetector — retrain signal
# ======================================================================

class TestRetrainSignal:
    """Tests for retrain_signal property."""

    def test_retrain_signal_false_initially(self, dd: DriftDetector) -> None:
        assert dd.retrain_signal is False

    def test_retrain_signal_true_after_critical(self) -> None:
        det = DriftDetector(window_size=100, threshold_drift=0.01, threshold_critical=0.1)
        rng = np.random.default_rng(9)
        det.set_reference(rng.standard_normal((100, 2)))
        det.update(rng.standard_normal((100, 2)) + 10.0)
        assert det.retrain_signal is True

    def test_retrain_signal_false_when_stable(self) -> None:
        det = DriftDetector(window_size=1000, threshold_drift=0.1, threshold_critical=0.3, n_bins=10)
        rng = np.random.default_rng(0)
        det.set_reference(rng.standard_normal((1000, 3)))
        det.update(rng.standard_normal((500, 3)))
        assert det.retrain_signal is False


# ======================================================================
# DriftDetector — window rotation (acknowledge_retrain)
# ======================================================================

class TestWindowRotation:
    """Tests for acknowledge_retrain (window rotation)."""

    def test_acknowledge_clears_retrain_signal(self) -> None:
        det = DriftDetector(window_size=100, threshold_drift=0.01, threshold_critical=0.1)
        rng = np.random.default_rng(2)
        det.set_reference(rng.standard_normal((100, 2)))
        det.update(rng.standard_normal((100, 2)) + 10.0)
        assert det.retrain_signal is True
        det.acknowledge_retrain()
        assert det.retrain_signal is False

    def test_acknowledge_resets_phase_to_stable(self) -> None:
        det = DriftDetector(window_size=100, threshold_drift=0.01, threshold_critical=0.1)
        rng = np.random.default_rng(4)
        det.set_reference(rng.standard_normal((100, 2)))
        det.update(rng.standard_normal((100, 2)) + 10.0)
        assert det._phase == "CRITICAL"
        det.acknowledge_retrain()
        assert det._phase == "STABLE"

    def test_acknowledge_shifts_reference_window(self) -> None:
        det = DriftDetector(window_size=100, threshold_drift=0.01, threshold_critical=0.1)
        rng = np.random.default_rng(6)
        det.set_reference(rng.standard_normal((100, 2)))
        new_data = rng.standard_normal((100, 2)) + 10.0
        det.update(new_data)
        det.acknowledge_retrain()
        # Reference should now be the current window
        np.testing.assert_array_equal(det._reference_window, det._current_window)

    def test_acknowledge_without_current_window(self, dd: DriftDetector) -> None:
        """acknowledge_retrain before any update should not crash."""
        dd.set_reference(np.random.randn(200, 3))
        dd.acknowledge_retrain()
        assert dd.retrain_signal is False

    def test_after_acknowledge_same_distribution_stable(self) -> None:
        """After window rotation, same-distribution data should be STABLE."""
        det = DriftDetector(window_size=1000, threshold_drift=0.1, threshold_critical=0.3)
        rng = np.random.default_rng(8)
        det.set_reference(rng.standard_normal((1000, 2)))
        shifted = rng.standard_normal((1000, 2)) + 10.0
        det.update(shifted)
        det.acknowledge_retrain()
        # Now send data from the shifted distribution (matches new reference)
        report = det.update(rng.standard_normal((1000, 2)) + 10.0)
        assert report.phase == "STABLE"


# ======================================================================
# DriftDetector — KL computation internals
# ======================================================================

class TestKLComputation:
    """Tests for the internal _compute_kl method."""

    def test_kl_identical_distributions_near_zero(self, dd: DriftDetector) -> None:
        rng = np.random.default_rng(10)
        data = rng.standard_normal((200, 3))
        dd.set_reference(data)
        kl = dd._compute_kl(data, data)
        assert kl < 0.01

    def test_kl_shifted_distributions_positive(self, dd: DriftDetector) -> None:
        rng = np.random.default_rng(12)
        ref = rng.standard_normal((200, 3))
        cur = rng.standard_normal((200, 3)) + 5.0
        kl = dd._compute_kl(ref, cur)
        assert kl > 0.0

    def test_kl_single_feature(self) -> None:
        det = DriftDetector(window_size=100, n_bins=5)
        rng = np.random.default_rng(13)
        ref = rng.standard_normal((100, 1))
        cur = rng.standard_normal((100, 1)) + 3.0
        kl = det._compute_kl(ref, cur)
        assert kl > 0.0


# ======================================================================
# Integration: ActiveLearning + DriftDetector together
# ======================================================================

class TestIntegration:
    """End-to-end workflow combining active learning and drift detection."""

    def test_full_workflow(self) -> None:
        rng = np.random.default_rng(42)
        al = ActiveLearning(w_uncertainty=0.7, w_diversity=0.3)
        dd = DriftDetector(window_size=1000, threshold_drift=0.1, threshold_critical=0.3)

        pool = rng.standard_normal((1000, 4))
        dd.set_reference(pool)

        # Round 1: select batch from same distribution — should be STABLE
        uncert = rng.random(pool.shape[0])
        result = al.select_batch(pool, uncert, k=5)
        al.step(result["selected_indices"])

        stable_batch = rng.standard_normal((500, 4))
        report = dd.update(stable_batch)
        assert report.phase == "STABLE"

        # Round 2: big distribution shift triggers CRITICAL
        drifted_pool = rng.standard_normal((200, 4)) + 20.0
        uncert2 = rng.random(200)
        result2 = al.select_batch(drifted_pool, uncert2, k=5)
        al.step(result2["selected_indices"])

        report2 = dd.update(rng.standard_normal((1000, 4)) + 20.0)
        assert report2.phase == "CRITICAL"
        assert dd.retrain_signal is True

        # Acknowledge and verify
        dd.acknowledge_retrain()
        assert dd.retrain_signal is False
        assert len(al.get_query_history()) == 2

    def test_diversity_preserves_spread_in_high_dim(self) -> None:
        """In higher dimensions, selected batch should remain spread out."""
        rng = np.random.default_rng(99)
        al = ActiveLearning(w_uncertainty=0.3, w_diversity=0.7)
        pool = rng.standard_normal((500, 20))
        uncert = rng.random(500)
        result = al.select_batch(pool, uncert, k=15)

        selected = pool[result["selected_indices"]]
        dists = []
        for i in range(selected.shape[0]):
            for j in range(i + 1, selected.shape[0]):
                dists.append(float(np.linalg.norm(selected[i] - selected[j])))
        assert min(dists) > 0
