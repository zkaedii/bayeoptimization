"""Comprehensive tests for the evidential module.

Covers EvidentialClassifier, EvidentialRegressor, MultimodalFusion,
and OpenSetRecognition with 95%+ line coverage.
"""

import math

import numpy as np
import pytest
from scipy.special import gammaln

from src.evidential.classifier import EvidentialClassifier
from src.evidential.regressor import EvidentialRegressor, _softplus
from src.evidential.fusion import FusionPhase, MultimodalFusion
from src.evidential.openset import OpenSetRecognition


# ======================================================================
# EvidentialClassifier
# ======================================================================


class TestEvidentialClassifier:
    """Tests for EvidentialClassifier."""

    # --- Construction ---

    def test_init_valid(self):
        clf = EvidentialClassifier(n_classes=3, kl_weight=0.01)
        assert clf.n_classes == 3
        assert clf.kl_weight == 0.01
        assert clf.kl_annealing is True
        assert clf.anneal_steps == 100

    def test_init_invalid_n_classes(self):
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            EvidentialClassifier(n_classes=1)

    def test_init_no_annealing(self):
        clf = EvidentialClassifier(n_classes=2, kl_annealing=False)
        assert clf.kl_annealing is False

    # --- forward ---

    def test_forward_basic(self):
        clf = EvidentialClassifier(n_classes=3)
        result = clf.forward(np.array([5.0, 2.0, 1.0]))
        assert result["alpha"].shape == (3,)
        np.testing.assert_allclose(result["alpha"], [6.0, 3.0, 2.0])
        assert result["dirichlet_strength"] == pytest.approx(11.0)
        np.testing.assert_allclose(result["probs"], [6 / 11, 3 / 11, 2 / 11])
        assert 0.0 <= result["vacuity"] <= 1.0
        assert result["vacuity"] == pytest.approx(3.0 / 11.0)

    def test_forward_uniform_evidence(self):
        clf = EvidentialClassifier(n_classes=4)
        result = clf.forward(np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result["probs"], [0.25, 0.25, 0.25, 0.25])
        assert result["vacuity"] == pytest.approx(1.0)

    def test_forward_clamps_negative(self):
        clf = EvidentialClassifier(n_classes=2)
        result = clf.forward(np.array([-5.0, 3.0]))
        np.testing.assert_allclose(result["alpha"], [1.0, 4.0])

    def test_forward_high_evidence_low_vacuity(self):
        clf = EvidentialClassifier(n_classes=3)
        result = clf.forward(np.array([1000.0, 0.0, 0.0]))
        assert result["vacuity"] < 0.01

    def test_forward_aleatoric_is_entropy(self):
        clf = EvidentialClassifier(n_classes=2)
        result = clf.forward(np.array([1.0, 1.0]))
        probs = result["probs"]
        expected_entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
        assert result["aleatoric"] == pytest.approx(expected_entropy, abs=1e-8)

    def test_forward_epistemic(self):
        clf = EvidentialClassifier(n_classes=3)
        result = clf.forward(np.array([10.0, 0.0, 0.0]))
        # epistemic = 1 - max(alpha)/S = 1 - 11/13
        assert result["epistemic"] == pytest.approx(1.0 - 11.0 / 13.0)

    # --- predict ---

    def test_predict_top_class(self):
        clf = EvidentialClassifier(n_classes=4)
        pred = clf.predict(np.array([0.0, 0.0, 12.0, 0.0]))
        assert pred["predicted_class"] == 2
        assert "probs" in pred
        assert "uncertainty" in pred
        assert "epistemic" in pred
        assert "aleatoric" in pred

    def test_predict_matches_forward(self):
        clf = EvidentialClassifier(n_classes=3)
        ev = np.array([3.0, 1.0, 0.5])
        fwd = clf.forward(ev)
        pred = clf.predict(ev)
        assert pred["predicted_class"] == int(np.argmax(fwd["probs"]))
        assert pred["uncertainty"] == pytest.approx(fwd["vacuity"])
        assert pred["epistemic"] == pytest.approx(fwd["epistemic"])
        assert pred["aleatoric"] == pytest.approx(fwd["aleatoric"])

    # --- compute_loss ---

    def test_compute_loss_positive(self):
        clf = EvidentialClassifier(n_classes=3)
        loss = clf.compute_loss(np.array([5.0, 1.0, 0.5]), target=0, step=50)
        assert loss["total_loss"] > 0
        assert loss["nll_loss"] > 0
        assert loss["kl_loss"] >= 0
        assert 0.0 <= loss["annealing_factor"] <= 1.0

    def test_compute_loss_nll_decreases_with_evidence(self):
        clf = EvidentialClassifier(n_classes=3, kl_annealing=False)
        loss_low = clf.compute_loss(np.array([1.0, 0.0, 0.0]), target=0)
        loss_high = clf.compute_loss(np.array([100.0, 0.0, 0.0]), target=0)
        assert loss_high["nll_loss"] < loss_low["nll_loss"]

    def test_compute_loss_wrong_target_higher(self):
        clf = EvidentialClassifier(n_classes=3, kl_weight=0.0)
        loss_correct = clf.compute_loss(np.array([10.0, 0.0, 0.0]), target=0)
        loss_wrong = clf.compute_loss(np.array([10.0, 0.0, 0.0]), target=1)
        assert loss_wrong["nll_loss"] > loss_correct["nll_loss"]

    # --- KL annealing schedule ---

    def test_kl_annealing_zero_at_start(self):
        clf = EvidentialClassifier(n_classes=3, kl_annealing=True, anneal_steps=100)
        loss = clf.compute_loss(np.array([5.0, 1.0, 0.5]), target=0, step=0)
        assert loss["annealing_factor"] == pytest.approx(0.0)

    def test_kl_annealing_half(self):
        clf = EvidentialClassifier(n_classes=3, kl_annealing=True, anneal_steps=100)
        loss = clf.compute_loss(np.array([5.0, 1.0, 0.5]), target=0, step=50)
        assert loss["annealing_factor"] == pytest.approx(0.5)

    def test_kl_annealing_full(self):
        clf = EvidentialClassifier(n_classes=3, kl_annealing=True, anneal_steps=100)
        loss = clf.compute_loss(np.array([5.0, 1.0, 0.5]), target=0, step=100)
        assert loss["annealing_factor"] == pytest.approx(1.0)

    def test_kl_annealing_beyond_steps(self):
        clf = EvidentialClassifier(n_classes=3, kl_annealing=True, anneal_steps=100)
        loss = clf.compute_loss(np.array([5.0, 1.0, 0.5]), target=0, step=200)
        assert loss["annealing_factor"] == pytest.approx(1.0)

    def test_kl_no_annealing(self):
        clf = EvidentialClassifier(n_classes=3, kl_annealing=False)
        loss = clf.compute_loss(np.array([5.0, 1.0, 0.5]), target=0, step=0)
        assert loss["annealing_factor"] == pytest.approx(1.0)

    def test_kl_annealing_zero_anneal_steps(self):
        """anneal_steps=0 should not divide by zero; max(anneal_steps,1) used."""
        clf = EvidentialClassifier(n_classes=3, kl_annealing=True, anneal_steps=0)
        loss = clf.compute_loss(np.array([1.0, 1.0, 1.0]), target=0, step=0)
        # step/max(0,1) = 0/1 = 0, clamped to min(1.0, 0.0) = 0.0
        assert loss["annealing_factor"] == pytest.approx(0.0)

    def test_kl_loss_at_step_affects_total(self):
        """With annealing at step=0, KL contributes nothing; at step=100 it does."""
        clf = EvidentialClassifier(n_classes=3, kl_annealing=True, anneal_steps=100,
                                   kl_weight=1.0)
        ev = np.array([5.0, 1.0, 0.5])
        loss_start = clf.compute_loss(ev, target=0, step=0)
        loss_end = clf.compute_loss(ev, target=0, step=100)
        # At step 0, total = nll only; at step 100, total = nll + kl_weight * kl
        assert loss_start["total_loss"] == pytest.approx(loss_start["nll_loss"])
        assert loss_end["total_loss"] > loss_end["nll_loss"]

    # --- is_high_uncertainty ---

    def test_is_high_uncertainty_true(self):
        clf = EvidentialClassifier(n_classes=3)
        assert clf.is_high_uncertainty(np.array([0.0, 0.0, 0.0]), threshold=0.5) is True

    def test_is_high_uncertainty_false(self):
        clf = EvidentialClassifier(n_classes=3)
        assert clf.is_high_uncertainty(np.array([100.0, 0.0, 0.0]), threshold=0.5) is False

    def test_is_high_uncertainty_boundary(self):
        """Vacuity exactly at threshold should return True (>=)."""
        clf = EvidentialClassifier(n_classes=2)
        # vacuity = K/S = 2/(e0+1+e1+1). Set e0=e1=0 => vacuity=2/2=1.0
        assert clf.is_high_uncertainty(np.array([0.0, 0.0]), threshold=1.0) is True

    # --- _validate_evidence ---

    def test_validate_wrong_shape(self):
        clf = EvidentialClassifier(n_classes=3)
        with pytest.raises(ValueError, match="Evidence must be 1-D"):
            clf.forward(np.array([1.0, 2.0]))  # length 2 != 3

    def test_validate_2d_array(self):
        clf = EvidentialClassifier(n_classes=3)
        with pytest.raises(ValueError, match="Evidence must be 1-D"):
            clf.forward(np.array([[1.0, 2.0, 3.0]]))

    # --- _kl_dirichlet ---

    def test_kl_uniform_is_zero(self):
        clf = EvidentialClassifier(n_classes=3)
        kl = clf._kl_dirichlet(np.array([1.0, 1.0, 1.0]))
        assert kl == pytest.approx(0.0, abs=1e-10)

    def test_kl_non_uniform_positive(self):
        clf = EvidentialClassifier(n_classes=3)
        kl = clf._kl_dirichlet(np.array([5.0, 2.0, 1.0]))
        assert kl > 0.0

    # --- Uncertainty calibration: ECE < 0.05 with well-separated data ---

    def test_uncertainty_calibration_ece(self):
        """ECE should be < 0.05 when data is well-separated (high evidence on correct class)."""
        clf = EvidentialClassifier(n_classes=3)
        rng = np.random.RandomState(42)
        n_samples = 500
        n_bins = 10

        true_labels = rng.randint(0, 3, size=n_samples)
        confidences = []
        predictions = []

        for label in true_labels:
            # Well-separated: high evidence on correct class, near zero elsewhere
            ev = np.zeros(3)
            ev[label] = rng.uniform(50, 200)
            pred = clf.predict(ev)
            predictions.append(pred["predicted_class"])
            confidences.append(np.max(pred["probs"]))

        predictions = np.array(predictions)
        confidences = np.array(confidences)
        correct = (predictions == true_labels).astype(float)

        # Compute ECE
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() / n_samples * abs(bin_acc - bin_conf)

        assert ece < 0.05, f"ECE = {ece:.4f} exceeds 0.05"


# ======================================================================
# EvidentialRegressor
# ======================================================================


class TestSoftplus:
    """Tests for the standalone _softplus helper."""

    def test_softplus_zero(self):
        assert _softplus(0.0) == pytest.approx(math.log(2.0))

    def test_softplus_large(self):
        # x > 20 returns x directly
        assert _softplus(25.0) == 25.0

    def test_softplus_negative(self):
        assert _softplus(-5.0) > 0.0
        assert _softplus(-5.0) == pytest.approx(float(np.log1p(np.exp(-5.0))))

    def test_softplus_boundary(self):
        # x = 20.0 should go through the log1p path (not >20)
        assert _softplus(20.0) == pytest.approx(float(np.log1p(np.exp(20.0))))


class TestEvidentialRegressor:
    """Tests for EvidentialRegressor."""

    # --- Construction ---

    def test_init(self):
        reg = EvidentialRegressor(kl_weight=0.01)
        assert reg.kl_weight == 0.01

    # --- _parse_outputs ---

    def test_parse_outputs_constraints(self):
        reg = EvidentialRegressor()
        g, n, a, b = reg._parse_outputs(np.array([0.0, 0.0, 0.0, 0.0]))
        assert n > 0
        assert a > 1.0
        assert b > 0

    def test_parse_outputs_gamma_passthrough(self):
        reg = EvidentialRegressor()
        g, n, a, b = reg._parse_outputs(np.array([3.5, 1.0, 1.0, 1.0]))
        assert g == 3.5

    def test_parse_wrong_shape(self):
        reg = EvidentialRegressor()
        with pytest.raises(ValueError, match="Outputs must be 1-D with length 4"):
            reg._parse_outputs(np.array([1.0, 2.0]))

    def test_parse_2d(self):
        reg = EvidentialRegressor()
        with pytest.raises(ValueError, match="Outputs must be 1-D with length 4"):
            reg._parse_outputs(np.array([[1.0, 2.0, 3.0, 4.0]]))

    # --- forward ---

    def test_forward_keys(self):
        reg = EvidentialRegressor()
        fwd = reg.forward(np.array([1.0, 0.0, 2.0, 0.5]))
        assert set(fwd.keys()) == {"gamma", "nu", "alpha", "beta",
                                    "aleatoric", "epistemic", "total"}
        assert fwd["gamma"] == 1.0

    def test_forward_uncertainty_positive(self):
        reg = EvidentialRegressor()
        fwd = reg.forward(np.array([0.0, 0.0, 0.0, 0.0]))
        assert fwd["aleatoric"] > 0
        assert fwd["epistemic"] > 0
        assert fwd["total"] == pytest.approx(fwd["aleatoric"] + fwd["epistemic"])

    # --- NIG uncertainty split: aleatoric vs epistemic ---

    def test_nig_uncertainty_split_high_nu_low_epistemic(self):
        """High nu (lots of virtual observations) should reduce epistemic uncertainty."""
        reg = EvidentialRegressor()
        fwd_low_nu = reg.forward(np.array([0.0, -5.0, 2.0, 0.5]))   # small nu
        fwd_high_nu = reg.forward(np.array([0.0, 5.0, 2.0, 0.5]))   # large nu
        assert fwd_high_nu["epistemic"] < fwd_low_nu["epistemic"]

    def test_nig_uncertainty_split_aleatoric_independent_of_nu(self):
        """Aleatoric = beta/(alpha-1) is independent of nu."""
        reg = EvidentialRegressor()
        fwd1 = reg.forward(np.array([0.0, -3.0, 2.0, 0.5]))
        fwd2 = reg.forward(np.array([0.0, 5.0, 2.0, 0.5]))
        assert fwd1["aleatoric"] == pytest.approx(fwd2["aleatoric"], rel=1e-6)

    def test_nig_uncertainty_split_formula(self):
        """Verify aleatoric = beta/(alpha-1) and epistemic = beta/(nu*(alpha-1))."""
        reg = EvidentialRegressor()
        fwd = reg.forward(np.array([2.0, 1.0, 1.5, 0.8]))
        g, nu, alpha, beta = reg._parse_outputs(np.array([2.0, 1.0, 1.5, 0.8]))
        expected_ale = beta / (alpha - 1.0)
        expected_epi = beta / (nu * (alpha - 1.0))
        assert fwd["aleatoric"] == pytest.approx(expected_ale)
        assert fwd["epistemic"] == pytest.approx(expected_epi)

    # --- predict ---

    def test_predict_keys(self):
        reg = EvidentialRegressor()
        pred = reg.predict(np.array([3.0, 1.0, 2.0, 0.5]))
        assert set(pred.keys()) == {"mean", "aleatoric", "epistemic", "total"}
        assert isinstance(pred["mean"], float)

    def test_predict_matches_forward(self):
        reg = EvidentialRegressor()
        outputs = np.array([2.5, 0.0, 1.0, 0.5])
        fwd = reg.forward(outputs)
        pred = reg.predict(outputs)
        assert pred["mean"] == fwd["gamma"]
        assert pred["aleatoric"] == fwd["aleatoric"]
        assert pred["epistemic"] == fwd["epistemic"]

    # --- compute_loss ---

    def test_compute_loss_positive(self):
        reg = EvidentialRegressor()
        loss = reg.compute_loss(np.array([2.0, 0.0, 1.5, 0.3]), target=2.1)
        assert "total_loss" in loss
        assert "nig_loss" in loss
        assert "reg_loss" in loss
        assert loss["reg_loss"] >= 0.0

    def test_compute_loss_zero_residual_low_reg(self):
        """When prediction matches target exactly, reg_loss should be zero."""
        reg = EvidentialRegressor(kl_weight=0.01)
        outputs = np.array([5.0, 1.0, 2.0, 0.5])
        loss = reg.compute_loss(outputs, target=5.0)
        assert loss["reg_loss"] == pytest.approx(0.0)

    def test_compute_loss_increases_with_residual(self):
        reg = EvidentialRegressor()
        loss_close = reg.compute_loss(np.array([5.0, 1.0, 2.0, 0.5]), target=5.1)
        loss_far = reg.compute_loss(np.array([5.0, 1.0, 2.0, 0.5]), target=10.0)
        assert loss_far["reg_loss"] > loss_close["reg_loss"]

    # --- _log_gamma ---

    def test_log_gamma_one(self):
        assert EvidentialRegressor._log_gamma(1.0) == pytest.approx(0.0, abs=1e-10)

    def test_log_gamma_known_value(self):
        # gammaln(5) = log(4!) = log(24)
        assert EvidentialRegressor._log_gamma(5.0) == pytest.approx(
            math.log(24.0), abs=1e-10
        )


# ======================================================================
# MultimodalFusion
# ======================================================================


class TestFusionPhase:
    """Tests for the FusionPhase enum."""

    def test_values(self):
        assert FusionPhase.CONSENSUS.value == "CONSENSUS"
        assert FusionPhase.CONFLICT.value == "CONFLICT"
        assert FusionPhase.UNCERTAIN.value == "UNCERTAIN"

    def test_is_str(self):
        assert isinstance(FusionPhase.CONSENSUS, str)


class TestMultimodalFusion:
    """Tests for MultimodalFusion."""

    # --- Construction ---

    def test_init_valid(self):
        f = MultimodalFusion(n_classes=3)
        assert f.n_classes == 3
        assert f.threshold_consensus == 0.3
        assert f.threshold_uncertain == 0.7

    def test_init_invalid(self):
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            MultimodalFusion(n_classes=1)

    # --- Fusion phases ---

    def test_consensus_phase(self):
        """Both modalities agree on class 0 with high evidence => CONSENSUS."""
        fuser = MultimodalFusion(n_classes=3)
        result = fuser.fuse([
            np.array([10.0, 1.0, 0.5]),
            np.array([8.0, 0.5, 0.2]),
        ])
        assert result["phase"] == "CONSENSUS"
        assert result["fused_uncertainty"] < 0.3

    def test_conflict_phase(self):
        """Modalities disagree on top class => CONFLICT."""
        fuser = MultimodalFusion(n_classes=3)
        result = fuser.fuse([
            np.array([10.0, 0.0, 0.0]),  # top class 0
            np.array([0.0, 10.0, 0.0]),  # top class 1
        ])
        assert result["phase"] == "CONFLICT"
        assert result["modality_agreements"] == [0, 1]

    def test_uncertain_phase(self):
        """Low evidence on all => high uncertainty => UNCERTAIN."""
        fuser = MultimodalFusion(n_classes=3, threshold_uncertain=0.5)
        result = fuser.fuse([
            np.array([0.1, 0.1, 0.1]),
        ])
        assert result["phase"] == "UNCERTAIN"
        assert result["fused_uncertainty"] >= 0.5

    def test_uncertain_phase_between_thresholds(self):
        """Agreement but uncertainty between consensus and uncertain thresholds."""
        fuser = MultimodalFusion(n_classes=3,
                                 threshold_consensus=0.1,
                                 threshold_uncertain=0.9)
        # Single modality with moderate evidence; same top class
        result = fuser.fuse([np.array([2.0, 0.5, 0.5])])
        # fused uncertainty is K/S = 3/(3+1.5+1.5) = 0.5 => between 0.1 and 0.9
        # Not conflict (single modality), not consensus (>=0.1), defaults to UNCERTAIN
        if result["fused_uncertainty"] >= fuser.threshold_consensus:
            assert result["phase"] == "UNCERTAIN"

    # --- None modalities ---

    def test_fuse_with_none_modalities(self):
        """None entries should be silently skipped."""
        fuser = MultimodalFusion(n_classes=2)
        result = fuser.fuse([np.array([5.0, 0.1]), None, np.array([6.0, 0.3])])
        assert result["phase"] == "CONSENSUS"
        assert len(result["modality_agreements"]) == 2

    def test_fuse_all_none(self):
        """All None => uniform probs and UNCERTAIN."""
        fuser = MultimodalFusion(n_classes=3)
        result = fuser.fuse([None, None, None])
        np.testing.assert_allclose(result["fused_probs"],
                                   [1 / 3, 1 / 3, 1 / 3])
        assert result["fused_uncertainty"] == 1.0
        assert result["phase"] == "UNCERTAIN"
        assert result["modality_agreements"] == []

    def test_fuse_empty_list(self):
        """Empty list => same as all None."""
        fuser = MultimodalFusion(n_classes=2)
        result = fuser.fuse([])
        assert result["fused_uncertainty"] == 1.0
        assert result["phase"] == "UNCERTAIN"

    # --- Evidence validation ---

    def test_fuse_wrong_evidence_shape(self):
        fuser = MultimodalFusion(n_classes=3)
        with pytest.raises(ValueError, match="Evidence at index 0"):
            fuser.fuse([np.array([1.0, 2.0])])

    def test_fuse_2d_evidence(self):
        fuser = MultimodalFusion(n_classes=2)
        with pytest.raises(ValueError, match="Evidence at index 0"):
            fuser.fuse([np.array([[1.0, 2.0]])])

    # --- Product of experts ---

    def test_product_of_experts_single(self):
        fuser = MultimodalFusion(n_classes=3)
        alpha = np.array([5.0, 2.0, 1.0])
        result = fuser._product_of_experts([alpha])
        # n_experts=1 so fused = summed - 0 * ones = alpha
        np.testing.assert_allclose(result, alpha)

    def test_product_of_experts_two(self):
        fuser = MultimodalFusion(n_classes=3)
        a1 = np.array([3.0, 1.0, 1.0])
        a2 = np.array([2.0, 1.0, 1.0])
        result = fuser._product_of_experts([a1, a2])
        expected = a1 + a2 - np.ones(3)
        expected = np.maximum(expected, 1.0)
        np.testing.assert_allclose(result, expected)

    def test_product_of_experts_clamp(self):
        """Fused alpha is clamped to >= 1.0."""
        fuser = MultimodalFusion(n_classes=3)
        a1 = np.array([1.0, 1.0, 1.0])
        a2 = np.array([1.0, 1.0, 1.0])
        result = fuser._product_of_experts([a1, a2])
        # sum - ones = [2,2,2] - [1,1,1] = [1,1,1]
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])

    # --- _classify_phase ---

    def test_classify_phase_consensus(self):
        fuser = MultimodalFusion(n_classes=3)
        assert fuser._classify_phase([0, 0, 0], 0.1) == "CONSENSUS"

    def test_classify_phase_conflict(self):
        fuser = MultimodalFusion(n_classes=3)
        assert fuser._classify_phase([0, 1], 0.1) == "CONFLICT"

    def test_classify_phase_uncertain(self):
        fuser = MultimodalFusion(n_classes=3)
        assert fuser._classify_phase([0, 0], 0.8) == "UNCERTAIN"

    # --- Negative evidence clamping ---

    def test_fuse_clamps_negative_evidence(self):
        fuser = MultimodalFusion(n_classes=2)
        result = fuser.fuse([np.array([-5.0, 10.0])])
        # -5 clamped to 0, alpha = [1.0, 11.0]
        expected_probs = np.array([1.0 / 12.0, 11.0 / 12.0])
        np.testing.assert_allclose(result["fused_probs"], expected_probs, atol=1e-8)


# ======================================================================
# OpenSetRecognition
# ======================================================================


class TestOpenSetRecognition:
    """Tests for OpenSetRecognition."""

    # --- Construction ---

    def test_init(self):
        osr = OpenSetRecognition(n_classes=3, threshold=0.6)
        assert osr.n_classes == 3
        assert osr.threshold == 0.6

    def test_init_invalid(self):
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            OpenSetRecognition(n_classes=0)

    # --- predict_open ---

    def test_predict_open_known(self):
        osr = OpenSetRecognition(n_classes=3, threshold=0.5)
        pred = osr.predict_open(np.array([10.0, 1.0, 0.5]))
        assert pred["rejected"] is False
        assert pred["label"] != "UNKNOWN"
        assert pred["predicted_class"] == 0

    def test_predict_open_rejected(self):
        osr = OpenSetRecognition(n_classes=3, threshold=0.5)
        pred = osr.predict_open(np.array([0.0, 0.0, 0.0]))
        assert pred["rejected"] is True
        assert pred["label"] == "UNKNOWN"
        assert pred["predicted_class"] == -1

    def test_predict_open_keys(self):
        osr = OpenSetRecognition(n_classes=2, threshold=0.5)
        pred = osr.predict_open(np.array([5.0, 0.0]))
        expected_keys = {"predicted_class", "label", "probs", "uncertainty",
                         "epistemic", "aleatoric", "rejected"}
        assert set(pred.keys()) == expected_keys

    def test_predict_open_label_is_class_string(self):
        osr = OpenSetRecognition(n_classes=3, threshold=0.3)
        pred = osr.predict_open(np.array([50.0, 0.0, 0.0]))
        assert pred["label"] == "0"

    # --- Open-set rejection rate ---

    def test_open_set_rejection_rate_unknown_data(self):
        """Unknown data (zero evidence) should be mostly rejected."""
        osr = OpenSetRecognition(n_classes=3, threshold=0.5)
        n_unknown = 100
        rejected = 0
        for _ in range(n_unknown):
            pred = osr.predict_open(np.array([0.0, 0.0, 0.0]))
            if pred["rejected"]:
                rejected += 1
        assert rejected / n_unknown >= 0.9  # at least 90% rejection

    def test_open_set_rejection_rate_known_data(self):
        """Known data (high evidence) should rarely be rejected."""
        osr = OpenSetRecognition(n_classes=3, threshold=0.5)
        n_known = 100
        rejected = 0
        for _ in range(n_known):
            pred = osr.predict_open(np.array([20.0, 1.0, 0.5]))
            if pred["rejected"]:
                rejected += 1
        assert rejected / n_known < 0.1  # less than 10% rejection

    # --- calibrate ---

    def test_calibrate_updates_threshold(self):
        osr = OpenSetRecognition(n_classes=2)
        evidences = [np.array([10.0, 0.5])] * 20 + [np.array([0.1, 0.1])] * 5
        labels = [0] * 20 + [1] * 5
        old_threshold = osr.threshold
        new_threshold = osr.calibrate(evidences, labels, fpr_target=0.05)
        assert new_threshold > 0
        assert osr.threshold == new_threshold

    def test_calibrate_mismatched_lengths(self):
        osr = OpenSetRecognition(n_classes=2)
        with pytest.raises(ValueError, match="same length"):
            osr.calibrate([np.array([1.0, 0.0])], [0, 1])

    def test_calibrate_empty(self):
        osr = OpenSetRecognition(n_classes=2)
        with pytest.raises(ValueError, match="empty"):
            osr.calibrate([], [])

    def test_calibrate_fpr_respected(self):
        """After calibration, the fraction of known samples rejected ≈ fpr_target."""
        osr = OpenSetRecognition(n_classes=3)
        rng = np.random.RandomState(123)
        n_samples = 200
        evidences = []
        labels = []
        for _ in range(n_samples):
            label = rng.randint(0, 3)
            ev = np.zeros(3)
            ev[label] = rng.uniform(5, 30)
            evidences.append(ev)
            labels.append(label)

        fpr_target = 0.1
        osr.calibrate(evidences, labels, fpr_target=fpr_target)

        # Count rejections on the same validation data
        rejected = 0
        for ev in evidences:
            pred = osr.predict_open(ev)
            if pred["rejected"]:
                rejected += 1
        actual_fpr = rejected / n_samples
        # Allow some tolerance
        assert actual_fpr <= fpr_target + 0.05

    def test_calibrate_single_sample(self):
        osr = OpenSetRecognition(n_classes=2)
        thr = osr.calibrate([np.array([5.0, 0.5])], [0], fpr_target=0.05)
        assert thr > 0

    # --- get_rejection_stats ---

    def test_get_rejection_stats_empty(self):
        osr = OpenSetRecognition(n_classes=2, threshold=0.5)
        stats = osr.get_rejection_stats()
        assert stats["total_seen"] == 0
        assert stats["total_rejected"] == 0
        assert stats["rejection_rate"] == 0.0
        assert stats["per_class_rates"] == {}

    def test_get_rejection_stats_after_predictions(self):
        osr = OpenSetRecognition(n_classes=2, threshold=0.9)
        # Known sample — should not be rejected
        osr.predict_open(np.array([10.0, 0.5]))
        # Unknown sample — should be rejected (zero evidence, vacuity=1.0 >= 0.9)
        osr.predict_open(np.array([0.0, 0.0]))

        stats = osr.get_rejection_stats()
        assert stats["total_seen"] == 2
        assert stats["total_rejected"] == 1
        assert stats["rejection_rate"] == pytest.approx(0.5)

    def test_get_rejection_stats_all_rejected(self):
        osr = OpenSetRecognition(n_classes=3, threshold=0.1)
        # Very low threshold, almost everything gets rejected unless very strong evidence
        osr.predict_open(np.array([0.0, 0.0, 0.0]))
        osr.predict_open(np.array([0.0, 0.0, 0.0]))
        stats = osr.get_rejection_stats()
        assert stats["total_rejected"] == 2
        assert stats["rejection_rate"] == pytest.approx(1.0)

    def test_get_rejection_stats_per_class(self):
        osr = OpenSetRecognition(n_classes=2, threshold=0.9)
        # Two known predictions for class 0
        osr.predict_open(np.array([20.0, 0.0]))
        osr.predict_open(np.array([20.0, 0.0]))
        stats = osr.get_rejection_stats()
        assert 0 in stats["per_class_rates"]
        assert stats["per_class_rates"][0] == pytest.approx(0.0)


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    """Additional edge-case tests across all evidential modules."""

    def test_classifier_two_classes(self):
        clf = EvidentialClassifier(n_classes=2)
        pred = clf.predict(np.array([1.0, 0.0]))
        assert pred["predicted_class"] == 0

    def test_classifier_large_n_classes(self):
        clf = EvidentialClassifier(n_classes=50)
        ev = np.zeros(50)
        ev[25] = 100.0
        pred = clf.predict(ev)
        assert pred["predicted_class"] == 25

    def test_regressor_negative_gamma(self):
        reg = EvidentialRegressor()
        pred = reg.predict(np.array([-10.0, 0.0, 1.0, 0.5]))
        assert pred["mean"] == -10.0

    def test_regressor_very_large_outputs(self):
        reg = EvidentialRegressor()
        # softplus(100) ≈ 100
        fwd = reg.forward(np.array([0.0, 100.0, 100.0, 100.0]))
        assert fwd["nu"] > 0
        assert fwd["alpha"] > 1.0
        assert fwd["beta"] > 0

    def test_regressor_very_negative_log_params(self):
        """Very negative log params => softplus ≈ 0 => clamped to 1e-6."""
        reg = EvidentialRegressor()
        g, nu, alpha, beta = reg._parse_outputs(np.array([0.0, -100.0, -100.0, -100.0]))
        assert nu == pytest.approx(1e-6, abs=1e-7)
        assert beta == pytest.approx(1e-6, abs=1e-7)
        # softplus(-100) ≈ 0, so alpha ≈ 1.0 (clamped softplus + 1)
        assert alpha >= 1.0
        assert alpha < 1.0 + 1e-4

    def test_classifier_all_equal_evidence(self):
        clf = EvidentialClassifier(n_classes=3)
        pred = clf.predict(np.array([5.0, 5.0, 5.0]))
        # All probs equal
        np.testing.assert_allclose(pred["probs"], [1 / 3, 1 / 3, 1 / 3])
        # Predicted class is 0 (argmax returns first index for ties)
        assert pred["predicted_class"] == 0

    def test_fusion_single_modality(self):
        fuser = MultimodalFusion(n_classes=3)
        result = fuser.fuse([np.array([50.0, 1.0, 0.5])])
        assert result["phase"] == "CONSENSUS"
        assert len(result["modality_agreements"]) == 1

    def test_fusion_many_modalities_consensus(self):
        fuser = MultimodalFusion(n_classes=2)
        evidence_list = [np.array([10.0, 0.5])] * 10
        result = fuser.fuse(evidence_list)
        assert result["phase"] == "CONSENSUS"
        assert len(result["modality_agreements"]) == 10

    def test_fusion_three_modalities_two_conflict(self):
        """Two agreeing modalities + one disagreeing => CONFLICT."""
        fuser = MultimodalFusion(n_classes=3)
        result = fuser.fuse([
            np.array([10.0, 0.0, 0.0]),
            np.array([10.0, 0.0, 0.0]),
            np.array([0.0, 10.0, 0.0]),
        ])
        assert result["phase"] == "CONFLICT"

    def test_openset_accumulates_stats(self):
        """Stats should accumulate across multiple calls."""
        osr = OpenSetRecognition(n_classes=2, threshold=0.5)
        for _ in range(10):
            osr.predict_open(np.array([10.0, 0.0]))
        for _ in range(5):
            osr.predict_open(np.array([0.0, 0.0]))
        stats = osr.get_rejection_stats()
        assert stats["total_seen"] == 15

    def test_classifier_integer_evidence(self):
        """Integer evidence should be promoted to float."""
        clf = EvidentialClassifier(n_classes=3)
        result = clf.forward(np.array([5, 2, 1]))
        assert result["alpha"].dtype == np.float64

    def test_fusion_negative_evidence_clamped(self):
        """Negative evidence values in fusion are clamped to 0."""
        fuser = MultimodalFusion(n_classes=2)
        result = fuser.fuse([np.array([-10.0, -10.0])])
        # alpha = [1, 1], probs = [0.5, 0.5], uncertainty = K/S = 2/2 = 1.0
        assert result["fused_uncertainty"] == pytest.approx(1.0)

    def test_classifier_kl_weight_zero(self):
        """With kl_weight=0, total_loss should equal nll_loss."""
        clf = EvidentialClassifier(n_classes=3, kl_weight=0.0, kl_annealing=False)
        loss = clf.compute_loss(np.array([5.0, 1.0, 0.5]), target=0, step=50)
        assert loss["total_loss"] == pytest.approx(loss["nll_loss"])

    def test_regressor_kl_weight_zero_means_no_reg(self):
        """With kl_weight=0, reg_loss should be 0."""
        reg = EvidentialRegressor(kl_weight=0.0)
        loss = reg.compute_loss(np.array([5.0, 1.0, 2.0, 0.5]), target=10.0)
        assert loss["reg_loss"] == pytest.approx(0.0)
