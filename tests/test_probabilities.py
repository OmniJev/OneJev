import math

import pytest

from qev.answers import choice_confidence, expected_score, make_choice, make_noul, make_score, score_confidence
from qev.calibrate import Calibration, adaptive_ece, brier, ece, fit_temperature, nll, softmax, tvd


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_softmax_rejects_invalid_temperatures(temperature):
    with pytest.raises(ValueError, match="temperature"):
        softmax([1, 2], temperature)


def test_softmax_large_offsets_and_negative_infinity():
    assert softmax([10000, 10000]) == [0.5, 0.5]
    assert softmax([float("-inf"), 1]) == [0, 1]


@pytest.mark.parametrize("logits", [[], [float("nan"), 1], [float("inf"), 1], [float("-inf")]])
def test_softmax_rejects_unusable_logits(logits):
    with pytest.raises(ValueError):
        softmax(logits)


@pytest.mark.parametrize("value", [0, -1, "bad", float("inf"), float("nan")])
def test_calibration_rejects_invalid_temperature_table(value):
    with pytest.raises(ValueError, match="temperature"):
        Calibration({"choice:3": value})


def test_calibration_round_trip_and_fallback(tmp_path):
    calibration = Calibration({"choice": 2, "choice:3": 4})
    assert calibration.temperature("choice", 3) == 4
    assert calibration.temperature("choice", 2) == 2
    assert calibration.temperature("noul", 2) == 1
    path = tmp_path / "calibration.json"
    calibration.save(path)
    assert Calibration.load(path).temperatures == calibration.temperatures


def test_fitted_temperature_recovers_known_soft_targets():
    logits = [[2, 0], [1, -1, 0], [-2, 1]]
    targets = [softmax(row, 2.5) for row in logits]
    assert fit_temperature(logits, targets) == pytest.approx(2.5, rel=1e-5)


@pytest.mark.parametrize("logits,targets", [([], []), ([[1, 0]], []), ([[1, 0]], [[1]]), ([[1, 0]], [[-1, 2]])])
def test_temperature_fit_rejects_invalid_training_pairs(logits, targets):
    with pytest.raises(ValueError):
        fit_temperature(logits, targets)


@pytest.mark.parametrize("fn", [choice_confidence, score_confidence, expected_score])
@pytest.mark.parametrize("probs", [[], [-1, 2], [float("nan"), 1], [float("inf"), 1]])
def test_answer_helpers_reject_invalid_probabilities(fn, probs):
    with pytest.raises(ValueError):
        fn(probs)


def test_uniform_and_certain_confidence():
    assert choice_confidence([0, 0, 0]) == 0
    assert choice_confidence([0, 1, 0]) == 1
    assert score_confidence([0, 0, 1]) == 1
    assert expected_score([0, 0, 1]) == 2
    assert choice_confidence([1]) == score_confidence([1]) == 1
    assert 0 <= choice_confidence([0.1] * 7) <= 1


def test_probabilities_with_large_finite_values_are_normalized():
    answer = make_choice(["a", "b"], [1e308, 1e308])
    assert answer.probabilities == {"a": 0.5, "b": 0.5}


@pytest.mark.parametrize("fn,labels", [(make_choice, ["a", "b"]), (make_score, ["bad", "good"])])
def test_answer_shape_mismatch(fn, labels):
    with pytest.raises(ValueError, match="same length"):
        fn(labels, [1])


def test_noul_rejects_nan_and_keeps_clamping_finite_values():
    with pytest.raises(ValueError):
        make_noul(float("nan"))
    assert make_noul(-0.1).noul == 0
    assert make_noul(1.1).noul == 1


def test_metric_reference_values():
    assert brier([0.8, 0.2], [1, 0]) == pytest.approx(0.08)
    assert nll([0.8, 0.2], [1, 0]) == pytest.approx(-math.log(0.8))
    assert tvd([0.8, 0.2], [0.2, 0.8]) == pytest.approx(0.6)
    assert ece([0, 0.5, 1], [0, 1, 1], n_bins=2) == pytest.approx(1 / 6)


@pytest.mark.parametrize("fn", [brier, nll, tvd, ece, adaptive_ece])
def test_metrics_reject_mismatched_lengths(fn):
    with pytest.raises(ValueError, match="same length"):
        fn([0.5, 0.5], [1])


@pytest.mark.parametrize("fn", [ece, adaptive_ece])
def test_ece_rejects_invalid_bins(fn):
    with pytest.raises(ValueError, match="n_bins"):
        fn([0.5], [1], n_bins=0)


def test_adaptive_ece_uses_requested_number_of_balanced_bins():
    # Five rows in two bins: [0, .2] and [.4, .6, .8].
    assert adaptive_ece([0, 0.2, 0.4, 0.6, 0.8], [0, 0, 0, 0, 1], n_bins=2) == pytest.approx(0.2)
    assert adaptive_ece([0.1, 0.9], [0, 1], n_bins=10) == pytest.approx(0.1)
