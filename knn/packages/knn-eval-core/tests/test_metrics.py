from __future__ import annotations

import math

import numpy as np
import pytest

from knn_eval_core.domain import Seed
from knn_eval_core.metrics import CvSettings, Scores

U, A, G, V = 0, 1, 2, 3


class TestScores:
    def test_perfect_predictions(self) -> None:
        y = np.array([U, U, A, G, V])
        assert Scores.of(y, y) == Scores(1.0, 1.0, 1.0, 0.0, 0.0, 0)

    def test_false_acceptance_counts_unacc_rated_acceptable(self) -> None:
        truth = np.array([U, U, U, U, A])
        pred = np.array([U, A, V, U, A])
        assert Scores.of(truth, pred).false_acceptance == pytest.approx(2 / 4)

    def test_false_rejection_counts_acceptable_rated_unacc(self) -> None:
        truth = np.array([A, G, V, V, U])
        pred = np.array([U, G, U, V, U])
        assert Scores.of(truth, pred).false_rejection == pytest.approx(2 / 4)

    def test_confusion_among_acceptable_classes_is_not_a_boundary_error(self) -> None:
        scores = Scores.of(np.array([A, G, V, U]), np.array([G, V, A, U]))
        assert scores.false_acceptance == 0.0
        assert scores.false_rejection == 0.0

    def test_severe_errors_are_two_or_more_classes_off(self) -> None:
        truth = np.array([U, U, U, A, V])
        pred = np.array([A, G, V, V, G])
        assert Scores.of(truth, pred).severe_errors == 3

    def test_rates_are_nan_when_a_side_is_absent(self) -> None:
        only_acceptable = Scores.of(np.array([A, G]), np.array([A, G]))
        assert math.isnan(only_acceptable.false_acceptance)
        only_unacc = Scores.of(np.array([U, U]), np.array([U, U]))
        assert math.isnan(only_unacc.false_rejection)

    def test_macro_f1_counts_absent_classes_as_zero(self) -> None:
        # Classes that never occur still enter the macro average, so ignoring rare
        # classes is punished rather than hidden.
        scores = Scores.of(np.array([U, U, A]), np.array([U, U, A]))
        assert scores.macro_f1 == pytest.approx(2 / 4)

    def test_majority_vote_has_high_accuracy_but_low_macro_f1(self) -> None:
        truth = np.array([U] * 70 + [A] * 20 + [G] * 5 + [V] * 5)
        scores = Scores.of(truth, np.full(100, U))
        assert scores.accuracy == pytest.approx(0.70)
        assert scores.balanced_accuracy == pytest.approx(0.25)
        assert scores.macro_f1 < 0.25

    def test_accepts_pandas_and_lists(self) -> None:
        import pandas as pd

        assert Scores.of(pd.Series([U, A]), np.array([U, A])).accuracy == 1.0

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            Scores.of(np.array([U, A]), np.array([U]))

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Scores.of(np.array([], dtype=int), np.array([], dtype=int))


class TestCvSettings:
    def test_splitter_uses_settings(self) -> None:
        splitter = CvSettings(4, 2).splitter(Seed(0))
        assert splitter.get_n_splits() == 8

    @pytest.mark.parametrize(("n_splits", "n_repeats"), [(1, 1), (5, 0)])
    def test_rejects_invalid(self, n_splits: int, n_repeats: int) -> None:
        with pytest.raises(ValueError, match="must be at least"):
            CvSettings(n_splits, n_repeats)
