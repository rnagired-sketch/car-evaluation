from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
import pytest

from knn_eval_core.dataset import (
    COLUMN_NAMES,
    CarDataset,
    DataSplits,
    Partition,
    SplitFractions,
    ValidationReport,
)
from knn_eval_core.domain import Acceptability, Seed
from knn_eval_core.features import FEATURE_NAMES

MakeRaw = Callable[[Sequence[Sequence[str | None]]], pd.DataFrame]

GOOD_ROW = ["low", "low", "4", "4", "big", "high", "vgood"]
OTHER_ROW = ["vhigh", "vhigh", "2", "2", "small", "low", "unacc"]


class TestFromFrame:
    def test_real_dataset_is_clean(self, raw_frame: pd.DataFrame) -> None:
        dataset, report = CarDataset.from_frame(raw_frame)
        assert report == ValidationReport(1728, 0, 0, 0, 0, 1728)
        assert len(dataset) == 1728
        assert dataset.data.class_counts() == {
            Acceptability.UNACC: 1210,
            Acceptability.ACC: 384,
            Acceptability.GOOD: 69,
            Acceptability.VGOOD: 65,
        }

    def test_encodes_ranks_and_class_codes(self, make_raw: MakeRaw) -> None:
        dataset, _ = CarDataset.from_frame(make_raw([GOOD_ROW, OTHER_ROW]))
        assert dataset.data.X.iloc[0].tolist() == [0, 0, 2, 1, 2, 2]
        assert dataset.data.X.iloc[1].tolist() == [3, 3, 0, 0, 0, 0]
        assert dataset.data.y.tolist() == [Acceptability.VGOOD, Acceptability.UNACC]
        assert all(dtype.kind == "i" for dtype in dataset.data.X.dtypes)
        assert dataset.data.y.name == "class"

    def test_index_is_reset(self, make_raw: MakeRaw) -> None:
        dataset, _ = CarDataset.from_frame(make_raw([[None] * 7, GOOD_ROW, OTHER_ROW]))
        assert dataset.data.X.index.tolist() == [0, 1]

    def test_normalises_case_and_whitespace(self, make_raw: MakeRaw) -> None:
        dataset, report = CarDataset.from_frame(
            make_raw([[" LOW", "Low ", "4", "4", "BIG", "high", " VGood "]])
        )
        assert report.rows_kept == 1
        assert dataset.data.y.iloc[0] == Acceptability.VGOOD

    def test_drops_rows_with_missing_values(self, make_raw: MakeRaw) -> None:
        _, report = CarDataset.from_frame(make_raw([GOOD_ROW, [*GOOD_ROW[:3], None, *GOOD_ROW[4:]]]))
        assert report.rows_with_missing_values == 1
        assert report.rows_kept == 1

    @pytest.mark.parametrize(("column", "value"), [(0, "cheap"), (2, "6"), (6, "great")])
    def test_drops_rows_with_unknown_levels_or_labels(
        self, make_raw: MakeRaw, column: int, value: str
    ) -> None:
        bad = list(OTHER_ROW)
        bad[column] = value
        _, report = CarDataset.from_frame(make_raw([GOOD_ROW, bad]))
        assert report.rows_with_unknown_levels == 1
        assert report.rows_kept == 1

    def test_missing_is_not_double_counted_as_unknown(self, make_raw: MakeRaw) -> None:
        _, report = CarDataset.from_frame(make_raw([GOOD_ROW, [None, "cheap", *OTHER_ROW[2:]]]))
        assert report.rows_with_missing_values == 1
        assert report.rows_with_unknown_levels == 0

    def test_collapses_exact_duplicates(self, make_raw: MakeRaw) -> None:
        _, report = CarDataset.from_frame(make_raw([GOOD_ROW, GOOD_ROW, GOOD_ROW, OTHER_ROW]))
        assert report.exact_duplicates_removed == 2
        assert report.rows_kept == 2

    def test_drops_every_copy_of_a_contradicted_car(self, make_raw: MakeRaw) -> None:
        contradiction = [*GOOD_ROW[:6], "acc"]
        dataset, report = CarDataset.from_frame(make_raw([GOOD_ROW, contradiction, OTHER_ROW]))
        assert report.conflicting_rows_removed == 2
        assert dataset.data.y.tolist() == [Acceptability.UNACC]

    def test_rejects_wrong_columns(self, make_raw: MakeRaw) -> None:
        frame = make_raw([GOOD_ROW]).rename(columns={"class": "label"})
        with pytest.raises(ValueError, match="expected columns"):
            CarDataset.from_frame(frame)

    def test_rejects_data_with_nothing_valid(self, make_raw: MakeRaw) -> None:
        with pytest.raises(ValueError, match="no valid rows"):
            CarDataset.from_frame(make_raw([[None] * 7]))

    def test_report_rows_add_up(self, make_raw: MakeRaw) -> None:
        rows = [GOOD_ROW, GOOD_ROW, [*GOOD_ROW[:6], "acc"], OTHER_ROW, [None] * 7, ["x", *OTHER_ROW[1:]]]
        _, r = CarDataset.from_frame(make_raw(rows))
        removed = (
            r.rows_with_missing_values
            + r.rows_with_unknown_levels
            + r.exact_duplicates_removed
            + r.conflicting_rows_removed
        )
        assert r.rows_read == len(rows) == removed + r.rows_kept


class TestPartition:
    def test_rejects_length_mismatch(self) -> None:
        X = pd.DataFrame([[0] * 6], columns=list(FEATURE_NAMES))
        with pytest.raises(ValueError, match="rows"):
            Partition(X, pd.Series([0, 1]))

    def test_rejects_wrong_columns(self) -> None:
        with pytest.raises(ValueError, match="expected columns"):
            Partition(pd.DataFrame({"a": [0]}), pd.Series([0]))

    def test_concat(self, splits: DataSplits) -> None:
        both = Partition.concat(splits.train, splits.validation)
        assert len(both) == len(splits.train) + len(splits.validation)


class TestSplitFractions:
    def test_default_matches_proposal(self) -> None:
        assert SplitFractions() == SplitFractions(0.5, 0.2, 0.3)

    @pytest.mark.parametrize("parts", [(0.5, 0.2, 0.2), (0.0, 0.5, 0.5), (0.6, -0.1, 0.5), (1.0, 0.1, 0.1)])
    def test_rejects_invalid(self, parts: tuple[float, float, float]) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            SplitFractions(*parts)

    def test_tolerates_float_rounding(self) -> None:
        SplitFractions(0.7, 0.1, 0.2)


class TestDataSplits:
    def test_sizes_follow_fractions(self, dataset: CarDataset, splits: DataSplits) -> None:
        n = len(dataset)
        assert abs(len(splits.test) - 0.3 * n) <= 1
        assert abs(len(splits.validation) - 0.2 * n) <= 1
        assert len(splits.train) + len(splits.validation) + len(splits.test) == n

    def test_splits_are_disjoint_and_cover_everything(self, dataset: CarDataset, splits: DataSplits) -> None:
        parts = [set(p.X.index) for p in (splits.train, splits.validation, splits.test)]
        assert not (parts[0] & parts[1])
        assert not (parts[0] & parts[2])
        assert not (parts[1] & parts[2])
        assert set.union(*parts) == set(dataset.data.X.index)

    def test_labels_stay_aligned_with_features(self, dataset: CarDataset, splits: DataSplits) -> None:
        for part in (splits.train, splits.validation, splits.test):
            assert part.y.index.equals(part.X.index)
            pd.testing.assert_series_equal(part.y, dataset.data.y.loc[part.y.index])

    def test_every_split_keeps_the_class_mix(self, dataset: CarDataset, splits: DataSplits) -> None:
        overall = dataset.data.class_counts()
        for name, fraction in (("train", 0.5), ("validation", 0.2), ("test", 0.3)):
            counts = splits.class_counts()[name]
            for label, total in overall.items():
                assert abs(counts[label] - fraction * total) <= 1.5, (name, label)

    def test_is_reproducible_for_a_seed(self, dataset: CarDataset, splits: DataSplits) -> None:
        again = DataSplits.stratified(dataset, SplitFractions(), Seed(42))
        assert again.test.X.index.equals(splits.test.X.index)

    def test_differs_across_seeds(self, dataset: CarDataset, splits: DataSplits) -> None:
        other = DataSplits.stratified(dataset, SplitFractions(), Seed(7))
        assert not other.test.X.index.equals(splits.test.X.index)

    def test_development_is_train_plus_validation(self, splits: DataSplits) -> None:
        development = splits.development
        assert set(development.X.index) == set(splits.train.X.index) | set(splits.validation.X.index)

    def test_class_counts_keys(self, splits: DataSplits) -> None:
        assert list(splits.class_counts()) == ["train", "validation", "test"]


def test_column_names() -> None:
    assert (*FEATURE_NAMES, "class") == COLUMN_NAMES


@pytest.mark.parametrize(
    ("seed", "train_sum", "validation_sum", "test_sum", "first_test_rows"),
    [
        (42, 730230, 300808, 461090, [988, 703, 1283, 20, 996]),
        (43, 752019, 300789, 439320, [8, 1166, 882, 663, 897]),
    ],
)
def test_splits_match_the_decision_tree_application(
    dataset: CarDataset,
    seed: int,
    train_sum: int,
    validation_sum: int,
    test_sum: int,
    first_test_rows: list[int],
) -> None:
    """Fingerprints of the decision tree's splits (row numbers of car.data), recorded from
    car_eval_core. Matching them means both models are trained and tested on the same cars."""
    splits = DataSplits.stratified(dataset, SplitFractions(), Seed(seed))
    assert sum(splits.train.X.index) == train_sum
    assert sum(splits.validation.X.index) == validation_sum
    assert sum(splits.test.X.index) == test_sum
    assert list(splits.test.X.index[:5]) == first_test_rows
