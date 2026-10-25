from __future__ import annotations

import dataclasses

import pytest

from car_eval_core.domain import CLASS_CODES, CLASS_LABELS, Acceptability, Car
from car_eval_core.features import FEATURE_NAMES


class TestAcceptability:
    def test_ordered_worst_to_best(self) -> None:
        assert list(Acceptability) == sorted(Acceptability)
        assert CLASS_LABELS == ("unacc", "acc", "good", "vgood")
        assert CLASS_CODES == (0, 1, 2, 3)

    @pytest.mark.parametrize("label", ["vgood", "VGOOD", " vgood "])
    def test_from_label_is_forgiving_about_case_and_whitespace(self, label: str) -> None:
        assert Acceptability.from_label(label) is Acceptability.VGOOD

    def test_from_label_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown class label"):
            Acceptability.from_label("great")

    def test_only_unacc_is_unacceptable(self) -> None:
        assert [a.is_acceptable for a in Acceptability] == [False, True, True, True]

    def test_label_round_trip(self) -> None:
        for a in Acceptability:
            assert Acceptability.from_label(a.label) is a


class TestCar:
    def test_valid_car(self) -> None:
        car = Car("low", "med", "5more", "more", "big", "high")
        assert car.doors == "5more"

    @pytest.mark.parametrize(
        ("field", "value"), [("buying", "cheap"), ("doors", "5"), ("persons", "3"), ("safety", "HIGH")]
    )
    def test_rejects_unknown_levels(self, field: str, value: str) -> None:
        values = {
            "buying": "low",
            "maint": "low",
            "doors": "4",
            "persons": "4",
            "lug_boot": "big",
            "safety": "high",
        }
        values[field] = value
        with pytest.raises(ValueError, match=field):
            Car(**values)

    def test_is_immutable(self) -> None:
        car = Car("low", "low", "4", "4", "big", "high")
        with pytest.raises(dataclasses.FrozenInstanceError):
            car.buying = "vhigh"  # type: ignore[misc]

    def test_encoded_gives_ranks_in_column_order(self) -> None:
        encoded = Car("vhigh", "low", "2", "more", "med", "high").encoded()
        assert tuple(encoded.columns) == FEATURE_NAMES
        assert encoded.iloc[0].tolist() == [3, 0, 0, 2, 1, 2]

    def test_from_ranks_inverts_encoded_for_every_car(self, all_cars: list[Car]) -> None:
        assert len(set(all_cars)) == 1728
        for car in all_cars:
            assert Car.from_ranks(car.encoded().iloc[0]) == car
