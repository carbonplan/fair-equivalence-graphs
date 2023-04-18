# isort: skip_file
import math

import numpy as np
import pytest
from fair.SSPs import ssp119, ssp245, ssp370  # noqa

from fair_equivalence_graphs import get_equivalence_ratio, get_perturbations, run_fair


@pytest.mark.parametrize("scenario", [ssp119, ssp245, ssp370])
@pytest.mark.parametrize("year", [2000, 2022, 2050])
@pytest.mark.parametrize("storage_length", [1, 10, 100])
@pytest.mark.parametrize("storage_magnitude", [0.1, 1, 10])
@pytest.mark.parametrize("equivalence_ratio", [0.5, 1, 2])
def test_get_perturbations(
    scenario, year, storage_length, storage_magnitude, equivalence_ratio
) -> None:
    storage, reemission, justified = get_perturbations(
        scenario, year, storage_length, storage_magnitude, equivalence_ratio
    )
    assert isinstance(storage, np.ndarray)
    assert isinstance(reemission, np.ndarray)
    assert isinstance(justified, np.ndarray)
    assert len(scenario.Emissions.co2) == len(storage)
    assert len(scenario.Emissions.co2) == len(reemission)
    assert len(scenario.Emissions.co2) == len(justified)


@pytest.mark.parametrize("storage_magnitude", [0.1, 1, 10])
@pytest.mark.parametrize("equivalence_ratio", [0.5, 1, 2])
def test_perturbations_values(storage_magnitude, equivalence_ratio) -> None:
    scenario = ssp245
    year = 2022
    length = 10
    storage, reemission, justified = get_perturbations(
        scenario, year, length, storage_magnitude, equivalence_ratio
    )
    assert sum(storage) == (storage_magnitude * -1)
    assert sum(reemission) == storage_magnitude
    assert sum(justified) == storage_magnitude / equivalence_ratio


def test_run_fair() -> None:
    scenario = ssp245
    year = 2022
    length = 10
    storage_magnitude = 0.1
    equivalence_ratio = 1
    storage, reemission, justified = get_perturbations(
        scenario, year, length, storage_magnitude, equivalence_ratio
    )
    experimental_scenarios, fair_results = run_fair(scenario, storage, reemission, justified)
    assert isinstance(experimental_scenarios, dict)
    assert isinstance(fair_results, dict)


@pytest.mark.parametrize("scenario", [ssp119, ssp245, ssp370])
@pytest.mark.parametrize("storage_magnitude", [0.1, 1])
def test_run_fair_rf_values(scenario, storage_magnitude) -> None:
    year = 2022
    length = 10
    equivalence_ratio = 1
    storage, reemission, justified = get_perturbations(
        scenario, year, length, storage_magnitude, equivalence_ratio
    )
    experimental_scenarios, fair_results = run_fair(scenario, storage, reemission, justified)
    assert sum(fair_results["justified"]["rf"] - fair_results["default"]["rf"]) > 0
    assert sum(fair_results["default"]["rf"] - fair_results["permanent"]["rf"]) > 0
    assert sum(fair_results["default"]["rf"] - fair_results["temporary"]["rf"]) > 0
    assert sum(fair_results["temporary"]["rf"] - fair_results["permanent"]["rf"]) > 0


def test_get_equivalence_ratio() -> None:
    scenario = ssp245
    year = 2022
    length = 10
    storage_magnitude = 0.1
    time_horizon = 100
    equivalence_ratio = get_equivalence_ratio(
        scenario, year, length, storage_magnitude, time_horizon
    )
    assert isinstance(equivalence_ratio, float)


@pytest.mark.parametrize("time_horizon", [100, 400])
def test_get_equivalence_ratio_values(time_horizon) -> None:
    scenario = ssp245
    year = 2022
    length = 10
    storage_magnitude = 0.1
    equivalence_ratio = get_equivalence_ratio(
        scenario, year, length, storage_magnitude, time_horizon
    )
    assert isinstance(equivalence_ratio, float)

    storage, reemission, justified = get_perturbations(
        scenario, year, length, storage_magnitude, equivalence_ratio
    )
    experimental_scenarios, fair_results = run_fair(scenario, storage, reemission, justified)
    yr_idx = np.where(scenario.Emissions.year == year)[0][0]
    assert math.isclose(
        sum(fair_results["default"]["rf"][yr_idx : yr_idx + time_horizon]),
        sum(fair_results["offset"]["rf"][yr_idx : yr_idx + time_horizon]),
        abs_tol=10**-12,
    )
