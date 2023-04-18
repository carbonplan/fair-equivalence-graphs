import copy
from collections import defaultdict

import fair
import numpy as np
from scipy import optimize


def get_perturbations(
    scenario: np.ndarray,
    temp_yr: int,
    temp_length: int,
    temp_magnitude: float,
    equivalence_ratio: float,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Build the purturbation arrays

    Parameters
    ----------
    scenario : np.ndarray
        FaIR provided SSP emissions scenario that perturbations will be applied to
    temp_yr : int
        Year at which temporary carbon storage scenario begins (e.g. 2022)
    temp_length : int
        Duration of temporary carbon storage (e.g. 10 years)
    temp_magnitude : float
        Magnitude of temporary storage in GtC. The conversion factor from C to CO2 is 3.67.
        For orientation, annual CO2 emissions are ~36 GtCO2, so a little under 10 GtC.
    equivalence_ratio : float
        Describes how many tons of CO₂ need to be temporarily stored to justify an additional
        ton of CO₂ emitted to the atmosphere.

    Returns
    -------
    storage : np.ndarray
        1D array representing the storage of {temp_magnitude} carbon at {temp_yr}
    reemission : np.ndarray
        1D array representing the re-emission of {temp_magnitude} after {temp_length}
        years of storage
    justified : np.ndarray
        1D array representing additional emissions justified by the temporary storage
        according to the provided {equivalence_ratio}
    """

    yr_idx = np.where(scenario.Emissions.year == temp_yr)[0][0]

    storage = np.zeros_like(scenario.Emissions.co2)
    reemission = np.zeros_like(scenario.Emissions.co2)
    justified = np.zeros_like(scenario.Emissions.co2)

    storage[yr_idx] += -1 * temp_magnitude
    reemission[yr_idx + temp_length] += temp_magnitude
    justified[yr_idx] += temp_magnitude * (1 / equivalence_ratio)

    return storage, reemission, justified


def run_fair(
    scenario: np.ndarray,
    storage: np.ndarray,
    reemission: np.ndarray,
    justified: np.ndarray,
) -> (dict, dict):
    """Run the FaIR model

    Parameters
    ----------
    scenario : np.ndarray
        FaIR provided SSP emissions scenario that modeled scenarios are based upon
    storage : np.ndarray
        1D array representing the magnitude and timing of carbon storage
    reemission : np.ndarray
        1D array representing the magnitude and timing of carbon re-emission
    justified : np.ndarray
        1D array representing emissions justified by temporary carbon storage

    Returns
    -------
    emissions_scenarios : dict
        Dict representing different emission scenarios constructed from the input
        parameters. Dict has the following keys:

        - `default` : array representing the emission scenario without perturbation
        - `permanent` : array representing the emission scenario + storage, with the
        assumption that stored carbon is never re-emitted
        - `temporary` : array representing the emission scenario + storage, with the
        assumption that stored carbon is fully re-emitted
        - `offset` : array representing the emission scenario + storage, with the
        assumption temporary carbon storage is used to justify additional emissions
        - `justified` : array representing the emission scenario + justified emissions

    fair_results : dict
        Dict containing FaIR results for each of the experimental scenarios. For each
        experimental scenario (keys the same as above) dict has the subsidiary keys:

        - `atm_c` : 1D array modeling CO2 concentrations (ppm) through time
        - `rf` : 1D array modeling total radiative forcing (W m-2) through time
        - `temp` : 1D array modeling temperature change (K) through time compared to
        preindustrial

    """

    fair_results = defaultdict(dict)
    default_emissions = scenario.Emissions.co2_land + scenario.Emissions.co2_fossil

    experimental_scenarios = {
        "default": default_emissions,
        "permanent": copy.copy(default_emissions) + storage,
        "temporary": copy.copy(default_emissions) + storage + reemission,
        "offset": copy.copy(default_emissions) + storage + reemission + justified,
        "justified": copy.copy(default_emissions) + justified,
    }

    for name, emissions in experimental_scenarios.items():
        C, F, T = fair.forward.fair_scm(
            emissions=emissions,
            other_rf=np.zeros_like(emissions),
            # ghg_forcing='Meinshausen',
            # rc=0.01,
            # rt=2,
            useMultigas=False,
            gir_carbon_cycle=False,
        )

        fair_results[name]["atm_c"] = C
        fair_results[name]["rf"] = F
        fair_results[name]["temp"] = T

    return experimental_scenarios, fair_results


def compare_rf(
    equivalence_ratio: float,
    scenario: np.ndarray,
    temp_yr: int,
    temp_length: int,
    temp_magnitude: float,
    time_horizon: int,
) -> float:
    """Helper function for finding an equivalence ratio that compares radiative forcing outcomes
    between a default scenario and an offset scenario.

    Parameters
    ----------
    equivalence_ratio : float
        Describes how many tons of CO₂ justify an additional ton of CO₂ emitted to the atmosphere.
    temp_yr : int
        Year at which temporary carbon storage scenario begins (e.g. 2022)
    temp_length : int
        Duration of temporary carbon storage (e.g. 10 years)
    temp_magnitude : float
        Magnitude of temporary storage in GtC. The conversion factor from C to CO2 is 3.67.
        For orientation, annual CO2 emissions are ~36 GtCO2, so a little under 10 GtC.
    time_horizon : time horizon over which cumulative radiative forcing is compared

    Returns
    -------
    diff : float
        Returns the difference in cumulative radiative forcing (default scenario - offset scenario)
    """
    storage, reemission, justified = get_perturbations(
        scenario, temp_yr, temp_length, temp_magnitude, equivalence_ratio
    )
    _, fair_results = run_fair(scenario, storage, reemission, justified)

    yr_idx = np.where(scenario.Emissions.year == temp_yr)[0][0]
    rf_d = sum(fair_results["default"]["rf"][yr_idx : yr_idx + time_horizon])
    rf_o = sum(fair_results["offset"]["rf"][yr_idx : yr_idx + time_horizon])
    diff = rf_d - rf_o
    return diff


def get_equivalence_ratio(
    scenario: np.ndarray,
    temp_yr: int,
    temp_length: int,
    temp_magnitude: float,
    time_horizon: int,
) -> float:
    """Use binary search to find an equivalence ratio on radiative forcing outcomes modeled by FaIR

    Parameters
    ----------
    equivalence_ratio : float
        Describes how many tons of CO₂ justify an additional ton of CO₂ emitted to the atmosphere.
    temp_yr : int
        Year at which temporary carbon storage scenario begins (e.g. 2022)
    temp_length : int
        Duration of temporary carbon storage (e.g. 10 years)
    temp_magnitude : float
        Magnitude of temporary storage in GtC. The conversion factor from C to CO2 is 3.67.
        For orientation, annual CO2 emissions are ~36 GtCO2, so a little under 10 GtC.
    time_horizon : time horizon over which cumulative radiative forcing is compared

    Returns
    -------
    Return float is the equivalence factor representing how much of the temporary storage
    described in the input parameters would be needed to justify an addition tCO2 emissions
    """
    return optimize.bisect(
        compare_rf,
        1,
        1000,
        args=(scenario, temp_yr, temp_length, temp_magnitude, time_horizon),
    )
