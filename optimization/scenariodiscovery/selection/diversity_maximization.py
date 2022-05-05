"""
This module contains a function for the diversity maximization approach.
Remark: Some used functions have been adopted from
https://github.com/shajeelwn/PyDICE/blob/master/4_Scenario_Discovery/Scenario_Selection_only_util_ds.py
"""

import numpy as np
import os
import itertools
import math

import pandas as pd
from scipy.spatial.distance import pdist
from concurrent.futures import ProcessPoolExecutor
from model.enumerations import *


def _n_combinations(n, r):
    """
    Return number of combinations given n and r (nCr style).
    @param n: int
    @param r: int
    @return:
        n: int
    """
    factorial = math.factorial
    n = factorial(n) / (factorial(r) * factorial(n-r))
    return int(n)


def _normalize(outcomes):
    """
    Normalizes outcomes dictionary.
    @param outcomes: dictionary
    @return:
        new_outcomes: dictionary
    """
    new_outcomes = {}
    for outcome_name in outcomes.keys():
        values = outcomes[outcome_name]
        max_value = max(values)
        min_value = min(values)
        range_value = max_value - min_value

        if max_value == min_value:
            new_outcomes[outcome_name] = values - min_value
        else:
            new_outcomes[outcome_name] = (values - min_value) / range_value

    return new_outcomes


def _calculate_distance(data, oois, scenarios=None, distance='euclidean'):
    """outcomes is the outcomes of exploration results,
    scenarios is a list of scenario indices (decision variables),
    outcome_names is a list of variable names,
    distance is to choose the distance metric. options:
            bray-curtis, canberra, chebyshev, cityblock (manhattan), correlation,
            cosine, euclidian, mahalanobis, minkowski, seuclidian,
            sqeuclidian, wminkowski
    returns a list of distance values
    """
    # make a matrix of the outcomes n_scenarios x outcome_names
    scenario_data = np.zeros((len(scenarios), len(oois)))
    for i, s in enumerate(scenarios):
        for j, ooi in enumerate(oois):
            scenario_data[i][j] = data[ooi][s]

    distances = pdist(scenario_data, distance)
    return distances


def _evaluate_diversity_single(x, outcomes, outcome_names, weight=0.5):
    """
    This function returns a single diversity value for a given scenario set.
    @param x:
    @param outcomes : outcomes dictionary of the scenario ensemble
    @param outcome_names: list with Strings
    @param weight : weight for the mean of the diversity metric. 0: only minimum; 1: only mean
    @param
    """
    distances = _calculate_distance(outcomes, outcome_names, list(x), 'euclidean')
    minimum = np.min(distances)
    mean = np.mean(distances)
    diversity = (1 - weight) * minimum + weight * mean

    return [diversity]


def _find_max_diverse_scenarios(combinations, outcomes, outcome_names):
    """

    @param combinations:
    @return:
    """
    diversity = 0.0
    solutions = []

    for sc_set in combinations:
        temp_div = _evaluate_diversity_single(x=list(sc_set), outcomes=outcomes, outcome_names=outcome_names)
        if temp_div[0] > diversity:
            diversity = temp_div[0]
            solutions = [sc_set]
        elif temp_div[0] == diversity:
            solutions.append(sc_set)
    return diversity, solutions


def _select_scenarios(combos, outcomes, outcome_names):
    """
    @param combos:
    @param outcomes: dictionary
    @param outcome_names: list with Strings
    @return:
    """
    results = _find_max_diverse_scenarios(combos, outcomes, outcome_names)
    return [results]


def _look_up_scenarios(all_scenarios, indices):
    """
    Looks up the indices in all scenarios and returns a dataframe with the final solutions.
    @param all_scenarios: dictionary: contains all loaded scenarios
    @param indices: list with integers
    @return
        solutions: DataFrame
    """
    all_scenarios = pd.DataFrame(all_scenarios)
    solutions = all_scenarios.iloc[indices, :]
    return solutions


def compute_reference_scenarios(scenarios=None, n_ref_scenarios=4, saving=False):
    """
    This function uses the diversity maximization approach to compute a number of reference scenarios.
    @param scenarios: DataFrame
    @param n_ref_scenarios: int: number of desired reference scenarios
    @param saving: Boolean: whether to save the resuls or not
    @return:
        solutions: list with 4 items
    """

    # Loading results
    if scenarios is None:
        target_directory = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'example_scenarios.csv'
        )
        scenarios = pd.read_csv(target_directory, index_col='Unnamed: 0').iloc[:10, :9]  # Load only uncertainties
    scenarios = scenarios.to_dict('series')

    # Preparing outcomes
    scenarios = _normalize(scenarios)
    outcome_names = list(scenarios.keys())

    # Preparing combinations
    n_scenarios = len(scenarios[outcome_names[0]])
    indices = list(range(n_scenarios))
    potential_solutions = []
    n_combinations = _n_combinations(len(indices), n_ref_scenarios)

    # Iterating through the combinations
    with ProcessPoolExecutor() as executor:
        for idx, item in enumerate(itertools.combinations(indices, n_ref_scenarios)):
            print(f'it #{idx}/{n_combinations}') if idx % 500 == 0 else 0
            combos = [item]
            solution = executor.submit(_select_scenarios, combos=combos, outcomes=scenarios, outcome_names=outcome_names)
            potential_solutions.extend(solution.result())

    # Selecting maximum diverse scenarios
    max_diversity = 0.0
    solutions = []
    for r in potential_solutions:
        if r[0] >= max_diversity:
            max_diversity = r[0]
            solutions = [r[1]]
        elif r[0] == max_diversity:
            solutions.append(r[1])

    solutions = list(solutions[0][0])

    if saving:
        scenarios_df = _look_up_scenarios(scenarios, solutions)
        directory = os.path.join(os.getcwd(), 'data', 'reference_scenarios.csv')
        scenarios_df.to_csv(directory)

    return solutions


def merge_all_worst_scenarios(saving=False):
    """
    Merge all worst scenarios resulting from time series clustering and from directed scenario search.
    @param saving: Boolean: whether to save the scenarios or not
    @return:
        all_scenarios: DataFrame
    """

    # Load scenarios from time series clustering (tsc)
    target_directory = os.path.join(os.path.dirname(os.getcwd()), 'clustering/data',  'time_series_scenarios.csv')
    scenarios_tsc = pd.read_csv(target_directory)

    # Load scenarios from directed scenario search (dss)
    target_directory = os.path.join(os.path.dirname(os.getcwd()), 'search/data')
    n = 20000
    searchover = 'uncertainties'
    scenarios_dss = pd.DataFrame()

    for idx, problem_formulation in enumerate(ProblemFormulation.get_8_problem_formulations()):

        folder = f'{problem_formulation}_{searchover}_{n}'
        target_directory = os.path.join(target_directory, folder, 'results.csv')

        new_scenarios = pd.read_csv(target_directory)
        if idx == 0:
            scenarios_dss = new_scenarios
        else:
            scenarios_dss.append(new_scenarios)

    # Merging all scenarios
    all_scenarios = scenarios_tsc.append(scenarios_dss)

    # Save scenarios
    if saving:
        target_directory = os.path.join(os.getcwd(), 'data', 'all_worst_scenarios.csv')
        all_scenarios.to_csv(target_directory)

    return all_scenarios


if __name__ == "__main__":

    scenarios = merge_all_worst_scenarios(saving=True)
    ref_scenarios = compute_reference_scenarios(scenarios=scenarios, n_ref_scenarios=4, saving=True)
    print(f'ref_scenarios: {ref_scenarios}')


