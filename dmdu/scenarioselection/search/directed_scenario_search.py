"""
This module is used to run directed scenario search for the process of scenario discovery.
"""


import pandas as pd
import random
import numpy as np
import os
from dmdu.general.directed_search import run_optimization
from dmdu.exploration.perform_experiments import get_reference_policies
from dmdu.general.timer import *
from model.enumerations import *


def load_scenario_search_results(
        data_folder_path,
        searchover,
        nfe,
        problem_formulations=None,
        n_references=1,
        n_seeds=1
):
    """
    Load all results from directed scenario search.
    @param data_folder_path: String: path to the data folder in which all relevant data is saved
    @param searchover: String: {'levers', 'uncertainties'}
    @param nfe: int: this indicates what the desired nfe is (has to exist in the results)
    @param problem_formulations: list with ProblemFormulation objects
    @param n_references: int: how many reference policies have been used
    @param n_seeds: int: how many seeds have been used
    @return:
        scenarios: DataFrame
    """

    if problem_formulations is None:
        problem_formulations = [pf.name for pf in ProblemFormulation.get_8_problem_formulations()]
    else:
        problem_formulations = [pf.name for pf in problem_formulations]

    # Load results for each problem formulation
    scenarios = None

    reference_name = 'reference_scenario' if searchover == 'levers' else 'reference_policy'

    for problem_formulation in problem_formulations:
        for seed_index in range(n_seeds):
            for n_reference in range(n_references):

                problem_folder = f'{problem_formulation}_{nfe}'
                seed_folder = f'seed_{seed_index}'
                reference_folder = f'{reference_name}_{n_reference}'

                directory = os.path.join(
                    data_folder_path, problem_folder, seed_folder, reference_folder, 'results.csv'
                )

                df = pd.read_csv(directory)
                if scenarios is None:
                    scenarios = df
                else:
                    scenarios = pd.concat([scenarios, df])

    scenarios = scenarios.iloc[:, 1:10]

    return scenarios


if __name__ == "__main__":

    timer = Timer()

    seeds = [9845531]
    problem_formulations = ProblemFormulation.get_8_problem_formulations()
    reference_policies = get_reference_policies()

    for idx, problem_formulation in enumerate(problem_formulations):
        for seed_index, seed in enumerate(seeds):
            for reference_index, reference_policy in enumerate(reference_policies):

                print(
                    f'Running problem formulation {idx+1}/{len(problem_formulations)} ({problem_formulation.name})'
                )

                # Setting seeds
                random.seed(seed)
                np.random.seed(seed)

                run_optimization(
                    problem_formulation=problem_formulation,
                    nfe=200000,
                    searchover='uncertainties',
                    seed_index=seed_index,
                    reference=(reference_index, reference_policy),
                    saving_results=True,
                    with_convergence=True,
                )

    timer.stop()
