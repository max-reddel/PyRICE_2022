"""
This module is used to run directed scenario search for the process of scenario discovery.
"""


import pandas as pd
import os
from optimization.general.directed_search import run_optimization
from optimization.general.timer import *
from model.enumerations import *


def load_scenario_search_results(data_folder_path, searchover, nfe, n_references=1):
    """
    Load all results from directed scenario search.
    @param data_folder_path: String: path to the data folder in which all relevant data is saved
    @param searchover: String: {'levers', 'uncertainties'}
    @param nfe: int: this indicates what the desired nfe is (has to exist in the results)
    @param n_references: int: how many reference policies have been used
    @return:
        scenarios: DataFrame
    """
    problem_formulations = [pf.name for pf in ProblemFormulation.get_8_problem_formulations()]

    # Load results for each problem formulation
    scenarios = None

    reference_name = 'scenario' if searchover == 'levers' else 'policy'

    for idx, problem_formulation in enumerate(problem_formulations):

        for n in range(n_references):
            directory = os.path.join(
                data_folder_path,
                f'{reference_name}_{n}_{problem_formulation}_{searchover}_{nfe}'
            )

            df = pd.read_csv(os.path.join(directory, 'results.csv'))
            if scenarios is None:
                scenarios = df
            else:
                scenarios = pd.concat([scenarios, df])

    scenarios = scenarios.iloc[:, 1:10]

    return scenarios


if __name__ == "__main__":

    timer = Timer()

    problem_formulations = ProblemFormulation.get_8_problem_formulations()

    for idx, problem_formulation in enumerate(problem_formulations):

        print(
            f"Running problem formulation {idx+1}/{len(problem_formulations)} ({problem_formulation.name})"
        )

        run_optimization(
            problem_formulation=problem_formulation,
            nfe=200000,
            searchover="uncertainties",
            saving_results=True,
            with_convergence=True,
        )

    timer.stop()
