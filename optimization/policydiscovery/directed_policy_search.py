"""
This module is used to run directed policy search.
"""


from optimization.general.directed_search import run_optimization
from optimization.general.timer import *
from model.enumerations import *
import os
import pandas as pd
from optimization.scenariodiscovery.selection.scenario_selection import load_reference_scenarios
from ema_workbench import Policy


def load_optimal_policies(target_directory, searchover='levers', nfe=200000):
    """
    Loads Pareto-optimal policies
    @param target_directory: String
    @param searchover: String: {'uncertainties', 'levers'}
    @param nfe: int
    @return:
        policies: list with Policy Objects

    """
    problem_formulations = ProblemFormulation.get_8_problem_formulations()

    policies = []

    for idx, problem_formulation in enumerate(problem_formulations):

        folder = f'{problem_formulation.name}_{searchover}_{nfe}'
        current_directory = os.path.join(target_directory, folder, 'results.csv')

        df = pd.read_csv(current_directory, index_col='Unnamed: 0')
        if idx == 0:
            policies = df
        else:
            policies.append(df)

    policies = [Policy('idx', **row) for idx, row in policies.iterrows()]

    return policies


if __name__ == "__main__":

    timer = Timer()

    problem_formulations = ProblemFormulation.get_8_problem_formulations()
    reference_scenarios = load_reference_scenarios()

    for idx, problem_formulation in enumerate(problem_formulations):

        print(f'Running problem formulation {idx + 1}/{len(problem_formulations)} ({problem_formulation.name})')
        for ref_idx, reference_scenario in enumerate(reference_scenarios):

            run_optimization(
                problem_formulation=problem_formulation,
                nfe=10,
                searchover='levers',
                reference=(ref_idx, reference_scenario),
                saving_results=True,
                with_convergence=True,
            )

        timer.stop()
