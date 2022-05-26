"""
This module is used to run directed policy search.
"""


from optimization.general.directed_search import run_optimization
from model.enumerations import *
import os
import pandas as pd
from optimization.scenariodiscovery.selection.scenario_selection import load_reference_scenarios
from optimization.general.xlm_constants_epsilons import get_lever_names
from ema_workbench import Policy


def load_optimal_policies(target_directory, problem_formulations=None, searchover='levers', nfe=200000, n_references=4):
    """
    Loads Pareto-optimal policies
    @param target_directory: String
    @param problem_formulations: list with ProblemFormulation objects
    @param searchover: String: {'uncertainties', 'levers'}
    @param nfe: int
    @param n_references: int: number of used reference scenarios
    @return:
        policies: list with Policy Objects

    """
    if problem_formulations is None:
        problem_formulations = ProblemFormulation.get_util_and_prio_problem_formulations()

    policies = None

    reference_name = 'scenario' if searchover == 'levers' else 'policy'

    for idx, problem_formulation in enumerate(problem_formulations):

        for n in range(n_references):

            folder = f'{reference_name}_{n}_{problem_formulation.name}_{searchover}_{nfe}'
            current_directory = os.path.join(target_directory, folder, 'results.csv')

            df = pd.read_csv(current_directory, index_col='Unnamed: 0')
            if policies is None:
                policies = df
            else:
                policies = pd.concat([policies, df])

    # Keep only lever columns
    lever_names = get_lever_names()
    policies = policies.loc[:, lever_names]

    policies = [Policy('idx', **row) for idx, row in policies.iterrows()]

    return policies


if __name__ == "__main__":

    fully_fledged = True

    if fully_fledged:
        problem_formulations = ProblemFormulation.get_util_and_prio_problem_formulations()
        reference_scenarios = load_reference_scenarios()

        for idx, problem_formulation in enumerate(problem_formulations):

            print(f'Running problem formulation {idx + 1}/{len(problem_formulations)} ({problem_formulation.name})')

            # Running the optimization for each reference scenario
            for ref_idx, reference_scenario in enumerate(reference_scenarios):

                run_optimization(
                    problem_formulation=problem_formulation,
                    nfe=200000,
                    searchover='levers',
                    reference=(ref_idx, reference_scenario),
                    saving_results=True,
                    with_convergence=True,
                )
    else:

        problem_formulation = ProblemFormulation.UTILITARIAN_DISAGGREGATED
        run_optimization(
            problem_formulation=problem_formulation,
            nfe=250000,
            searchover='levers',
            saving_results=True,
            with_convergence=True,
        )
