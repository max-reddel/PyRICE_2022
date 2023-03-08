"""
This module is used to run directed policy search.
"""


from dmdu.general.directed_search import run_optimization
from model.enumerations import *
import os
import random
import numpy as np
import pandas as pd
from dmdu.scenarioselection.selection.scenario_selection import load_reference_scenarios
from dmdu.general.xlm_constants_epsilons import get_lever_names
from ema_workbench import Policy


class PolicyCounter(object):
    """
    A helper object
    """

    def __init__(self):
        self.counter = 0

    def next(self):
        """
        Increment and return next policy id.
        @return:
            next_id: int
        """
        next_id = self.counter
        self.counter += 1
        return next_id


def load_optimal_policies(
        target_directory,
        problem_formulations=None,
        searchover='levers',
        nfe=200000,
        n_references=4,
        n_seeds=1,
):
    """
    Loads Pareto-optimal policies
    @param target_directory: String
    @param problem_formulations: list with ProblemFormulation objects
    @param searchover: String: {'uncertainties', 'levers'}
    @param nfe: int
    @param n_references: int: number of used reference scenarios
    @param n_seeds: int: how many seeds have been used
    @return:
        optimal_policies: dictionary with {ProblemFormulation: list with Policy objects}

    """
    if problem_formulations is None:
        problem_formulations = ProblemFormulation.get_util_and_suff_problem_formulations()

    optimal_policies = {}
    policies = None
    counter = PolicyCounter()

    reference_name = 'reference_scenario' if searchover == 'levers' else 'reference_policy'

    for problem_formulation in problem_formulations:
        for seed_index in range(n_seeds):
            for n_reference in range(n_references):

                problem_folder = f'{problem_formulation.name}_{nfe}'
                seed_folder = f'seed_{seed_index}'
                reference_folder = f'{reference_name}_{n_reference}'

                current_directory = os.path.join(
                    target_directory, problem_folder, seed_folder, reference_folder, 'results.csv'
                )

                df = pd.read_csv(current_directory, index_col='Unnamed: 0')
                if policies is None:
                    policies = df
                else:
                    policies = pd.concat([policies, df])

        # Keep only lever columns
        lever_names = get_lever_names()
        policies = policies.loc[:, lever_names]

        policies = [Policy(f'{counter.next()}', **row) for _, row in policies.iterrows()]

        optimal_policies[problem_formulation] = policies
        policies = None

    return optimal_policies


def load_optimal_policy_dataframes(
        target_directory,
        problem_formulations,
        searchover='levers',
        nfe=200000,
        n_references=4,
        n_seeds=1,
):
    """
    Loads dataframe from Pareto-optimal policies
    @param target_directory: String
    @param problem_formulations: list with ProblemFormulation objects
    @param searchover: String: {'uncertainties', 'levers'}
    @param nfe: int
    @param n_references: int: number of used reference scenarios
    @param n_seeds: int: how many seeds have been used
    @return:
        optimal_policies: dictionary with {ProblemFormulation: DataFrame}

    """

    optimal_policies = {}
    reference_name = 'reference_scenario' if searchover == 'levers' else 'reference_policy'

    for problem_formulation in problem_formulations:
        policies = None
        for seed_index in range(n_seeds):
            for n_reference in range(n_references):

                if (seed_index == 1) and ((n_reference == 2) or (n_reference == 3)) and (problem_formulation == ProblemFormulation.EGALITARIAN_DISAGGREGATED):
                    continue
                else:
                    problem_folder = f'{problem_formulation.name}_{nfe}'
                    seed_folder = f'seed_{seed_index}'
                    reference_folder = f'{reference_name}_{n_reference}'

                    current_directory = os.path.join(
                        target_directory,
                        problem_folder,
                        seed_folder,
                        reference_folder,
                        'results.csv'
                    )

                    df = pd.read_csv(current_directory, index_col='Unnamed: 0')
                    df = df.loc[:, get_lever_names()]

                    if policies is None:
                        policies = df
                    else:
                        policies = pd.concat([policies, df])
                        policies.index = list(range(len(policies)))

        optimal_policies[problem_formulation] = policies

    return optimal_policies


if __name__ == "__main__":

    seeds = [
        9845531,
        1644652,
        3569126,
        6075612,
        521475,
    ]
    problem_formulations = ProblemFormulation.get_8_problem_formulations()
    reference_scenarios = load_reference_scenarios()

    for problem_formulation in problem_formulations:
        for seed_index, seed in enumerate(seeds):
            for reference_index, reference_scenario in enumerate(reference_scenarios):

                # Setting seeds
                random.seed(seed)
                np.random.seed(seed)

                # Running optimizations
                run_optimization(
                    problem_formulation=problem_formulation,
                    nfe=200000,
                    searchover='levers',
                    seed_index=seed_index,
                    reference=(reference_index, reference_scenario),
                    saving_results=True,
                    with_convergence=True
                )
