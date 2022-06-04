"""
This module is used to run experiments with Pareto-optimal policies per problem formulation per seed.
The emphasis is on the seeds.
"""


import pandas as pd
from ema_workbench import Policy

from dmdu.exploration.perform_experiments import perform_own_experiments
from dmdu.general.xlm_constants_epsilons import get_lever_names
from model.enumerations import ProblemFormulation
import os


def load_optimal_policies_per_seed(target_directory, problem_formulation, seed_index, nfe=200000, n_references=1):
    """
    Load the optimal policies for a specific problem formulation and seed.
    @param target_directory: string
    @param problem_formulation: ProblemFormulation
    @param seed_index: int
    @param nfe: int
    @param n_references: int: how many reference scenarios have been used
    @return policies: list with Policy objects
    """

    policies = None
    reference_name = 'reference_scenario'

    for n_reference in range(n_references):

        problem_folder = f'{problem_formulation.name}_{nfe}'
        seed_folder = f'seed_{seed_index}'
        reference_folder = f'{reference_name}_{n_reference}'
        current_directory = os.path.join(target_directory, problem_folder, seed_folder, reference_folder, 'results.csv')

        df = pd.read_csv(current_directory, index_col='Unnamed: 0')
        if policies is None:
            policies = df
        else:
            policies = pd.concat([policies, df])

    # Keep only lever columns
    lever_names = get_lever_names()
    policies = policies.loc[:, lever_names]
    policies = [Policy(f'{idx}', **row) for idx, row in policies.iterrows()]

    return policies


if __name__ == '__main__':

    problem_formulations = [
        ProblemFormulation.PRIORITARIAN_DISAGGREGATED,
        ProblemFormulation.PRIORITARIAN_AGGREGATED,
    ]

    policy_dict = {}  # {(ProblemFormulation, seed_idx): list with policies}
    n_seeds = 2

    policy_directory = os.path.join(os.getcwd(), 'data')
    for problem_formulation in problem_formulations:
        for seed_idx in range(n_seeds):
            policy_dict[problem_formulation, seed_idx] = load_optimal_policies_per_seed(
                target_directory=policy_directory,
                problem_formulation=problem_formulation,
                nfe=10,
                seed_index=seed_idx
            )

    target_directory = os.path.join(os.getcwd(), 'data', 'experiments')

    for (problem_formulation, seed_idx), policies in policy_dict.items():
        perform_own_experiments(
            problem_formulation=ProblemFormulation.ALL_KPIS,
            n_scenarios=10,
            n_policies=policies,
            saving_results=True,
            folder=target_directory,
            file_name=f'{problem_formulation.name}_seed_{seed_idx}',
        )
