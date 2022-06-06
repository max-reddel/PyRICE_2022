"""
This module contains functionality to run experiments with discovered policies.
"""
from ema_workbench import Policy

from dmdu.exploration.perform_experiments import perform_own_experiments
from dmdu.general.xlm_constants_epsilons import get_lever_names
from model.enumerations import ProblemFormulation
import os
import pandas as pd


if __name__ == '__main__':

    lever_names = get_lever_names()

    problem_formulations = [
        ProblemFormulation.UTILITARIAN_AGGREGATED,
        ProblemFormulation.UTILITARIAN_DISAGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_AGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED
    ]

    for problem_formulation in problem_formulations:
        target_directory = os.path.join(
            os.getcwd(),
            'data',
            'optimalpolicies',
            f'optimal_policies_{problem_formulation.name}.csv'
        )

        # Load policies
        policies_df = pd.read_csv(target_directory, index_col='Unnamed: 0')
        policies_df = policies_df.loc[:, lever_names]
        policy_list = [Policy(f'{idx}', **row) for idx, row in policies_df.iterrows()]

        saving_directory = os.path.join(
            os.getcwd(),
            'data',
            'experimentsextensive'
        )

        perform_own_experiments(
            problem_formulation=ProblemFormulation.ALL_KPIS,
            n_scenarios=400,
            n_policies=policy_list,
            saving_results=True,
            folder=saving_directory,
            file_name=f'results_{problem_formulation.name}'
        )
