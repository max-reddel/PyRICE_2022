"""
This module contains functionality to run experiments with discovered policies.
"""
from ema_workbench import Policy
from dmdu.exploration.perform_experiments import perform_own_experiments
from dmdu.general.xlm_constants_epsilons import get_lever_names
from dmdu.policydiscovery.directed_policy_search import PolicyCounter
from dmdu.scenariodiscovery.selection.scenario_selection import load_reference_scenarios
from model.enumerations import ProblemFormulation
import os
import pandas as pd


if __name__ == '__main__':

    lever_names = get_lever_names()
    reference_scenarios = load_reference_scenarios()
    counter = PolicyCounter()

    problem_formulations = [
        ProblemFormulation.UTILITARIAN_AGGREGATED,
        ProblemFormulation.UTILITARIAN_DISAGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_AGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED
    ]

    for problem_formulation in problem_formulations:
        target_directory = os.path.join(
            os.getcwd(),
            'paretosorting',
            'data',
            'final',
            f'sorted_{problem_formulation.name}.csv'
        )

        # Load policies
        policies_df = pd.read_csv(target_directory, index_col='Unnamed: 0')
        policies_df = policies_df.loc[:, lever_names]
        policy_list = [Policy(f'{counter.next()}', **row) for _, row in policies_df.iterrows()]

        saving_directory = os.path.join(
            os.getcwd(),
            'data',
            'experimentsextensive'
        )

        perform_own_experiments(
            problem_formulation=ProblemFormulation.ALL_KPIS,
            n_scenarios=50,
            n_policies=policy_list,
            saving_results=True,
            folder=saving_directory,
            file_name=f'results_{problem_formulation.name}'
        )
