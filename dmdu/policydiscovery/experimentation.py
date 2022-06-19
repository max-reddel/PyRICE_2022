"""
This module contains functionality to run experiments with discovered policies.
"""
from ema_workbench import Policy, Scenario
from dmdu.exploration.perform_experiments import perform_own_experiments
from dmdu.general.xlm_constants_epsilons import get_lever_names, get_uncertainty_names, adjust_integers_in_uncertainties
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

    # Starting with 50 random scenarios
    init_scenarios = 50
    scenarios = init_scenarios

    # Do I want to reuse scenarios?
    reuse_scenarios = True

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

        results = perform_own_experiments(
            problem_formulation=ProblemFormulation.ALL_KPIS,
            n_scenarios=scenarios,
            n_policies=policy_list,
            saving_results=True,
            folder=saving_directory,
            file_name=f'results_{problem_formulation.name}'
        )

        # Overwrite scenarios to the specific set of scenarios that have been used in first iteration
        if reuse_scenarios and (scenarios == init_scenarios):
            experiments, outcomes = results
            x_names = get_uncertainty_names()

            uncertainty_df = experiments.loc[:, x_names]
            scenario_df = uncertainty_df.drop_duplicates()
            scenario_df = adjust_integers_in_uncertainties(scenario_df)

            scenarios_list = [Scenario(f"{index}", **row) for index, row in scenario_df.iterrows()]
            scenarios = scenarios_list
