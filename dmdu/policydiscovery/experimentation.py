"""
This module contains functionality to run experiments with discovered policies.
There are four ways to run experiments with different kind of scenarios.

    - 'reference_scenarios': Load and use 4 reference scenarios
    - 'bad': Load random 50 scenarios that we have identified in all worst scenarios (from scenario discovery process)
    - 'random_reused': Sample 50 random scenarios but reuse them for each problem formulation (better comparability)
    - 'random_not_reused': Sample 50 random scenarios without reusing them.
"""
from ema_workbench import Policy, Scenario
from dmdu.exploration.perform_experiments import perform_own_experiments
from dmdu.general.xlm_constants_epsilons import get_lever_names, get_uncertainty_names, adjust_integers_in_uncertainties
from dmdu.policydiscovery.directed_policy_search import PolicyCounter
from dmdu.scenarioselection.selection.scenario_selection import load_reference_scenarios, load_n_bad_scenarios
from model.enumerations import ProblemFormulation
import os
import pandas as pd


if __name__ == '__main__':

    scenario_types = ['references', 'bad', 'random_reused', 'random_not_reused']
    scenario_type = scenario_types[1]  # Change index here

    lever_names = get_lever_names()
    counter = PolicyCounter()

    problem_formulations = [
        ProblemFormulation.UTILITARIAN_AGGREGATED,
        ProblemFormulation.UTILITARIAN_DISAGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_AGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED,
        ProblemFormulation.EGALITARIAN_AGGREGATED,
        ProblemFormulation.EGALITARIAN_DISAGGREGATED,
        ProblemFormulation.PRIORITARIAN_AGGREGATED,
        ProblemFormulation.PRIORITARIAN_DISAGGREGATED
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
            f'experiments_{scenario_type}'
        )

        init_scenarios = 50

        # Settle on what scenarios should be used
        if scenario_type == 'references':
            scenarios = load_reference_scenarios()
        elif scenario_type == 'bad':
            scenarios = load_n_bad_scenarios(n_samples=50)
        elif scenario_type == 'random_reused' or scenario_type == 'random_not_reused':
            scenarios = init_scenarios
        else:
            raise ValueError('Use a valid way of choosing your scenarios.')

        # Experiments
        results = perform_own_experiments(
            problem_formulation=ProblemFormulation.ALL_KPIS,
            n_scenarios=scenarios,
            n_policies=policy_list,
            saving_results=True,
            folder=saving_directory,
            file_name=f'results_regional_{problem_formulation.name}'
        )

        # Overwrite scenarios to the specific set of scenarios that have been used in first iteration
        if (scenario_type == 'random_reused') and (scenarios == init_scenarios):
            experiments, outcomes = results
            x_names = get_uncertainty_names()

            uncertainty_df = experiments.loc[:, x_names]
            scenario_df = uncertainty_df.drop_duplicates()
            scenario_df = adjust_integers_in_uncertainties(scenario_df)

            scenarios_list = [Scenario(f"{index}", **row) for index, row in scenario_df.iterrows()]
            scenarios = scenarios_list
