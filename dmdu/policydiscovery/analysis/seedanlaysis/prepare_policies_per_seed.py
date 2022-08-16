"""
This module is used to prepare optimal policies in a way that we get them in (experiments, outcomes) format per
problem formulation per seed. This can be used for seed analysis. Without extensive experiments (e.g., 200 scenarios
per policy). It's just one (reference) scenario per policy.
"""
from ema_workbench import Policy

from dmdu.exploration.perform_experiments import perform_own_experiments
from dmdu.general.xlm_constants_epsilons import get_lever_names
from dmdu.policydiscovery.directed_policy_search import PolicyCounter
from dmdu.scenarioselection.selection.scenario_selection import load_reference_scenarios
from model.enumerations import ProblemFormulation
import os
import pandas as pd


if __name__ == '__main__':

    reference_scnearios = load_reference_scenarios()

    problem_formulations = [
        # ProblemFormulation.UTILITARIAN_AGGREGATED,
        ProblemFormulation.UTILITARIAN_DISAGGREGATED,
        # ProblemFormulation.SUFFICIENTARIAN_AGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED
    ]

    target_directory = os.path.join(
        os.path.dirname(os.getcwd()),
        'data'
    )

    n_seeds = 2
    n_references = 4
    nfe = 200000
    searchover = 'levers'
    reference_name = 'reference_scenario' if searchover == 'levers' else 'reference_policy'
    lever_names = get_lever_names()

    counter = PolicyCounter()

    for problem_formulation in problem_formulations:
        for seed_index in range(n_seeds):
            policies = None

            for n_reference in range(n_references):

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

                # Remove outcome columns
                df = pd.read_csv(current_directory, index_col='Unnamed: 0')
                df = df.loc[:, lever_names]

                if policies is None:
                    policies = df
                else:
                    policies = pd.concat([policies, df])

                policy_list = [Policy(f'{counter.next()}', **row) for _, row in policies.iterrows()]

                saving_directory = os.path.join(
                    os.path.dirname(os.getcwd()),
                    'data',
                    'experiments'
                )

                reference_scneario = reference_scnearios[n_reference]

                perform_own_experiments(
                    problem_formulation=ProblemFormulation.ALL_KPIS,
                    n_scenarios=[reference_scneario],
                    n_policies=policy_list,
                    saving_results=True,
                    folder=saving_directory,
                    file_name=f'{problem_formulation.name}_seed_{seed_index}_reference_{n_reference}',
                )
