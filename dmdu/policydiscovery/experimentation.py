"""
This module contains functionality to run experiments with discovered policies.
"""

from dmdu.exploration.perform_experiments import perform_own_experiments
from dmdu.policydiscovery.directed_policy_search import load_optimal_policies
import os

if __name__ == '__main__':

    target_directory = os.path.join(os.getcwd(), 'data')
    policies = load_optimal_policies(target_directory=target_directory, nfe=10, n_references=1)

    perform_own_experiments(
        n_scenarios=100,
        n_policies=policies,
        saving_results=True,
        folder=target_directory,
        file_name='exploration_with_optimal_policies',
    )
