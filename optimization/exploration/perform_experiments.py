"""
This module contains a function to perform experiments and a function to get the standard lists for uncertainties,
levers, and constants.
"""

# Imports
from model.pyrice import PyRICE
from optimization.general.xlm_constants_epsilons import *
import os
# EMA
from ema_workbench import (Model, Policy, ema_logging, MultiprocessingEvaluator)
from ema_workbench.util.utilities import save_results
ema_logging.log_to_stderr(ema_logging.INFO)


def get_reference_policies():
    """
    Return appropriate reference policies for open exploration.
    @return:
        policies: list with Policy objects
    """
    policies = []

    nordhaus_policy = Policy(
        'Nordhaus',
        **{'sr': 0.248, 'miu': 2135, 'irstp_consumption': 0.015, 'irstp_damage': 0.015}
    )
    policies.append(nordhaus_policy)

    return policies


def perform_own_experiments(
        damage_function=DamageFunction.NORDHAUS,
        n_scenarios=100,
        n_policies=None,
        saving_results=False,
        file_name=None
):
    """
    Perform a bunch of experiments and return the results.
    @param damage_function: DamageFunction
    @param n_scenarios: int: number of scenarios
    @param n_policies: int: number of policies
    @param saving_results: Boolean: whether to save the results or not
    @param file_name: String: name of file to save
    @return:
        results: dataframe, dictionary: experiments, outcomes
    """

    model = PyRICE(model_specification=ModelSpec.STANDARD,
                   damage_function=damage_function,
                   welfare_function=WelfareFunction.UTILITARIAN)

    model = Model('RICE', function=model)

    model.uncertainties, model.levers, model.constants = get_xlc()

    if n_policies is None:
        # Use specific reference policies instead of a number of policies
        n_policies = get_reference_policies()

    model.outcomes, _ = get_outcomes_and_epsilons(problem_formulation=ProblemFormulation.ALL_OBJECTIVES)

    with MultiprocessingEvaluator(model) as evaluator:

        results = evaluator.perform_experiments(
            scenarios=n_scenarios,
            policies=n_policies,
            reporting_frequency=100
        )

        if saving_results:

            if file_name is None:
                file_name = f'results_open_exploration_{n_scenarios}'

            target_directory = os.getcwd() + '/results/'

            save_results(results=results, file_name=target_directory + file_name)

    return results


if __name__ == '__main__':

    printing = False

    n = 100
    results = perform_own_experiments(
        n_scenarios=n,
        saving_results=True,
        file_name=f'results_open_exploration_{n}'
    )

    if printing:
        experiments, outcomes = results

        for o in outcomes.items():
            print(o)