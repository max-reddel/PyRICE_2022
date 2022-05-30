"""
This module contains functions that support the run of an optimization.
"""

# Imports

from model.pyrice import PyRICE
from model.enumerations import *
from dmdu.general.xlm_constants_epsilons import (
    get_outcomes_and_epsilons,
    get_xlc,
)
import os

# EMA
from ema_workbench.em_framework.optimization import EpsilonProgress, ArchiveLogger
from ema_workbench import Model, MultiprocessingEvaluator, ema_logging

ema_logging.log_to_stderr(ema_logging.INFO)


def define_path_name(problem_formulation, nfe, seed_index, ref_index, directory=None, d_type="results", searchover=None):
    """
    Define path and file name such that it can be used to save results_formatted and/or covergence outcomes.
    @param problem_formulation: ProblemFormulation
    @param nfe: integer
    @param seed_index: int
    @param ref_index: int: number of current reference scenario/policy
    @param directory: String: where to save results and covergence outcomes
    @param d_type: string: {'results_formatted', 'convergence'}
    @param searchover: String
    @return:
        path: string (path + file name that is used for saving_results)
    """

    if d_type == 'results' or d_type == 'epsilon_progress':
        file_name = f'{d_type}.csv'
    elif d_type == 'hypervolume':
        file_name = ''
    else:
        raise ValueError(
            'You passed an unvalid d_type in order to save your resulting outcomes.'
        )

    if directory is None:
        directory = get_directory(d_type, searchover, seed_index, problem_formulation, nfe, ref_index)

    path = os.path.join(directory, file_name)

    return path


def get_directory(d_type, searchover, seed_index, problem_formulation, nfe, ref_index):
    """
    Create a directory if necessary.
    @return:
        path: string
    """
    reference_name = 'reference_scenario' if searchover == 'levers' else 'reference_policy'

    directory = os.path.abspath(os.getcwd())
    data_folder = 'data'
    problem_folder = f'{problem_formulation.name}_{nfe}'
    seed_folder = f'seed_{seed_index}'
    scenario_folder = f'{reference_name}_{ref_index}'

    if d_type == 'hypervolume':
        sub_folder = d_type
        path = os.path.join(directory, data_folder, problem_folder, seed_folder, scenario_folder, sub_folder)
    else:
        path = os.path.join(directory, data_folder, problem_folder, seed_folder, scenario_folder)

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directory failed")
            raise

    return path


def run_optimization(
    damage_function=DamageFunction.NORDHAUS,
    problem_formulation=ProblemFormulation.UTILITARIAN_AGGREGATED,
    nfe=5000,
    searchover='levers',
    seed_index=None,
    reference=None,
    saving_results=False,
    with_convergence=False,
):
    """
    This function runs an optimization with the PyRICE model.
    @param damage_function: DamageFunction
    @param problem_formulation: ProblemFormulation
    @param nfe: integer
    @param searchover: String: {'levers', 'uncertainties'}
    @param seed_index: int
    @param reference: tuple with (index, Scenario/Policy object)
    @param saving_results: Boolean: whether to save results_formatted or not
    @param with_convergence: Boolean: whether to save convergence outcomes or not
    """
    welfare_function, aggregation, _ = problem_formulation.value

    # Instantiate the model
    model_specification = ModelSpec.STANDARD

    model = PyRICE(
        model_specification=model_specification,
        damage_function=damage_function,
        welfare_function=welfare_function,
    )

    model = Model("RICE", function=model)

    model.uncertainties, model.levers, model.constants = get_xlc()
    model.outcomes, epsilons = get_outcomes_and_epsilons(problem_formulation=problem_formulation, searchover=searchover)

    # look at reference
    if reference is None:
        ref_index = 0
    else:
        ref_index = reference[0]
        reference = reference[1]

    # Run optimization
    if with_convergence:

        directory = define_path_name(
            problem_formulation=problem_formulation,
            nfe=nfe,
            seed_index=seed_index,
            ref_index=ref_index,
            d_type="hypervolume",
            searchover=searchover
        )
        convergence_metrics = [
            EpsilonProgress(),
            ArchiveLogger(
                directory,
                [l.name for l in model.levers],
                [o.name for o in model.outcomes if o.kind != o.INFO],
            ),
        ]

        with MultiprocessingEvaluator(model) as evaluator:
            results, convergence = evaluator.optimize(
                nfe=nfe,
                searchover=searchover,
                epsilons=epsilons,
                convergence=convergence_metrics,
                reference=reference
            )

            if saving_results:
                # Save results_formatted
                path = define_path_name(
                    problem_formulation=problem_formulation,
                    nfe=nfe,
                    seed_index=seed_index,
                    ref_index=ref_index,
                    d_type="results",
                    searchover=searchover,
                )
                results.to_csv(path)

                # Save convergence
                path = define_path_name(
                    problem_formulation=problem_formulation,
                    nfe=nfe,
                    seed_index=seed_index,
                    ref_index=ref_index,
                    d_type="epsilon_progress",
                    searchover=searchover,
                )
                convergence.to_csv(path)

    else:

        with MultiprocessingEvaluator(model, n_processes=50) as evaluator:
            results = evaluator.optimize(
                nfe=nfe,
                searchover=searchover,
                epsilons=epsilons,
                reference=reference
            )

            if saving_results:
                path = define_path_name(
                    problem_formulation=problem_formulation,
                    nfe=nfe,
                    seed_index=seed_index,
                    ref_index=ref_index,
                    d_type="results",
                    searchover=searchover,
                )
                results.to_csv(path)


if __name__ == "__main__":

    run_optimization(
        damage_function=DamageFunction.NORDHAUS,
        problem_formulation=ProblemFormulation.UTILITARIAN_AGGREGATED,
        nfe=200000,
        searchover="uncertainties",
        saving_results=False,
        with_convergence=False,
    )
