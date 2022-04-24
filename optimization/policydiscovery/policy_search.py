"""
This module contains functions that support the run of an optimization.
"""

# Imports

from model.pyrice import PyRICE
from model.enumerations import *
from optimization.general.xlm_constants_epsilons import get_outcomes_and_epsilons, get_xlc
import os

# EMA
from ema_workbench.em_framework.optimization import (EpsilonProgress, ArchiveLogger)
from ema_workbench import (Model, MultiprocessingEvaluator, ema_logging)
ema_logging.log_to_stderr(ema_logging.INFO)


def define_path_name(damage_function, welfare_function, nfe, prefix='results_formatted'):
    """
    Define path and file name such that it can be used to save results_formatted and/or covergence data.
    @param damage_function: DamageFunction
    @param welfare_function: WelfareFunction
    @param nfe: integer
    @param prefix: string: {'results_formatted', 'convergence'}
    @return:
        path: string (path + file name that used for saving_results)
    """

    file_name = damage_function.name + '_' + \
                welfare_function.name + '_' + \
                str(nfe) + '_' + \
                prefix + \
                '.csv'

    directory = get_directory(damage_function, welfare_function)
    path = os.path.join(directory, file_name)

    return path


def get_directory(damage_function, welfare_function):
    """
    Create a directory if necessary.
    @param damage_function: DamageFunction
    @param welfare_function: WelfareFunction
    @return:
        path: string
    """
    folder = 'results_formatted'
    directory = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    path = os.path.join(directory, folder, damage_function.name, welfare_function.name)

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directory failed")
            raise

    return path


def run_optimization(
        damage_function=DamageFunction.NORDHAUS,
        problem_formulation=ProblemFormulation.UITILITARIAN_AGGREGATED,
        nfe=5000,
        saving_results=False,
        with_convergence=False):
    """
    This function runs an optimization with the PyRICE model.
    @param damage_function: DamageFunction
    @param problem_formulation: ProblemFormulation
    @param nfe: integer
    @param saving_results: Boolean: whether to save results_formatted or not
    @param with_convergence: Boolean: whether to save convergence data or not
    """

    welfare_function, aggregation = problem_formulation.value

    # Instantiate the model
    model_specification = ModelSpec.STANDARD

    model = PyRICE(model_specification=model_specification,
                   damage_function=damage_function,
                   welfare_function=welfare_function)

    model = Model('RICE', function=model)

    model.uncertainties, model.levers, model.constants = get_xlc()
    model.outcomes, epsilons = get_outcomes_and_epsilons(problem_formulation=problem_formulation)

    # Run optimization
    if with_convergence:

        directory = get_directory(damage_function, welfare_function)
        convergence_metrics = [
            EpsilonProgress(),
            ArchiveLogger(
                directory, [l.name for l in model.levers], [o.name for o in model.outcomes if o.kind != o.INFO]
            )
        ]

        with MultiprocessingEvaluator(model, n_processes=50) as evaluator:
            results, convergence = evaluator.optimize(
                nfe=nfe,
                searchover='levers',
                epsilons=epsilons,
                convergence=convergence_metrics
            )

            if saving_results:
                # Save results_formatted
                path = define_path_name(damage_function, welfare_function, nfe, prefix='results_formatted')
                results.to_csv(path)
                # Save convergence
                path = define_path_name(damage_function, welfare_function, nfe, prefix='convergence')
                convergence.to_csv(path)

    else:

        with MultiprocessingEvaluator(model, n_processes=50) as evaluator:
            results = evaluator.optimize(
                nfe=nfe,
                searchover='levers',
                epsilons=epsilons
            )

            if saving_results:
                path = define_path_name(damage_function, welfare_function, nfe, prefix='results_formatted')
                results.to_csv(path)


if __name__ == '__main__':
    run_optimization(
        damage_function=DamageFunction.NORDHAUS,
        problem_formulation=ProblemFormulation.UITILITARIAN_AGGREGATED,
        nfe=100000,
        saving_results=True,
        with_convergence=True
    )