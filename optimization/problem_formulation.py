"""
This module contains functions that support the run of an optimization.
"""

# Imports

from model.pyrice import PyRICE
from model.enumerations import *
from optimization.outcomes_and_epsilons import get_outcomes_and_epsilons
import os

# EMA
from ema_workbench.em_framework.optimization import (EpsilonProgress, ArchiveLogger)
from ema_workbench import (Model, RealParameter, IntegerParameter, MultiprocessingEvaluator, ema_logging, Constant)

ema_logging.log_to_stderr(ema_logging.INFO)


def define_path_name(damage_function, welfare_function, nfe, prefix='results'):
    """
    Define path and file name such that it can be used to save results and/or covergence data.
    @param damage_function: DamageFunction
    @param welfare_function: WelfareFunction
    @param nfe: integer
    @param prefix: string: {'results', 'convergence'}
    @return:
        path: string (path + file name that used for saving)
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
    folder = 'results'
    directory = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    path = os.path.join(directory, folder, damage_function.name,
                        welfare_function.name)

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directory failed")
            raise

    return path


def run_optimization(damage_function=DamageFunction.NORDHAUS,
                     welfare_function=WelfareFunction.UTILITARIAN,
                     nfe=5000,
                     saving_results=False,
                     with_convergence=False):
    """
    This function runs an optimization with the PyRICE model.
    @param damage_function: DamageFunction
    @param welfare_function: WelfareFunction
    @param nfe: integer
    @param saving_results: Boolean: whether to save results or not
    @param with_convergence: Boolean: whether to save convergence data or not
    """

    # Instantiate the model
    model_specification = ModelSpec.STANDARD

    model = PyRICE(model_specification=model_specification,
                   damage_function=damage_function,
                   welfare_function=welfare_function)

    model = Model('RICE', function=model)

    # Specify uncertainties
    model.uncertainties = [IntegerParameter('t2xco2_index', 0, 999),
                           IntegerParameter('t2xco2_dist', 0, 2),
                           RealParameter('fosslim', 4000, 13649),
                           IntegerParameter('scenario_pop_gdp', 0, 5),
                           IntegerParameter('scenario_sigma', 0, 2),
                           IntegerParameter('scenario_cback', 0, 1),
                           IntegerParameter('scenario_elasticity_of_damages', 0, 2),
                           IntegerParameter('scenario_limmiu', 0, 1)]

    # Set levers, one for each time step
    model.levers = [RealParameter('sr', 0.1, 0.5),
                    RealParameter('miu', 2065, 2300),
                    RealParameter('irstp_consumption', 0.001, 0.015)]

    # Specify outcomes
    model.outcomes, epsilons = get_outcomes_and_epsilons(welfare_function=welfare_function)

    model.constants = [Constant('precision', 10)]

    constraints = []

    # Run optimization
    if with_convergence:

        directory = get_directory(damage_function, welfare_function)

        convergence_metrics = [EpsilonProgress(),
                               ArchiveLogger(directory,
                                             [l.name for l in model.levers],
                                             [o.name for o in model.outcomes
                                              if o.kind != o.INFO])]

        with MultiprocessingEvaluator(model, n_processes=50) as evaluator:
            results, convergence = evaluator.optimize(nfe=nfe,
                                                      searchover='levers',
                                                      epsilons=epsilons,
                                                      convergence=convergence_metrics,
                                                      constraints=constraints)

            if saving_results:
                # Save results
                path = define_path_name(damage_function, welfare_function, nfe, prefix='results')
                results.to_csv(path)
                # Save convergence
                path = define_path_name(damage_function, welfare_function, nfe, prefix='convergence')
                convergence.to_csv(path)

    else:

        with MultiprocessingEvaluator(model, n_processes=50) as evaluator:
            results = evaluator.optimize(nfe=nfe,
                                         searchover='levers',
                                         epsilons=epsilons,
                                         constraints=constraints)

            if saving_results:
                path = define_path_name(damage_function, welfare_function, nfe, prefix='results')
                results.to_csv(path)


if __name__ == '__main__':

    n = 100000
    run_optimization(welfare_function=WelfareFunction.UTILITARIAN,
                     damage_function=DamageFunction.NORDHAUS,
                     nfe=n,
                     saving_results=True,
                     with_convergence=True)
