"""
Same as in the notebook but as a script.
"""

from model.pyrice import PyRICE
from model.enumerations import *
from exploration import *
import os

from ema_workbench import (Model, RealParameter, IntegerParameter, MultiprocessingEvaluator, ema_logging)
from ema_workbench.util.utilities import save_results

ema_logging.log_to_stderr(ema_logging.INFO)

if __name__ == '__main__':

    parent_directory = os.path.dirname(os.getcwd())
    target_directory = parent_directory + '/exploration/results/'

    # Instantiate the model
    model_specification = ModelSpec.STANDARD
    damage_function = DamageFunction.NORDHAUS
    welfare_function = WelfareFunction.SUFFICIENTARIAN

    model = PyRICE(model_specification=model_specification,
                   damage_function=damage_function,
                   welfare_function=welfare_function)

    model = Model('RICE', function=model)

    # Specify uncertainties
    model.uncertainties = [
        IntegerParameter('t2xco2_index', 0, 999),
        IntegerParameter('t2xco2_dist', 0, 2),
        RealParameter('fosslim', 4000, 13649),
        IntegerParameter('scenario_pop_gdp', 0, 5),
        IntegerParameter('scenario_sigma', 0, 2),
        IntegerParameter('scenario_cback', 0, 1),
        IntegerParameter('scenario_elasticity_of_damages', 0, 2),
        IntegerParameter('scenario_limmiu', 0, 1),
        RealParameter('emdd', 0.001, 0.6)
    ]

    # Set levers, one for each time step
    model.levers = [
        RealParameter('sr', 0.1, 0.5),
        RealParameter('miu', 2065, 2300),
        RealParameter('irstp_consumption', 0.001, 0.015),
        RealParameter('irstp_damage', 0.001, 0.015)
    ]

    # Define relevant outcome variables (time series)
    outcome_names = [
        'Distance to consumption threshold',
        'Distance to damage threshold',
        'Population below consumption threshold',
        'Population above damage threshold',
        'Utility',
        'Disutility'
    ]
    model.outcomes = prepare_info_outcomes(outcome_names)

    loading = True
    n = 100
    file_name = f'results_new_damage_threshold_implementation_{n*n}'

    if not loading:
        # Run experiments
        with MultiprocessingEvaluator(model) as evaluator:
            results = evaluator.perform_experiments(scenarios=n, policies=n)
            save_results(results=results, file_name=target_directory + file_name)
    else:
        # Loading results
        results = load_results(file_name=target_directory + file_name)

    experiments, outcomes = results
    outcomes = pd.DataFrame(outcomes)
    plot_pathways(outcomes, outcome_names)
