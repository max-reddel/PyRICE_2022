"""
This module contains functions to specify uncertainties, outcomes, levers, constants, and to compute epsilon values for
the optimization process.
"""

from ema_workbench import ScalarOutcome, RealParameter, IntegerParameter, Constant
from model.enumerations import *


def get_xlc():
    """
    Get standard levers, uncertainties and constants for performing experiments or running optimizations.
    @return:
        levers: list of Parameters
        uncertainties: list of Parameters
        constants: list of Constants
    """
    levers = [
        RealParameter('sr', 0.1, 0.5),
        RealParameter('miu', 2065, 2300),
        RealParameter('irstp_consumption', 0.001, 0.015),
        RealParameter('irstp_damage', 0.001, 0.015)
    ]

    uncertainties = [
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

    constants = [Constant('precision', 10)]

    return uncertainties, levers, constants


def _get_outcomes_for_years(outcome_name, years_list, direction):
    """
    Return a list of outcomes with their proper names (including years) and their direction of optimization.
    @param outcome_name: stirng
    @param years_list: list
    @param direction: ScalarOutcome.kind
    @return:
        outcomes: list with ScalarOutcome objects
    """

    outcomes = []

    for year in years_list:
        outcome_name_with_year = outcome_name + ' ' + str(year)
        o = ScalarOutcome(outcome_name_with_year, direction)
        outcomes.append(o)

    return outcomes


def _get_outcomes_to_optimize(outcomes_maximize_names, outcomes_minimize_names, years_optimize):
    """
    Return all the outcomes that should be optimized.
    @param outcomes_maximize_names: list of outcome names (without years)
    @param outcomes_minimize_names: list of outcome names (without years)
    @param years_optimize: list with integers
    @return:
        outcomes_optimize: list with ScalarOutcome objects
    """
    outcomes_optimize = []

    for outcome_name in outcomes_maximize_names:
        outcomes_optimize += _get_outcomes_for_years(outcome_name, years_optimize, ScalarOutcome.MAXIMIZE)

    for outcome_name in outcomes_minimize_names:
        outcomes_optimize += _get_outcomes_for_years(outcome_name, years_optimize, ScalarOutcome.MINIMIZE)

    return outcomes_optimize


def _get_outcomes_to_info(outcomes_info_names, outcomes_optimize_names, years_optimize, years_info):
    """
    Return all the outcomes that should not be optimized but just presented as info.
    @param outcomes_info_names: list of outcome names (without years)
    @param outcomes_optimize_names: list of outcome names (without years)
    @param years_optimize: list with integers
    @param years_info: list with integers
    @return:
        outcomes_info: list with ScalarOutcome objects
    """

    outcomes_info = []

    for outcome_name in outcomes_info_names:
        outcomes_info += _get_outcomes_for_years(outcome_name, years_optimize + years_info, ScalarOutcome.INFO)

    # Add remaining variables
    for outcome_name in outcomes_optimize_names:
        outcomes_info += _get_outcomes_for_years(outcome_name, years_info, ScalarOutcome.INFO)

    return outcomes_info


def _get_relevant_outcomes(
        outcomes_all_names, outcomes_maximize_names, outcomes_minimize_names, outcomes_maximize_aggregated,
        outcomes_minimize_aggregated, years_optimize, years_info, outcomes_info_aggregated
):
    """
    This is a helper function. It returns the outcomes given a specification of all the above variables.
    @param outcomes_all_names: list of outcome names (without years)
    @param outcomes_maximize_names: list of outcome names (without years)
    @param outcomes_minimize_names: list of outcome names (without years)
    @param outcomes_maximize_aggregated: list of outcome names (without years)
    @param outcomes_minimize_aggregated: list of outcome names (without years)
    @param years_optimize: list with integers
    @param years_info: list with integers
    @param outcomes_info_aggregated: list of outcome names (without years)
    @return:
        outcomes: list with ScalarOutcome objects
    """

    # Outcomes that should be optimized
    outcomes_optimize = []

    for outcome in outcomes_maximize_aggregated:
        outcomes_optimize.append((ScalarOutcome(outcome, ScalarOutcome.MAXIMIZE)))

    for outcome in outcomes_minimize_aggregated:
        outcomes_optimize.append((ScalarOutcome(outcome, ScalarOutcome.MINIMIZE)))

    outcomes_optimize += _get_outcomes_to_optimize(outcomes_maximize_names, outcomes_minimize_names, years_optimize)

    # Outcomes that should only be displayed (i.e., not optimized)
    outcomes_info_names = list(
        set(outcomes_all_names)
        - set(outcomes_maximize_names)
        - set(outcomes_minimize_names)
    )

    outcomes_info = []
    for outcome in outcomes_info_aggregated:
        outcomes_info.append((ScalarOutcome(outcome, ScalarOutcome.INFO)))

    outcomes_info += _get_outcomes_to_info(
        outcomes_info_names,
        outcomes_minimize_names + outcomes_maximize_names,
        years_optimize,
        years_info
    )

    # Put them all together
    outcomes = outcomes_optimize + outcomes_info

    return outcomes


def _get_epsilons(
        dict_epsilons, years_optimize, outcomes_maximize_names, outcomes_minimize_names, outcomes_maximize_aggregated,
        outcomes_minimize_aggregated
):
    """
    Calculate epsilon values.
    @param dict_epsilons: dictionary with {outcome_name: epsilon}
    @param years_optimize: list with outcome names (without year)
    @param outcomes_maximize_names: list with outcome names (without year)
    @param outcomes_minimize_names: list with outcome names (without year)
    @param outcomes_maximize_aggregated: list with outcome names (without year)
    @param outcomes_minimize_aggregated: list with outcome names (without year)
    @return:
        epsilons: list with floats
    """
    epsilons = []

    for outcome_name in outcomes_maximize_aggregated:
        epsilon = dict_epsilons[outcome_name]
        epsilons.append(epsilon)

    for outcome_name in outcomes_minimize_aggregated:
        epsilon = dict_epsilons[outcome_name]
        epsilons.append(epsilon)

    for outcome_name in outcomes_maximize_names:
        epsilon = dict_epsilons[outcome_name]
        for _ in years_optimize:
            epsilons.append(epsilon)

    for outcome_name in outcomes_minimize_names:
        epsilon = dict_epsilons[outcome_name]
        for _ in years_optimize:
            epsilons.append(epsilon)

    return epsilons


def get_outcomes_and_epsilons(problem_formulation, years=None, searchover='levers'):
    """
    Returns a list of outcomes and a list of epsilons for the STANDARD workbench.
    @param problem_formulation: ProblemFormulation
    @param years: list of integers
    @param searchover: String
    @return:
            outcomes: list of ScalarOutcomes
            epsilons: list of epsilon values (floats)
    """

    dict_epsilons = {
        'Utility': 0.1,  # 7.6
        'Disutility': 0.1,  # 6.4
        'Lowest income per capita': 0.1,  # 0.516
        'Intratemporal consumption Gini': 0.0001,  # 0.00049
        'Highest damage per capita': 0.01,  # 0.071
        'Intratemporal damage Gini': 0.001,  # 0.038
        'Population below consumption threshold': 5.0,  # 75
        'Distance to consumption threshold': 0.1,  # 0.172
        'Population above damage threshold': 5.0,  # 691
        'Distance to damage threshold': 0.1,  # 0.4
        'Temperature overshoot': 0.1,

        'Damages': 0.1,  # 0.1
        'Industrial Emission': 0.1,  # 1.0
        'Atmospheric Temperature': 0.1,  # 0.55
        'Intertemporal consumption distance': 1.0,  # 140
        'Intertemporal consumption population': 100.0,  # 4200
        'Intertemporal damage distance': 1.0,  # 137
        'Intertemporal damage population': 100.0,  # 19500
        'Intertemporal lowest income p/c': 5.0,  # 52.31
        'Intertemporal highest damage p/c': 0.2,  # 2.328
        'Intertemporal consumption Gini': 0.01,  # 0.01
        'Intertemporal damage Gini': 0.01,  # 0.024
        'Total Aggregated Utility': 100,  # 1575
        'Total Aggregated Disutility': 100,  # 9942
        'Costs': 0.1,  # 0.18
        'Total Output': 1.0,  # 25
        'Total Aggregated Costs': 20,  # 943
    }

    # Relevant years
    if years is None:
        years_optimize = [2105]
    else:
        years_optimize = years
    years_info = []

    # All relevant timeseries outcome variable names
    outcomes_all_names = get_all_outcome_names()

    if problem_formulation == ProblemFormulation.ALL_OBJECTIVES:
        outcomes_maximize_names = [
            'Utility',
            'Lowest income per capita',
        ]
        outcomes_minimize_names = [
            'Disutility',
            'Intratemporal consumption Gini',
            'Intratemporal damage Gini',
            'Highest damage per capita',
            'Distance to consumption threshold',
            'Population below consumption threshold',
            'Distance to damage threshold',
            'Population above damage threshold',
            'Temperature overshoot'
        ]
        outcomes_maximize_aggregated = []
        outcomes_minimize_aggregated = []
        outcomes_info_aggregated = []

        # Need all time steps for running experiments
        years_optimize = []
        years_info = list(range(2005, 2310, 10))

    elif problem_formulation == ProblemFormulation.UTILITARIAN_COSTS:
        outcomes_maximize_names = []
        outcomes_minimize_names = ['Costs', 'Temperature overshoot']
        outcomes_maximize_aggregated = []
        outcomes_minimize_aggregated = []
        outcomes_info_aggregated = []

    elif problem_formulation == ProblemFormulation.UTILITARIAN_AGGREGATED:
        outcomes_maximize_names = ['Utility']
        outcomes_minimize_names = ['Temperature overshoot']
        outcomes_maximize_aggregated = []
        outcomes_minimize_aggregated = []
        outcomes_info_aggregated = []

    elif problem_formulation == ProblemFormulation.UTILITARIAN_DISAGGREGATED:
        outcomes_maximize_names = ['Utility']
        outcomes_minimize_names = ['Disutility', 'Temperature overshoot']
        outcomes_maximize_aggregated = []
        outcomes_minimize_aggregated = []
        outcomes_info_aggregated = []

    elif problem_formulation == ProblemFormulation.SUFFICIENTARIAN_AGGREGATED:
        outcomes_maximize_names = ['Utility']
        outcomes_minimize_names = [
            'Distance to consumption threshold',
            'Population below consumption threshold',
            'Temperature overshoot'
        ]
        outcomes_maximize_aggregated = []
        outcomes_minimize_aggregated = []
        outcomes_info_aggregated = []

    elif problem_formulation == ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED:
        outcomes_maximize_names = ['Utility']
        outcomes_minimize_names = [
            'Disutility',
            'Distance to consumption threshold',
            'Population below consumption threshold',
            'Distance to damage threshold',
            'Population above damage threshold',
            'Temperature overshoot'
        ]
        outcomes_maximize_aggregated = []
        outcomes_minimize_aggregated = []
        outcomes_info_aggregated = []

    elif problem_formulation == ProblemFormulation.PRIORITARIAN_AGGREGATED:
        outcomes_maximize_names = ['Utility', 'Lowest income per capita']
        outcomes_minimize_names = ['Temperature overshoot']
        outcomes_maximize_aggregated = []
        outcomes_minimize_aggregated = []
        outcomes_info_aggregated = []

    elif problem_formulation == ProblemFormulation.PRIORITARIAN_DISAGGREGATED:
        outcomes_maximize_names = ['Utility', 'Lowest income per capita']
        outcomes_minimize_names = ['Disutility', 'Highest damage per capita', 'Temperature overshoot']
        outcomes_maximize_aggregated = []
        outcomes_minimize_aggregated = []
        outcomes_info_aggregated = []

    elif problem_formulation == ProblemFormulation.EGALITARIAN_AGGREGATED:
        outcomes_maximize_names = ['Utility']
        outcomes_minimize_names = ['Intratemporal consumption Gini', 'Temperature overshoot']
        outcomes_maximize_aggregated = []
        outcomes_minimize_aggregated = []
        outcomes_info_aggregated = []

    elif problem_formulation == ProblemFormulation.EGALITARIAN_DISAGGREGATED:
        outcomes_maximize_names = ['Utility']
        outcomes_minimize_names = [
            'Disutility',
            'Intratemporal consumption Gini',
            'Intratemporal damage Gini',
            'Temperature overshoot'
        ]
        outcomes_maximize_aggregated = []
        outcomes_minimize_aggregated = []
        outcomes_info_aggregated = []

    else:
        outcomes_maximize_names = []
        outcomes_minimize_names = []
        outcomes_maximize_aggregated = []
        outcomes_minimize_aggregated = []
        outcomes_info_aggregated = []

    epsilons = _get_epsilons(
        dict_epsilons, years_optimize, outcomes_maximize_names, outcomes_minimize_names,
        outcomes_maximize_aggregated, outcomes_minimize_aggregated
    )

    outcomes = _get_relevant_outcomes(
        outcomes_all_names, outcomes_maximize_names, outcomes_minimize_names, outcomes_maximize_aggregated,
        outcomes_minimize_aggregated, years_optimize, years_info, outcomes_info_aggregated
    )

    # Inverting optimization direction (for scenario search only)
    if searchover == 'uncertainties':
        for o in outcomes:
            if o.kind == o.MINIMIZE:
                o.kind = o.MAXIMIZE
            elif o.kind == o.MAXIMIZE:
                o.kind = o.MINIMIZE

    return outcomes, epsilons


def get_all_outcome_names():
    """
    Return the most important outcome variable names.
    @return:
        outcomes_all_names: list with Strings
    """

    outcomes_all_names = [
        'Utility',
        'Disutility',
        'Intratemporal consumption Gini',
        'Intratemporal damage Gini',
        'Lowest income per capita',
        'Highest damage per capita',
        'Distance to consumption threshold',
        'Population below consumption threshold',
        'Distance to damage threshold',
        'Population above damage threshold',
        'Temperature overshoot'
    ]

    return outcomes_all_names


if __name__ == '__main__':

    with_all_PFs = True

    if with_all_PFs:
        pfs = list(ProblemFormulation)

        for p in pfs:
            results = get_outcomes_and_epsilons(problem_formulation=p)
            outcomes_list, eps = results

            print(f'PF: {p}')
            for out in outcomes_list:
                if out.kind != ScalarOutcome.INFO:
                    print(f'Outcome name: {out.name},\t optimization direction: {out.kind}')
            print()

    else:
        results = get_outcomes_and_epsilons(problem_formulation=ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED)
        outcomes_list, eps = results

        print('Outcomes:')
        for out in outcomes_list:
            print(f'Outcome name: {out.name},\t optimization direction: {out.kind}')

        print('\nEpsilons:')
        for e in eps:
            print(f'Epsilon: {e}')
