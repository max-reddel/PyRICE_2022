"""
This module contains functions to compute outcomes and epsilon values for the optimization process.
"""

from ema_workbench import ScalarOutcome
from model.enumerations import WelfareFunction


def get_outcomes_for_years(outcome_name, years_list, direction):
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


def get_outcomes_to_optimize(outcomes_maximize_names, outcomes_minimize_names, years_optimize):
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
        outcomes_optimize += get_outcomes_for_years(outcome_name, years_optimize, ScalarOutcome.MAXIMIZE)

    for outcome_name in outcomes_minimize_names:
        outcomes_optimize += get_outcomes_for_years(outcome_name, years_optimize, ScalarOutcome.MINIMIZE)

    return outcomes_optimize


def get_outcomes_to_info(outcomes_info_names, outcomes_optimize_names, years_optimize, years_info):
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
        outcomes_info += get_outcomes_for_years(outcome_name, years_optimize + years_info, ScalarOutcome.INFO)

    # Add remaining variables
    for outcome_name in outcomes_optimize_names:
        outcomes_info += get_outcomes_for_years(outcome_name, years_info, ScalarOutcome.INFO)

    return outcomes_info


def get_relevant_outcomes(outcomes_all_names, outcomes_maximize_names, outcomes_minimize_names,
                          outcomes_maximize_aggregated, outcomes_minimize_aggregated, years_optimize, years_info,
                          outcomes_info_aggregated):
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

    outcomes_optimize += get_outcomes_to_optimize(outcomes_maximize_names, outcomes_minimize_names, years_optimize)

    # Outcomes that should only be displayed (i.e., not optimized)
    outcomes_info_names = list(
        set(outcomes_all_names)
        - set(outcomes_maximize_names)
        - set(outcomes_minimize_names)
    )

    outcomes_info = []
    for outcome in outcomes_info_aggregated:
        outcomes_info.append((ScalarOutcome(outcome, ScalarOutcome.INFO)))

    outcomes_info += get_outcomes_to_info(
        outcomes_info_names,
        outcomes_minimize_names + outcomes_maximize_names,
        years_optimize,
        years_info
    )

    # Put them all together
    outcomes = outcomes_optimize + outcomes_info

    return outcomes


def get_epsilons(
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


def get_outcomes_and_epsilons(welfare_function=WelfareFunction.UTILITARIAN, aggregation=True, years=None):
    """
    Returns a list of outcomes and a list of epsilons for the STANDARD workbench.
    @param welfare_function: WelfareFunction
    @param aggregation: Boolean
    @param years: list of integers
    @return:
            outcomes: list of ScalarOutcomes
            epsilons: list of epsilon values (floats)
    """

    dict_epsilons = {
        'Total Aggregated Utility': 50,
        'Utility': 5,
        'Total Aggregated Disutility': 50,
        'Disutility': 5,
        'Lowest income per capita': 0.02,
        'Intratemporal consumption GINI': 0.001,
        'Total Output': 1.0,
        'Atmospheric Temperature': 0.1,
        'Highest damage per capita': 0.01,
        'Intratemporal damage GINI': 0.01,
        'Damages': 0.4,
        'Industrial Emission': 0.1,
        'Population below consumption threshold': 0.01,
        'Distance to consumption threshold': 0.000001,
        'Population above damage threshold': 0.01,
        'Distance to damage threshold': 0.000001,
        'Intertemporal consumption distance': 0.00001,
        'Intertemporal consumption population': 0.1,
        'Intertemporal damage distance': 0.00001,
        'Intertemporal damage population': 0.1,
        'Intertemporal lowest income p/c': 0.2,
        'Intertemporal highest damage p/c': 0.1,
        'Intertemporal consumption GINI': 0.001,
        'Intertemporal damage GINI': 0.01
    }

    # Relevant years
    if years is None:
        years_optimize = [2035, 2055, 2075]
    else:
        years_optimize = years
    years_info = [2105, 2205, 2305]

    # All relevant timeseries outcome variable names
    outcomes_all_names = [
        'Damages',
        'Utility',
        'Disutility',
        'Intratemporal consumption GINI',
        'Intratemporal damage GINI',
        'Lowest income per capita',
        'Highest damage per capita',
        'Distance to consumption threshold',
        'Population below consumption threshold',
        'Distance to damage threshold',
        'Population above damage threshold',
        'Atmospheric Temperature',
        'Industrial Emission',
        'Total Output',
    ]

    if welfare_function == WelfareFunction.UTILITARIAN:

        if aggregation:
            outcomes_maximize_names = ['Utility']
            outcomes_minimize_names = []
            outcomes_maximize_aggregated = ['Total Aggregated Utility']
            outcomes_minimize_aggregated = []
            outcomes_info_aggregated = ['Total Aggregated Disutility']
        else:
            outcomes_maximize_names = ['Utility']
            outcomes_minimize_names = ['Disutility']
            outcomes_maximize_aggregated = ['Total Aggregated Utility']
            outcomes_minimize_aggregated = ['Total Aggregated Disutility']
            outcomes_info_aggregated = []

    elif welfare_function == WelfareFunction.SUFFICIENTARIAN:

        if aggregation:
            outcomes_maximize_names = []
            outcomes_minimize_names = ['Distance to consumption threshold', 'Population below consumption threshold']
            outcomes_maximize_aggregated = ['Total Aggregated Utility']
            outcomes_minimize_aggregated = [
                'Intertemporal consumption distance',
                'Intertemporal consumption population'
            ]
            outcomes_info_aggregated = ['Total Aggregated Disutility']
        else:
            outcomes_maximize_names = []
            outcomes_minimize_names = [
                'Distance to consumption threshold',
                'Population below consumption threshold',
                'Distance to damage threshold',
                'Population above damage threshold'
            ]
            outcomes_maximize_aggregated = ['Total Aggregated Utility']
            outcomes_minimize_aggregated = [
                'Total Aggregated Disutility',
                'Intertemporal consumption distance',
                'Intertemporal consumption population',
                'Intertemporal damage distance',
                'Intertemporal damage population'
            ]
            outcomes_info_aggregated = []

    elif welfare_function == WelfareFunction.PRIORITARIAN:

        if aggregation:
            outcomes_maximize_names = ['Lowest income per capita']
            outcomes_minimize_names = []
            outcomes_maximize_aggregated = ['Intertemporal lowest income p/c']
            outcomes_minimize_aggregated = []
            outcomes_info_aggregated = ['Total Aggregated Utility', 'Total Aggregated Disutility']
        else:
            outcomes_maximize_names = ['Lowest income per capita']
            outcomes_minimize_names = ['Highest damage per capita']
            outcomes_maximize_aggregated = ['Intertemporal lowest income p/c']
            outcomes_minimize_aggregated = ['Intertemporal highest damage p/c']
            outcomes_info_aggregated = ['Total Aggregated Utility', 'Total Aggregated Disutility']

    elif welfare_function == WelfareFunction.EGALITARIAN:

        if aggregation:
            outcomes_maximize_names = []
            outcomes_minimize_names = ['Intratemporal consumption GINI']

            outcomes_maximize_aggregated = ['Total Aggregated Utility']
            outcomes_minimize_aggregated = ['Intertemporal consumption GINI']
            outcomes_info_aggregated = ['Total Aggregated Disutility']
        else:
            outcomes_maximize_names = []
            outcomes_minimize_names = ['Intratemporal consumption GINI', 'Intratemporal damage GINI']

            outcomes_maximize_aggregated = ['Total Aggregated Utility']
            outcomes_minimize_aggregated = [
                'Total Aggregated Disutility',
                'Intertemporal consumption GINI',
                'Intertemporal damage GINI'
            ]
            outcomes_info_aggregated = []

    else:
        outcomes_maximize_names = []
        outcomes_minimize_names = []
        outcomes_maximize_aggregated = []
        outcomes_minimize_aggregated = []
        outcomes_info_aggregated = []

    epsilons = get_epsilons(
        dict_epsilons, years_optimize, outcomes_maximize_names, outcomes_minimize_names,
        outcomes_maximize_aggregated, outcomes_minimize_aggregated
    )

    outcomes = get_relevant_outcomes(
        outcomes_all_names, outcomes_maximize_names, outcomes_minimize_names, outcomes_maximize_aggregated,
        outcomes_minimize_aggregated, years_optimize, years_info, outcomes_info_aggregated
    )

    return outcomes, epsilons


if __name__ == '__main__':

    swf = list(WelfareFunction)
    aggregations = (True, False)

    for s in swf:
        for a in aggregations:
            results = get_outcomes_and_epsilons(welfare_function=s, aggregation=a)
            outcomes_list, eps = results

            print(f'PF: {s}, aggregation: {a}')
            for out in outcomes_list:
                if out.kind != ScalarOutcome.INFO:
                    print(f'Outcome name: {out.name},\t optimization direction: {out.kind}')
            print()

    # results = get_outcomes_and_epsilons(welfare_function=WelfareFunction.EGALITARIAN, aggregation=True)
    # outcomes_list, eps = results
    #
    # print('Outcomes:')
    # for out in outcomes_list:
    #     print(f'Outcome name: {out.name},\t optimization direction: {out.kind}')
    #
    # print('\nEpsilons:')
    # for e in eps:
    #     print(f'Epsilon: {e}')
