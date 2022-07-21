"""
This module contains functions to compute several robustness metrics.
"""


import numpy as np
import pandas as pd


def get_robustness_dataframe(experiments, outcomes, problem_formulation):
    """
    Returns a dataframe that contains the robustness values for each outcome and each policy.
    Each outcome variable has an individual robustness metric assigned to it.

    @param experiments: DataFrame: contains all experimental data
    @param outcomes: DataFrame: contains all outcome values for all scenarios and policies
    @param problem_formulation: ProblemFormulation

    @return robust: DataFrame: contains all robustness values for each outcome and each policy
    """

    robust = {}

    for outcome_name in list(outcomes.columns):

        outcomes_dict = outcomes.to_dict('series')

        # Calculate Starr's domain criterion
        if outcome_name in ['Temperature overshoot 2105', 'Atmospheric Temperature 2105', 'Industrial Emission 2105']:
            robustness_value = starr(experiments, outcomes_dict, outcome_name, threshold=4, higher_is_better=False)
            robust[f'starr\natmospheric\ntemperature'] = robustness_value

        # Calculate maximax
        elif outcome_name in ['Total Output 2105']:
            robustness_value = max_regret(experiments, outcomes_dict, outcome_name)
            robust[f'minimax\n{get_name_without_year(outcome_name)}'] = robustness_value

        # Calculate Hurwicz criterion
        elif outcome_name in ['Utility 2105']:
            robustness_value = hurwicz_criterion(experiments, outcomes_dict, outcome_name)
            robust[f'hurwicz\n{"welfare"}'] = robustness_value

        # Calculate 90th percentile minimax regret
        elif outcome_name in ['Disutility 2105']:
            robustness_value = percentile_90_minimax_regret(experiments, outcomes_dict, outcome_name)
            robust[f'90minimax\n{"welfare loss"}'] = robustness_value

        # # Calculate maximin
        # elif outcome_name in []:
        #     robustness_value = maximin(experiments, outcomes_dict, outcome_name)
        #     robust[f'maximin({outcome_name})'] = robustness_value
        #
        # # Calculate minimax regret
        # if outcome_name in []:
        #     robustness_value = max_regret(experiments, outcomes_dict, outcome_name)
        #     robust[f'minimax({outcome_name})'] = robustness_value

        else:
            print(f'Something went wrong! An outcome name does not fit the outcomes.'
                  f'\nThe outcome in question = {outcome_name}')

    # Get the names of all outcome variables
    col_names = list(robust.keys())

    # Get all policy IDs
    policy_ids = []
    v = list(robust.values())[0]
    for policy_id in v.keys():
        policy_ids.append(policy_id)

    # Get the column values for each outcome
    cols = []
    for o_key in robust.values():
        col = []
        for v in o_key.values():
            col.append(v)
        cols.append(col)

    # Assign the column values to the corresponding robustness value of an outcome
    for idx, col in enumerate(cols):
        outcome_name = col_names[idx]
        robust[outcome_name] = cols[idx]

    # Parse from dictionary to dataframe
    robust = pd.DataFrame(robust)
    robust['Policy'] = policy_ids

    # Switch position of 'Policy' column from last to first position
    col_names = robust.columns.tolist()
    col_names = col_names[-1:] + col_names[:-1]
    robust = robust[col_names]

    robust['Problem Formulation'] = problem_formulation

    return robust


def get_name_without_year(name):
    """
    Return objective name without year-suffix.
    @param name: string
    @return:
        shortened: string
    """
    words = name.split(' ')

    shortened = ''
    for word in words[:-1]:
        shortened += word + ' '

    shortened = shortened[:-1]

    return shortened


def restructure_data(experiments, outcomes, outcome_name):
    """
    Filters and restructures the provided data to only contain what is needed.
    I.e., a dataframe for the passed outcome.

    Policies and Scenarios have to be properly numerated, i.e., all combinations of policy-scenario can only occur once.

    @param experiments: DataFrame: experiments from the model run
    @param outcomes: dictionary: outcomes from the model run
    @param outcome_name: str: outcome name

    @return data: DataFrame, for 1 outcome
                        rows : scenarios
                        cols : policies
    """
    # setup columns
    scenario_column = experiments['scenario']
    policy_column = experiments['policy']
    outcome_column = outcomes[outcome_name]

    # dict from columns
    data_dict = {'scenario': scenario_column,
                 'policy': policy_column,
                 outcome_name: outcome_column}
    # df from dict
    data = pd.DataFrame(data_dict)

    # restructure dataframe
    data = data.pivot(index='scenario', columns='policy')

    return data


def max_regret(experiments, outcomes, outcome_name, maximize=False):
    """
    Calculates max_regret for a specific outcome and all policies (for Savage's minimax regret).

    @param experiments: DataFrame: experiments from the model run
    @param outcomes: dictionary: of outcomes from the model run
    @param outcome_name: str: outcome name
    @param maximize: Boolean: whether outcome should be maximized

    @return max_regret: dictionary with
                    key:   int, policy_id
                    value: float, max_regret for that policy
    """

    # Create new dataframe with needed information
    data = restructure_data(experiments, outcomes, outcome_name)

    # Calculate regret values
    if maximize:
        regrets = (np.max(data.to_numpy(), axis=1)[:, np.newaxis] - data).abs()
    else:
        regrets = (np.min(data.to_numpy(), axis=1)[:, np.newaxis] - data).abs()

    # series
    max_regrets = regrets.max(axis=0)

    # polish dict (to not have tuple-key with 'outcome' included)
    max_regrets_dict = {}
    for policy, max_regret in max_regrets.iteritems():
        max_regrets_dict[policy[1]] = max_regret

    return max_regrets_dict


def starr(experiments, outcomes, outcome_name, threshold, higher_is_better=True):
    f"""
    Calculates for each policy the fraction of scenarios which meet a satisfactory outcome level
    for a specific outcome variable.
    I.e., 'Satisficing Type I' after Hadka et al., 2015, as an approximation of Starr's Domain Criterion.

    @param experiments: DataFrame: experiments from the model run
    @param outcomes: dictionary: outcomes from the model run
    @param outcome_name: str: outcome name
    @param threshold: float: satisfying threshold
    @param higher_is_better: Boolean: whether above the threshold is satisfying (rather than below the threshold)

    @return starr: dictionary with
                        key:   int, policy_id
                        value: float, fraction of scenarios in which policy was satisfying.
    """

    # Create new dataframe with needed information
    data = restructure_data(experiments, outcomes, outcome_name)

    # Calculate bool values
    fractions = {}
    # bools = pd.DataFrame(index=range(len(data)),columns=range(len(data.columns))) # empty dataframe

    for policy_id in data[outcome_name].columns:
        # print(data[outcome_name][policy_id])
        col = data[outcome_name][policy_id]

        if higher_is_better:
            satisfying = [x for x in col if x >= threshold]
        else:
            satisfying = [x for x in col if x < threshold]

        fraction = len(satisfying) / len(col)

        # saving fraction for each policy
        fractions[policy_id] = fraction

    return fractions


def hurwicz_criterion(experiments, outcomes, outcome_name, alpha=0.5):
    """
    Calculates Pessimism-Optimism Index Criterion of Hurwicz for a specific outcome and each policy.

    @param experiments: DataFrame: experiments from the model run
    @param outcomes: dictionary: outcomes from the model run
    @param outcome_name: str: outcome name
    @param alpha: float: actor's degree of optimism

    @return hurwicz_metric: dictionary with
                    key:   int, policy_id
                    value: float, hurwicz_metric criterion
    """

    # Create new dataframe with needed information
    data = restructure_data(experiments, outcomes, outcome_name)

    hurwicz_metric = {}

    for policy_id in data[outcome_name].columns:
        outcomes_for_a_policy = data[outcome_name][policy_id]

        # Hurwicz Equation
        largest = max(outcomes_for_a_policy)
        smallest = min(outcomes_for_a_policy)
        h = alpha * largest + (1 - alpha) * smallest

        # Adding metric to output dictionary
        hurwicz_metric[policy_id] = h

    return hurwicz_metric


def maximin(experiments, outcomes, outcome_name):
    """
    Calculates maximin robustness for a specific outcome and each policy.

    @param experiments: DataFrame: experiments from the model run
    @param outcomes: dictionary: outcomes from the model run
    @param outcome_name: str: outcome name

    @return maximin_dict: dictionary with
                    key:   int, policy_id
                    value: float, maximin value
    """

    # Create new dataframe with needed information
    data = restructure_data(experiments, outcomes, outcome_name)

    maximin_dict = {}

    for policy_id in data[outcome_name].columns:
        outcomes_for_a_policy = data[outcome_name][policy_id]

        minimum = min(outcomes_for_a_policy)
        maximin_dict[policy_id] = minimum

    return maximin_dict


def percentile_90_minimax_regret(experiments, outcomes, outcome_name, maximize=False):
    """
    Calculates 90th percentile minimax regret for a specific outcome and all policies.

    @param experiments: DataFrame: experiments from the model run
    @param outcomes: dictionary: outcomes from the model run
    @param outcome_name: str: outcome name
    @param maximize: Boolean: whether outcome should be maximized

    @return minimax_regret: dictionary with
                    key:   int, policy_id
                    value: float, max_regret for that policy
    """

    # Create new dataframe with needed information
    data = restructure_data(experiments, outcomes, outcome_name)

    # Calculate regret values
    if maximize:
        regrets = (np.percentile(data, 90) - data).abs()
    else:
        regrets = (np.percentile(data, 10) - data).abs()

    # series
    max_regrets = regrets.max(axis=0)

    # polish dict (to not have tuple-key with 'outcome' included)
    minimax_regret = {}
    for policy, max_regrets in max_regrets.iteritems():
        minimax_regret[policy[1]] = max_regrets

    return minimax_regret
