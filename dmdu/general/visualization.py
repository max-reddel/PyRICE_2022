"""
This module contains functions to visualize outcomes, hypervolume, etc.
"""
from enum import Enum

import plotly.graph_objects as go
from ema_workbench.util.utilities import load_results
from ema_workbench.analysis import plotting, Density, parcoords
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from dmdu.general.xlm_constants_epsilons import get_lever_names
from dmdu.scenariodiscovery.clustering.silhouette_widths import get_outcomes_reshaped


class Orientation(Enum):
    """
    This enumeration is used to determine whether the figure should be
        - horizontolly longer (for slides) or
        - vertically longer (for report)
    """

    HORIZONTAL = 0,
    VERTICAL = 1


def plot_pathways(outcomes_df, outcome_names, saving=False, file_name=None):
    """
    Plots pathways given an outcome DataFrame and outcomes-names.

    Remark: Currently not super stable. Might break because of length of args.

    @param outcomes_df: DataFrame
    @param outcome_names: list
    @param saving: Booelean
    @param file_name: String: file name for saving
    """

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")

    if len(outcome_names) == 1:
        nrows = 1
        ncols = 1
    elif len(outcome_names) == 2:
        nrows = 2
        ncols = 1
    elif len(outcome_names) == 3:
        nrows = 3
        ncols = 1
    elif len(outcome_names) <= 4:
        nrows = 2
        ncols = 2
    elif len(outcome_names) <= 6:
        nrows = 3
        ncols = 2
    elif len(outcome_names) <= 9:
        nrows = 3
        ncols = 3
    elif len(outcome_names) <= 12:
        nrows = 3
        ncols = 4
    elif len(outcome_names) <= 16:
        nrows = 4
        ncols = 4
    else:
        nrows = 5
        ncols = 5

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(36, 24), tight_layout=True
    )
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.8
    )

    years = list(range(2005, 2310, 10))

    size_df = outcomes_df.shape[0]
    if size_df >= 10000:
        linewidth = 0.5
        alpha = 0.1
    elif size_df >= 100:
        linewidth = 1.0
        alpha = 0.5
    else:
        linewidth = 1.0
        alpha = 1.0

    # Figures
    for i, ax in enumerate(axes.flat):
        if i >= len(outcome_names):
            break
        name = outcome_names[i]
        df = outcomes_df.filter(regex=name, axis=1)  # Filter columns to include "name"
        for idx, row in df.iterrows():

            ax.plot(
                years,
                row.iloc[:],
                linewidth=linewidth,
                alpha=alpha,
                color='forestgreen',
            )

        ax.set_title(name)
        ax.set_xlabel('Time in years')
        ax.set_ylabel(name)

    if len(outcome_names) > 6:
        axes[-1, -1].axis('off')
    plt.show()

    if saving:
        if file_name is None:
            file_name = 'open_exploration_pathways'
        save_own_figure(fig, file_name)


def save_own_figure(fig, file_name):
    """
    Save a figure with given file name in outputimages.
    @param fig: Figure
    @param file_name: string
    """
    visualization_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'outputimages')
    file_name += '.png'
    path = os.path.join(visualization_folder, file_name)
    fig.savefig(path, dpi=200, pad_inches=0.2, bbox_inches='tight')


def plot_kpi_pathways(problem_formulations_dict, outcome_names=None, saving=False, file_name=None):
    """
    Plots pathways given an outcome DataFrame and outcomes-names.

    Remark: Currently not super stable. Might break because of length of args.

    @param problem_formulations_dict: dictionary with
                                      {ProblemFormulation.name: (experiments: DataFrame, outcomes: DataFrame)}
    @param outcome_names: list
    @param saving: Booelean
    @param file_name: String: file name for saving
    """

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")

    nrows = 3
    ncols = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(36, 24), tight_layout=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.8)

    years = list(range(2005, 2310, 10))
    if outcome_names is None:
        outcome_names = [
            'Utility',
            'Total Output',
            'Damages',
            'Atmospheric Temperature',
            'Industrial Emission',
            'Temperature overshoot'
        ]

    # Problem formulations and colors
    color_mapping = {}
    for _, (problem_formulation, color) in enumerate(zip(problem_formulations_dict, sns.color_palette())):
        color_mapping[problem_formulation] = color

    # Figures
    for i, ax in enumerate(axes.flat):

        name = outcome_names[i]

        for problem_formulation, (experiments, outcomes) in problem_formulations_dict.items():

            df = outcomes.filter(regex=name, axis=1)  # Filter columns to include "name"

            for idx, row in df.iterrows():

                ax.plot(
                    years,
                    row.iloc[:],
                    linewidth=1.5,
                    alpha=1.0,
                    color=color_mapping[problem_formulation],
                    label=problem_formulation
                )

        ax.set_title(name)
        ax.set_xlabel("Time in years")
        ax.set_ylabel(name)

    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.suptitle('KPI Pathways', y=1.05)
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 1.0),
        loc='upper center',
        ncol=len(color_mapping)
    )
    plt.show()

    if saving:
        if file_name is None:
            file_name = "open_exploration_pathways"
        save_own_figure(fig, file_name)


def plot_kpi_pathways_with_seeds(
        seeds_dict,
        outcome_names=None,
        problem_formulation='',
        plot_orientation=Orientation.HORIZONTAL,
        saving=False,
        file_name=None
):
    """
    Plots pathways given one problem formulation and several seeds.

    Remark: Currently not super stable. Might break because of length of args.

    @param seeds_dict: dictionary with
                      {seed_idx: (experiments: DataFrame, outcomes: DataFrame)}
    @param outcome_names: list
    @param problem_formulation: String
    @param plot_orientation: Orientation
    @param saving: Booelean
    @param file_name: String: file name for saving
    """

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")

    # Size of plot dimensions
    long = 24
    short = 16

    # Adjusting orientation
    if plot_orientation == Orientation.VERTICAL:
        nrows = 3
        ncols = 2
        fig_size = (long, short)
    else:
        nrows = 2
        ncols = 3
        fig_size = (long, short)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size, tight_layout=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.8)

    years = list(range(2005, 2310, 10))
    if outcome_names is None:
        outcome_names = [
            'Utility',
            'Total Output',
            'Damages',
            'Atmospheric Temperature',
            'Industrial Emission',
            'Temperature overshoot'
        ]

    # Seeds and colors
    color_mapping = {}
    unique_seeds = list(set([k for k in seeds_dict.keys()]))
    for _, (seed, color) in enumerate(zip(unique_seeds, sns.color_palette())):
        color_mapping[seed] = color

    axes_font_size = 20

    # Figures
    for i, ax in enumerate(axes.flat):

        name = outcome_names[i]

        for seed_idx, (experiments, outcomes) in seeds_dict.items():

            df = outcomes.filter(regex=name, axis=1)  # Filter columns to include "name"

            for idx, row in df.iterrows():

                ax.plot(
                    years,
                    row.iloc[:],
                    linewidth=1.5,
                    alpha=1.0,
                    color=color_mapping[seed_idx],
                    label=f'seed {seed_idx}'
                )

        ax.set_title(name)
        ax.set_xlabel('Time in years', fontsize=axes_font_size)
        ax.set_ylabel(name, fontsize=axes_font_size)

    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.suptitle('KPI Pathways for ' + problem_formulation, y=1.05)
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 1.0),
        loc='upper center',
        ncol=len(color_mapping)
    )
    plt.show()

    if saving:
        if file_name is None:
            file_name = "open_exploration_pathways"
        save_own_figure(fig, file_name)


def plot_one_pathway(experiments, outcomes, outcome_name, saving=False, file_name=None):
    """
    Plot the pathways of a specific objective grouped by their clusters.
    @param experiments: DataFrame
    @param outcomes: DataFrame
    @param outcome_name: String
    @param saving: Boolean
    @param file_name: String
    """

    reshaphed_outcomes = get_outcomes_reshaped(
        outcomes_df=outcomes, objective_names=[outcome_name]
    )

    fig, axes = plotting.lines(
        experiments=experiments,
        outcomes=reshaphed_outcomes,
        outcomes_to_show=outcome_name,
        # group_by='clusters',
        density=Density.BOXPLOT,
    )
    fig.set_size_inches(15, 8)
    plt.show()

    if saving:
        if file_name is None:
            file_name = 'open_exploration_pathways'
        save_own_figure(fig, file_name)


def parallel_axis_plot(experiments, outcomes, limits, axis_width=120, font_size=14):
    """
    takes in outcomes, processes it up to the finished interactive parallel axis plot.

    @param experiments :   dataframe
                    all experimental outcomes
    @param outcomes :      dataframe
                    all outcome outcomes
    @param limits : dataframe
                    This should be an outcome dataframe. It can be the same as the 'outcome' parameter.
                    In this case, the plot will be shown as usual. If another outcomes dataframe is used, it will
                    take the limits of this other outcomes dataframe instead of the usual outcome limits.
    @param axis_width :    float
                    indicates how much horizontal space there will be between the axes.
    @param font_size :     int
                    font size
    @return
        fig: Figure
    """

    minimize_list = [
        "Disutility",
        "Intratemporal consumption Gini",
        "Intratemporal damage Gini",
        "Highest damage per capita",
        "Distance to consumption threshold",
        "Population below consumption threshold",
        "Distance to damage threshold",
        "Population above damage threshold",
        "Temperature overshoot",
    ]
    minimize_list = [x + " 2105" for x in minimize_list]

    dimensions_list = []

    max_length_objective_name = 0

    # Fill list with info for each objective:
    for obj_name in outcomes.columns.tolist():

        if len(obj_name) > max_length_objective_name:
            max_length_objective_name = len(obj_name)

        # Find lower & upper bound for each objective
        lower_bound = min(limits.loc[:, obj_name])
        upper_bound = max(limits.loc[:, obj_name])

        # Adjust axis orientation. If kind.MINIMIZE: lowest value at top of parallel axis plot.
        if obj_name in minimize_list:
            range_boundaries = [upper_bound, lower_bound]
        else:
            range_boundaries = [lower_bound, upper_bound]

        # Create dict for objectives (i.e., dimensions)
        objective = dict(
            range=range_boundaries, label=obj_name, values=outcomes.loc[:, obj_name]
        )

        # Add dict
        dimensions_list.append(objective)

    fig = go.Figure(
        data=go.Parcoords(
            # Policy lines
            # line=dict(color=experiments['policy'], showscale=True),
            # line=dict(color=((34, 139, 34), alpha)),
            # line=dict(color='forestgreen', showscale=True),
            # Outcomes
            dimensions=list(dimensions_list),
            # Formatting
            labelangle=-90,
            labelside="bottom",
        )
    )

    nr_of_axes = len(outcomes.columns)

    width = axis_width * nr_of_axes
    height = 800

    # Layout changes
    fig.update_layout(
        font=dict(size=font_size),
        plot_bgcolor="white",
        paper_bgcolor="white",
        overwrite=True,
        autosize=False,
        width=width,
        height=height,
        margin=dict(l=50, r=50, b=8 * max_length_objective_name, t=50, pad=4),
    )

    fig.show()


def get_y_labels_dict():
    """
    Returns a dictionary that provides y_label information for a given objective.
    @return
        info_dict: dictionary: {objective_name (string): name + units (string)}
    """

    info_dict = {
        'Utility': 'Welfare of Utility',
        'Disutility': 'Welfare of Disutility',
        'Lowest income per capita': 'Lowest income per capita ($1000)',
        'Intratemporal consumption Gini': 'Intratemporal consumption Gini',
        'Highest damage per capita': 'Highest damage per capita',
        'Intratemporal damage Gini': 'Intratemporal damage Gini',
        'Population below consumption threshold': 'Population below consumption threshold (million)',
        'Distance to consumption threshold': 'Distance to consumption threshold',
        'Population above damage threshold': 'Population above damage threshold (million)',
        'Distance to damage threshold': 'Distance to damage threshold (trillion $)',
        'Temperature overshoot': '# of temperature overshoot time steps',
        'Damages': 'Economic damages (trillion $)',
        'Industrial Emission': 'Global emissions (GTon CO2)',
        'Atmospheric Temperature': 'Increase in ATM temperature (Celsius)',
    }

    return info_dict


def plot_policies_per_problem_formulation(problem_formulations_dict, saving=False, file_name=None):
    """
    Plots a parallel axis plots with all levers as axes and color-coded by problem formulation.
    @param problem_formulations_dict: dictionary with
    @param saving: Boolean: whether to save the figure
    @param file_name: String
                                      {ProblemFormulation: (experiments: DataFrame, outcomes: DataFrame)}
    """
    lever_names = get_lever_names()
    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    sns.set(rc={'figure.figsize': (12, 8)})

    # Problem formulations and colors
    color_mapping = {}
    for _, (problem_formulation, color) in enumerate(zip(problem_formulations_dict, sns.color_palette())):
        color_mapping[problem_formulation] = color

    axes = None

    all_levers = None

    for problem_formulation, (experiments, outcomes) in problem_formulations_dict.items():

        levers = experiments.loc[:, lever_names]

        if all_levers is None:
            all_levers = levers
            limits = parcoords.get_limits(levers)
        else:
            all_levers = pd.concat([all_levers, levers])
            limits = parcoords.get_limits(all_levers)

        if axes is None:
            axes = parcoords.ParallelAxes(limits)
        axes.plot(levers, color=color_mapping[problem_formulation], label=problem_formulation)

    axes.legend()
    plt.show()

    if saving:
        if file_name is None:
            file_name = 'optimal_policies'
        save_own_figure(axes.fig, file_name)


def plot_robustness(robustness_dataframe, saving=False, file_name=None):
    """
    Plot the robustness of some KPIs on a parallel axis plot.
    @param robustness_dataframe: DataFrame
    @param saving: Boolean: whether to save the figure
    @param file_name: String
    """

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    sns.set(rc={'figure.figsize': (12, 8)})

    # Colors
    unique_problem_formulations = robustness_dataframe.loc[:, 'Problem Formulation'].unique()
    color_mapping = {}
    for _, (problem_formulation, color) in enumerate(zip(unique_problem_formulations, sns.color_palette())):
        color_mapping[problem_formulation] = color

    axes = None

    all_policies = None

    for problem_formulation in unique_problem_formulations:

        relevant_policies = robustness_dataframe[robustness_dataframe['Problem Formulation'] == problem_formulation]
        relevant_policies = relevant_policies.drop(columns=['Problem Formulation', 'Policy'])

        # Adjust limits
        if all_policies is None:
            all_policies = relevant_policies
            limits = parcoords.get_limits(relevant_policies)
        else:
            all_policies = pd.concat([all_policies, relevant_policies])
            limits = parcoords.get_limits(relevant_policies)

        if axes is None:
            axes = parcoords.ParallelAxes(limits)
        axes.plot(relevant_policies, color=color_mapping[problem_formulation], label=problem_formulation)

    axes.legend()
    plt.show()

    if saving:
        if file_name is None:
            file_name = 'robustness'
        save_own_figure(axes.fig, file_name)


if __name__ == "__main__":
    directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        'exploration', 'data'
    )
    results = load_results(file_name=os.path.join(directory, "results_open_exploration_10"))
    experiments, outcomes = results
    outcomes = pd.DataFrame(outcomes)

    outcome_names = [
        "Distance to consumption threshold",
        "Distance to damage threshold",
        "Population below consumption threshold",
        "Population above damage threshold",
        "Utility",
        "Disutility",
    ]

    plot_pathways(outcomes, outcome_names, saving=True, file_name="results_testing")
