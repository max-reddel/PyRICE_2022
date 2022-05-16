"""
This module contains functions to visualize outcomes, hypervolume, etc.
"""

import plotly.graph_objects as go
from ema_workbench.util.utilities import load_results
from ema_workbench.analysis import plotting, Density
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from optimization.scenariodiscovery.clustering.silhouette_widths import (
    get_outcomes_reshaped,
)


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
        nrows=nrows, ncols=ncols, figsize=(28, 24), tight_layout=True
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
            # print(f'len(years): {len(years)}')
            # print(f'len(row): {len(row.iloc[:])}')
            ax.plot(
                years,
                row.iloc[:],
                linewidth=linewidth,
                alpha=alpha,
                color="forestgreen",
            )

        ax.set_title(name)
        ax.set_xlabel("Time in years")
        ax.set_ylabel(name)

    axes[-1, -1].axis("off")
    plt.show()

    if saving:

        visualization_folder = (
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            + "/outputimages/"
        )
        if file_name is None:
            file_name = "open_exploration_pathways"
        file_name += ".png"
        fig.savefig(visualization_folder + file_name, dpi=200, pad_inches=0.2)


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

        visualization_folder = (
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            + "/outputimages/"
        )
        if file_name is None:
            file_name = "open_exploration_pathways"
        file_name += ".png"
        fig.savefig(visualization_folder + file_name, dpi=200, pad_inches=0.2)


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
            # line_color = 'blue',
            # line=dict(color=experiments['policy'], showscale=True),
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
    # return fig


if __name__ == "__main__":
    directory = (
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        + "/exploration/outcomes/"
    )
    results = load_results(file_name=directory + "results_open_exploration_10")
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
