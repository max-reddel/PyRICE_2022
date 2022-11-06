"""
This module contains functions to visualize outcomes, hypervolume, etc.
"""
from enum import Enum
import types

import matplotlib
import numpy as np
import plotly.graph_objects as go
from ema_workbench.analysis import plotting, Density, parcoords
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os
import matplotlib.cm as cm
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator

from dmdu.general.xlm_constants_epsilons import get_lever_names
from dmdu.scenarioselection.clustering.silhouette_widths import get_outcomes_reshaped
from model.enumerations import ProblemFormulation


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
        nrows=nrows, ncols=ncols, figsize=(30, 20), tight_layout=True
    )
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.8
    )

    y_labels = get_y_labels_dict()
    title_labels = get_flat_y_labels_dict()
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

        ax.set_title(title_labels[name])
        ax.set_xlabel('time in years')
        ax.set_ylabel(y_labels[name])
        ax.yaxis.set_major_locator(MaxNLocator(5))

    if len(outcome_names) > 6:
        axes[-1, -1].axis('off')
    plt.show()

    if saving:
        if file_name is None:
            file_name = 'open_exploration_pathways'
        sub_folder = 'exploration'
        save_own_figure(fig, file_name, sub_folder)


def save_own_figure(fig, file_name, sub_folder, resolution=200):
    """
    Save a figure with given file name in outputimages.
    @param fig: Figure
    @param file_name: string
    @param sub_folder: string
    @param resolution: int
    """
    visualization_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'outputimages')
    file_name += '.png'
    path = os.path.join(visualization_folder, sub_folder, file_name)
    fig.savefig(path, dpi=resolution, pad_inches=0.2, bbox_inches='tight')


def plot_kpi_pathways(problem_formulations_dict, outcome_names=None, saving=False, file_name=None):
    """
    Plots pathways given an outcome DataFrame and outcomes-names.

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
        sub_folder = 'optimalpolicies'
        save_own_figure(fig, file_name, sub_folder)


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
        sub_folder = 'seeds'
        save_own_figure(fig, file_name, sub_folder)


def plot_simple_kpi_pathways_with_seeds(
        seeds_dict,
        outcome_names=None,
        problem_formulation='',
        plot_orientation=Orientation.HORIZONTAL,
        saving=False,
        file_name=None
):
    """
    Plots pathways given one problem formulation and several seeds. This function considers the results from the
    reference scenarios.

    Remark: Currently not super stable. Might break because of length of args.

    @param seeds_dict: dictionary with
                      {seed_idx: (outcomes: DataFrame)}
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
    for _, (seed, color) in enumerate(zip(unique_seeds, sns.color_palette('pastel'))):
        color_mapping[seed] = color

    axes_font_size = 20

    # Figures
    for i, ax in enumerate(axes.flat):

        name = outcome_names[i]

        # for seed_idx, outcomes in reversed(seeds_dict.items()):
        for seed_idx, outcomes in seeds_dict.items():

            df = outcomes.filter(regex=name, axis=1)  # Filter columns to include "name"

            for idx, row in df.iterrows():

                ax.plot(
                    years,
                    row.iloc[:],
                    linewidth=1.0,
                    alpha=0.2,
                    linestyle='-',
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
        sub_folder = 'seeds'
        save_own_figure(fig, file_name, sub_folder)


def plot_conference_pathways(
        problem_formulations_dict,
        shaded_outcome_name=None,
        outcome_names=None,
        uni_color=False,
        saving=False,
        file_name=None
):
    """
    Plots pathways given one problem formulation and several seeds. This function considers the results from the
    reference scenarios.

    Remark: Currently not super stable. Might break because of length of args.

    @param problem_formulations_dict: DataFrame
    @param shaded_outcome_name: String: which variable should be related to color
    @param outcome_names: list
    @param uni_color: Boolean: using only one color instead of a color palette
    @param saving: Booelean
    @param file_name: String: file name for saving
    """

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 12), tight_layout=False, sharey='row')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)

    years = list(range(2005, 2310, 10))
    if outcome_names is None:
        outcome_names = [
            'Damages',
            'Atmospheric Temperature'
        ]

    y_labels = get_y_labels_dict()

    axes_font_size = 20

    # Collecting all outcomes
    all_outcomes = None
    for _, outcomes in problem_formulations_dict.items():
        outcomes.index = list(range(len(outcomes)))
        if all_outcomes is None:
            all_outcomes = outcomes
        else:
            all_outcomes = pd.concat([all_outcomes, outcomes])

    # Getting minimum and maximum for specific hue-related variable
    hue_column = all_outcomes.loc[:, shaded_outcome_name]
    color_variable_min = min(hue_column)
    color_variable_max = max(hue_column)

    # Setting up a color mapper
    norm = mpl.colors.Normalize(vmin=color_variable_min, vmax=color_variable_max, clip=True)
    # palette = 'rocket_r'
    palette = "crest"
    cmap = sns.color_palette(palette, as_cmap=True)

    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper_list = [mapper.to_rgba(x) for x in hue_column]

    # Splitting mapper into separate problem-formulations-specific lists
    lengths_of_outcomes = {pf: len(outcomes) for pf, outcomes in problem_formulations_dict.items()}
    mapper_dict = {}  # for use of color mapping
    checked_lengths = 0
    for pf, length in lengths_of_outcomes.items():
        mapper_dict[pf] = mapper_list[checked_lengths: checked_lengths+length]
        checked_lengths += length

    delft_blue = (0/256, 166/256, 214/256)  # TU Deflt Blue
    delft_green = (0/256, 155/256, 119/256)  # TU Deflt Green

    # Actual plotting
    for pf_idx, (problem_formulation, outcomes) in enumerate(problem_formulations_dict.items()):
        for name_idx, name in enumerate(outcome_names):

            outcomes.index = list(range(len(outcomes)))
            df = outcomes.filter(regex=name, axis=1)  # Filter columns to include "name"

            for row_idx, row in df.iterrows():

                if uni_color:
                    if pf_idx == 0:
                        color = to_rgb(delft_blue)
                    else:
                        color = to_rgb(delft_green)
                else:
                    color = mapper_dict[problem_formulation][row_idx]

                axes[name_idx, pf_idx].plot(
                    years,
                    row.iloc[:],
                    linewidth=1.0,
                    alpha=1.0,
                    linestyle='-',
                    color=color,
                )

                # Annotations
                if name_idx == 0:
                    short_name = (problem_formulation.lower()).split('_')[0]
                    axes[name_idx, pf_idx].set_title(short_name, fontsize=30, pad=20)
                axes[name_idx, pf_idx].set_xlabel('time in years', fontsize=axes_font_size)
                y_label = y_labels[name]
                axes[name_idx, pf_idx].set_ylabel(y_label, fontsize=axes_font_size)

    # Show labels although sharing y-axis
    for ax in axes.flatten():
        # ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)

    # Color bar
    if not uni_color:
        cbar = fig.colorbar(mapper, ax=axes, shrink=0.7)

        if shaded_outcome_name == 'Total Output 2105':
            bar_label = 'GWP in 2105 (trillion $)'
        elif shaded_outcome_name == 'Utility 2105':
            bar_label = 'Welfare'
        elif shaded_outcome_name == 'Temperature overshoot 2105':
            bar_label = 'Number of years with a 2°C temperature overshoot'
        else:
            bar_label = shaded_outcome_name
        cbar.set_label(bar_label, labelpad=15)

    plt.show()

    if saving:
        if file_name is None:
            file_name = "pathways"
        sub_folder = 'iEMSs'
        save_own_figure(fig, file_name, sub_folder)


def plot_kpi_pathways_with_color(
        problem_formulations_dict,
        shaded_outcome_name=None,
        outcome_names=None,
        uni_color=False,
        saving=False,
        file_name=None
):
    """
    Main functino for KPIs
    Plots pathways given one problem formulation. This function considers the results from the
    reference scenarios.

    @param problem_formulations_dict: dictionary
    @param shaded_outcome_name: String: which variable should be related to color
    @param outcome_names: list
    @param uni_color: Boolean: using only one color instead of a color palette
    @param saving: Booelean
    @param file_name: String: file name for saving
    """

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")

    if outcome_names is None:
        outcome_names = [
            'Damages',
            'Atmospheric Temperature'
        ]

    # Size of figure
    figsize_x = len(problem_formulations_dict) * 10
    if shaded_outcome_name is not None:
        figsize_x += 5

    figsize_y = len(outcome_names) * 6

    fig, axes = plt.subplots(
        nrows=len(outcome_names),
        ncols=len(problem_formulations_dict),
        figsize=(figsize_x, figsize_y),  # Ealier: (40, 25)
        tight_layout=False,
        sharey='row'
    )
    plt.subplots_adjust(
        left=None,
        bottom=None,
        right=None,
        top=None,
        wspace=0.2,
        hspace=0.2
    )

    years = list(range(2005, 2310, 10))

    y_labels = get_y_labels_dict()

    axes_font_size = 20

    # Collecting all outcomes
    all_outcomes = None
    for _, outcomes in problem_formulations_dict.items():
        outcomes.index = list(range(len(outcomes)))
        if all_outcomes is None:
            all_outcomes = outcomes
        else:
            all_outcomes = pd.concat([all_outcomes, outcomes])

    if shaded_outcome_name is not None:
        # Getting minimum and maximum for specific hue-related variable
        hue_column = all_outcomes.loc[:, shaded_outcome_name]
        color_variable_min = min(hue_column)
        color_variable_max = max(hue_column)

        # Setting up a color mapper
        norm = mpl.colors.Normalize(vmin=color_variable_min, vmax=color_variable_max, clip=True)
        palette = 'rocket'
        # palette = 'flare'
        cmap = sns.color_palette(palette, as_cmap=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        mapper_list = [mapper.to_rgba(x) for x in hue_column]

        # Splitting mapper into separate problem-formulations-specific lists
        lengths_of_outcomes = {pf: len(outcomes) for pf, outcomes in problem_formulations_dict.items()}
        mapper_dict = {}  # for use of color mapping
        checked_lengths = 0
        for pf, length in lengths_of_outcomes.items():
            mapper_dict[pf] = mapper_list[checked_lengths: checked_lengths+length]
            checked_lengths += length

    discrete_colors = sns.color_palette('Paired', len(problem_formulations_dict))

    # Actual plotting
    for name_idx, name in enumerate(outcome_names):
        for pf_idx, (problem_formulation, outcomes) in enumerate(problem_formulations_dict.items()):

            outcomes.index = list(range(len(outcomes)))
            df = outcomes.filter(regex=name, axis=1)  # Filter columns to include "name"

            for row_idx, row in df.iterrows():

                if uni_color:
                    color = discrete_colors[pf_idx]
                else:
                    color = mapper_dict[problem_formulation][row_idx]

                axes[name_idx, pf_idx].plot(
                    years,
                    row.iloc[:],
                    linewidth=1.0,
                    alpha=1.0,
                    linestyle='-',
                    color=color,
                )

                # Annotations
                if name_idx == 0:
                    terms = (problem_formulation.lower()).split('_')
                    short_name = f'{terms[0]}_{terms[1]}'
                    axes[name_idx, pf_idx].set_title(short_name, fontsize=30, pad=20)
                axes[name_idx, pf_idx].set_xlabel('time in years', fontsize=axes_font_size)
                y_label = y_labels[name]
                axes[name_idx, pf_idx].set_ylabel(y_label, fontsize=axes_font_size)

    # Show labels although sharing y-axis
    for ax in axes.flatten():
        # ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)

    # Color bar
    if shaded_outcome_name is not None:
        cbar = fig.colorbar(mapper, ax=axes, shrink=0.7)

        if shaded_outcome_name == 'Total Output 2105':
            bar_label = 'GWP in 2105 (trillion $)'
        elif shaded_outcome_name == 'Utility 2105':
            bar_label = 'Welfare'
        elif shaded_outcome_name == 'Temperature overshoot 2105':
            bar_label = 'Number of years with a 2°C temperature overshoot'
        else:
            bar_label = shaded_outcome_name
        cbar.set_label(bar_label, labelpad=15)

    plt.show()

    if saving:
        if file_name is None:
            file_name = "pathways"
        sub_folder = 'pathways'
        save_own_figure(fig, file_name, sub_folder)


def plot_pathways_all_problem_formulations(
        problem_formulations_dict,
        outcome_name,
        saving=False,
        file_name=None
):
    """
    Main functino for KPIs
    Plots pathways given one problem formulation. This function considers the results from the
    reference scenarios.

    @param problem_formulations_dict: dictionary
    @param outcome_name: String
    @param saving: Booelean
    @param file_name: String: file name for saving
    """

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(30, 12),
        tight_layout=True,
        sharey='all'
    )
    plt.subplots_adjust(
        left=None,
        bottom=None,
        right=None,
        top=None,
        wspace=0.6,
        hspace=0.4
    )

    years = list(range(2005, 2310, 10))

    y_labels = get_y_labels_dict()

    axes_font_size = 30

    # Collecting all outcomes
    all_outcomes = None
    for _, outcomes in problem_formulations_dict.items():
        outcomes.index = list(range(len(outcomes)))
        if all_outcomes is None:
            all_outcomes = outcomes
        else:
            all_outcomes = pd.concat([all_outcomes, outcomes])

    problem_formulations = [
        ProblemFormulation.UTILITARIAN_AGGREGATED,
        ProblemFormulation.UTILITARIAN_DISAGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_AGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED,
        ProblemFormulation.EGALITARIAN_AGGREGATED,
        ProblemFormulation.EGALITARIAN_DISAGGREGATED,
        ProblemFormulation.PRIORITARIAN_AGGREGATED,
        ProblemFormulation.PRIORITARIAN_DISAGGREGATED,
    ]
    color_mapping = {}
    for _, (problem_formulation, color) in enumerate(zip(problem_formulations, sns.color_palette('Paired'))):
        color_mapping[problem_formulation.name] = color

    # Actual plotting
    for pf_idx, (problem_formulation, outcomes) in enumerate(problem_formulations_dict.items()):

        outcomes.index = list(range(len(outcomes)))
        df = outcomes.filter(regex=outcome_name, axis=1)  # Filter columns to include "name"

        for row_idx, row in df.iterrows():
            color = color_mapping[problem_formulation]

            if pf_idx < 4:
                row_idx = 0
                col_idx = pf_idx
            else:
                row_idx = 1
                col_idx = pf_idx - 4

            axes[row_idx, col_idx].plot(
                years,
                row.iloc[:],
                linewidth=0.5,
                alpha=0.4,
                linestyle='-',
                color=color,
            )

            # Annotations
            terms = problem_formulation.split('_')
            short_name = r'$%s_{%s}$' % (terms[0][0], terms[1][0])
            axes[row_idx, col_idx].set_title(short_name, fontsize=35, pad=20)
            axes[row_idx, col_idx].set_xlabel('time in years', fontsize=axes_font_size)
            y_label = y_labels[outcome_name]
            axes[row_idx, col_idx].set_ylabel(y_label, fontsize=axes_font_size)
            if outcome_name == 'Atmospheric Temperature':
                axes[row_idx, col_idx].yaxis.set_major_locator(MaxNLocator(6))
            else:
                axes[row_idx, col_idx].yaxis.set_major_locator(MaxNLocator(5))

            axes[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=25)
            axes[row_idx, col_idx].tick_params(axis='both', which='minor', labelsize=25)

    # Show labels although sharing y-axis
    for ax in axes.flatten():
        # ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)

    plt.show()

    if saving:
        if file_name is None:
            file_name = "pathways"
        sub_folder = 'pathways'
        save_own_figure(fig, file_name, sub_folder)


def plot_regional_pathways(
        problem_formulations_dict,
        outcome_name,
        regions_list=None,
        saving=False,
        file_name=None,
        resolution=200,
):
    """
    Main functino for KPIs
    Plots pathways given one problem formulation. This function considers the results from the
    reference scenarios.

    @param problem_formulations_dict: dictionary
    @param outcome_name: String
    @param regions_list: list with Strings
    @param saving: Booelean
    @param file_name: String: file name for saving
    @param resolution: int: quality of saved figure
    """

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")

    if regions_list is None:
        regions_list = [
            "US",
            "OECD-Europe",
            "Japan",
            "Russia",
            "Non-Russia Eurasia",
            "China",
            "India",
            "Middle East",
            "Africa",
            "Latin America",
            "OHI",
            "Other non-OECD Asia",
        ]

    fig, axes = plt.subplots(
        nrows=len(regions_list),
        ncols=len(problem_formulations_dict),
        # figsize=(8*len(problem_formulations_dict), 8*len(regions_list)),
        figsize=(8 * len(problem_formulations_dict), 6 * len(regions_list)),
        tight_layout=True,
        sharey='all'
    )
    plt.subplots_adjust(
        left=None,
        bottom=None,
        right=None,
        top=None,
        wspace=0.6,
        hspace=0.4
    )

    years = list(range(2005, 2310, 10))

    axes_font_size = 30

    # Collecting all outcomes
    all_outcomes = None
    for _, outcomes in problem_formulations_dict.items():
        outcomes.index = list(range(len(outcomes)))
        if all_outcomes is None:
            all_outcomes = outcomes
        else:
            all_outcomes = pd.concat([all_outcomes, outcomes])

    # Set colors
    problem_formulations = [
        ProblemFormulation.UTILITARIAN_AGGREGATED,
        ProblemFormulation.UTILITARIAN_DISAGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_AGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED,
        ProblemFormulation.EGALITARIAN_AGGREGATED,
        ProblemFormulation.EGALITARIAN_DISAGGREGATED,
        ProblemFormulation.PRIORITARIAN_AGGREGATED,
        ProblemFormulation.PRIORITARIAN_DISAGGREGATED,
    ]
    color_mapping = {}
    for _, (problem_formulation, color) in enumerate(zip(problem_formulations, sns.color_palette('Paired'))):
        color_mapping[problem_formulation.name] = color

    # Actual plotting
    for region_idx, region in enumerate(regions_list):
        for pf_idx, (problem_formulation, outcomes) in enumerate(problem_formulations_dict.items()):

            outcomes.index = list(range(len(outcomes)))
            df = outcomes.filter(regex=f'{outcome_name} {region}', axis=1)  # Filter columns to include the string outcome_name

            for row_idx, row in df.iterrows():
                color = color_mapping[problem_formulation]

                axes[region_idx, pf_idx].plot(
                    years,
                    row.iloc[:],
                    linewidth=0.9,
                    alpha=0.8,
                    linestyle='-',
                    color=color,
                )

                # Annotations
                terms = problem_formulation.split('_')
                short_name = r'$%s_{%s}$' % (terms[0][0], terms[1][0])
                axes[region_idx, pf_idx].set_title(short_name, fontsize=40, pad=20)
                axes[region_idx, pf_idx].set_xlabel('time in years', fontsize=axes_font_size)
                y_label = outcome_name.split(' ')[1] + f' for {region}'
                axes[region_idx, pf_idx].set_ylabel(y_label, fontsize=axes_font_size)
                axes[region_idx, pf_idx].yaxis.set_major_locator(MaxNLocator(5))

                axes[region_idx, pf_idx].tick_params(axis='both', which='major', labelsize=25)
                axes[region_idx, pf_idx].tick_params(axis='both', which='minor', labelsize=25)

    # Show labels although sharing y-axis
    for ax in axes.flatten():
        # ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)

    plt.show()

    if saving:
        if file_name is None:
            file_name = "pathways"
        sub_folder = 'pathways'
        save_own_figure(fig, file_name, sub_folder, resolution)


def plot_one_pathway(experiments, outcomes, outcome_name, saving=False, file_name=None):
    """
    Plot the pathways of a specific objective grouped by their clusters.
    @param experiments: DataFrame
    @param outcomes: DataFrame
    @param outcome_name: String
    @param saving: Boolean
    @param file_name: String
    """
    sns.set_style("whitegrid")

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
        sub_folder = 'exploration'
        save_own_figure(fig, file_name, sub_folder)


def get_short_problem_formulation_names():
    """
    Return dictionary with short names for problem formulations.
    E.g., 'UTILITARIAN_AGGREGATED' will become 'UA'
    """
    pfs = list(ProblemFormulation)
    d = {}
    for pf in pfs:
        terms = pf.name.split('_')
        new_name = terms[0][0] + terms[1][0]
        d[pf.name] = new_name
    return d


def plot_one_pathway_attempt(problem_formulation_dict, outcome_name, saving=False, file_name=None):
    """
    Plot the pathways of a specific objective grouped by their clusters.
    @param problem_formulation_dict: dictionary {ProblemFormulation.name: experiments, outcomes}
    @param outcome_name: String
    @param saving: Boolean
    @param file_name: String
    """
    sns.set_style("whitegrid")

    name_dict = get_short_problem_formulation_names()

    all_experiments = None
    all_outcomes = None
    for problem_formulation, (experiments, outcomes) in problem_formulation_dict.items():

        experiments['Problem Formulation'] = name_dict[problem_formulation]
        if all_experiments is None:
            all_experiments = experiments
        else:
            all_experiments = pd.concat([all_experiments, experiments])

        if all_outcomes is None:
            all_outcomes = outcomes
        else:
            all_outcomes = pd.concat([all_outcomes, outcomes])

    reshaphed_outcomes = get_outcomes_reshaped(
        outcomes_df=all_outcomes, objective_names=[outcome_name]
    )

    fig, axes = plotting.lines(
        experiments=all_experiments,
        outcomes=reshaphed_outcomes,
        outcomes_to_show=outcome_name,
        group_by='Problem Formulation',
        density=Density.BOXPLOT,
    )
    fig.set_size_inches(12, 8)
    plt.show()

    if saving:
        if file_name is None:
            file_name = 'open_exploration_pathways'
        sub_folder = 'exploration'
        save_own_figure(fig, file_name, sub_folder)


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
        objective = dict(range=range_boundaries, label=obj_name, values=outcomes.loc[:, obj_name])

        # Add dict
        dimensions_list.append(objective)

    fig = go.Figure(
        data=go.Parcoords(
            # Policy lines
            # line=dict(color=experiments['policy'], showscale=True),
            # line=dict(color=((34, 139, 34), alpha)),
            # line=dict(color='forestgreen', showscale=True),
            line=dict(color='forestgreen'),
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
        'Utility': 'welfare',
        'Disutility': 'welfare loss',
        'Lowest income per capita': 'lowest income per capita ($1000)',
        'Intratemporal consumption Gini': 'Gini consumption',
        'Highest damage per capita': 'highest damage per capita',
        'Intratemporal damage Gini': 'Gini damage',
        'Population below consumption threshold': 'population below\nconsumption threshold (million)',
        'Distance to consumption threshold': 'distance to\nconsumption threshold',
        'Population above damage threshold': 'population above\ndamage threshold (million)',
        'Distance to damage threshold': 'distance to\ndamage threshold (trillion $)',
        'Temperature overshoot': '# of time steps\n with 2°C temperature overshoots',
        'Damages': 'economic damages (trillion $)',
        'Industrial Emission': 'global emissions (GTon CO2)',
        'Atmospheric Temperature': 'increase in\natmospheric temperature (°C)',
        'Total Output': 'GWP (trillion $)',
        'Number of regions above damage threshold': 'Number of regions\nabove damage threshold',
        'Number of regions below consumption threshold': 'Number of region-quintiles\nbelow consumption threshold'
    }

    return info_dict


def get_flat_y_labels_dict():
    """
    Returns a dictionary that provides y_label information for a given objective.
    @return
        info_dict: dictionary: {objective_name (string): name + units (string)}
    """

    info_dict = {
        'Utility': 'welfare',
        'Disutility': 'welfare loss',
        'Lowest income per capita': 'lowest income per capita ',
        'Intratemporal consumption Gini': 'Gini consumption',
        'Highest damage per capita': 'highest damage per capita',
        'Intratemporal damage Gini': 'Gini damage',
        'Population below consumption threshold': 'population below consumption threshold',
        'Distance to consumption threshold': 'distance to consumption threshold',
        'Population above damage threshold': 'population above damage threshold',
        'Distance to damage threshold': 'distance to damage threshold ',
        'Temperature overshoot': '# of time steps with 2°C temperature overshoots',
        'Damages': 'economic damages',
        'Industrial Emission': 'global emissions',
        'Atmospheric Temperature': 'increase in atmospheric temperature ',
        'Total Output': 'GWP',
        'Number of regions above damage threshold': 'Number of regions above damage threshold',
        'Number of regions below consumption threshold': 'Number of region-quintiles below consumption threshold'
    }

    return info_dict


def plot_policies_per_problem_formulation(problem_formulations_dict, saving=False, file_name=None):
    """
    Plots a parallel axis plots with all levers as axes and color-coded by problem formulation.
    @param problem_formulations_dict: dictionary with
                                      {ProblemFormulation: (experiments: DataFrame, outcomes: DataFrame)}
    @param saving: Boolean: whether to save the figure
    @param file_name: String

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
        sub_folder = 'optimalpolicies'
        save_own_figure(axes.fig, file_name, sub_folder)


def plot_optimal_policies(policies, saving=False, file_name=None):
    """
    Plots a parallel axis plots with all levers as axes and color-coded by problem formulation.
    @param policies: DataFrame
    @param saving: Boolean: whether to save the figure
    @param file_name: String

    """
    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    sns.set(rc={'figure.figsize': (12, 8)})

    axes = None

    limits = parcoords.get_limits(policies)

    if axes is None:
        axes = parcoords.ParallelAxes(limits)
    axes.plot(policies, color='forestgreen')

    # axes.legend()
    plt.show()

    if saving:
        if file_name is None:
            file_name = 'optimal_policies'
        file_name += '.png'
        sub_folder = 'optimalpolicies'

        save_own_figure(axes.fig, file_name, sub_folder)


def plot_optimal_policies_dict(policies_dict, saving=False, file_name=None):
    """
    Plots a parallel axis plots with all levers as axes and color-coded by problem formulation.
    @param policies_dict: {ProblemFormulation: DataFrame}
    @param saving: Boolean: whether to save the figure
    @param file_name: String

    """
    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    sns.set(rc={'figure.figsize': (18, 12)})

    # Problem formulations and colors
    color_mapping = {}
    for _, (problem_formulation, color) in enumerate(zip(policies_dict, sns.color_palette('Paired'))):
        color_mapping[problem_formulation] = color

    # Handling limits (such that all limits are considered)
    all_policies = None
    for problem_formulation, policies in policies_dict.items():

        if all_policies is None:
            all_policies = policies
        else:
            all_policies = pd.concat([all_policies, policies])

    limits = parcoords.get_limits(all_policies)
    axes = parcoords.ParallelAxes(limits)

    for problem_formulation, policies in policies_dict.items():
        axes.plot(
            policies,
            color=color_mapping[problem_formulation],
            label=problem_formulation,
            linewidth=1.5,
            alpha=1.0
        )

    axes.legend()
    plt.show()

    if saving:
        if file_name is None:
            file_name = 'optimal_policies'
        file_name += '.png'
        sub_folder = 'optimalpolicies'
        save_own_figure(axes.fig, file_name, sub_folder)


def plot_parallel_axis_plot_objectives(problem_formulations_dict, saving=False, file_name=None):
    """
    Plots a parallel axis plots with all levers as axes and color-coded by problem formulation.
    @param problem_formulations_dict: {ProblemFormulation: DataFrame}
    @param saving: Boolean: whether to save the figure
    @param file_name: String

    """
    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    sns.set(rc={'figure.figsize': (18, 12)})

    # Problem formulations and colors
    color_mapping = {}
    for _, (problem_formulation, color) in enumerate(zip(problem_formulations_dict, sns.color_palette('Paired'))):
        color_mapping[problem_formulation] = color

    # Handling limits (such that all limits are considered)
    all_outcomes = None
    for problem_formulation, outcomes in problem_formulations_dict.items():

        if all_outcomes is None:
            all_outcomes = outcomes
        else:
            all_outcomes = pd.concat([all_outcomes, outcomes])

    limits = parcoords.get_limits(all_outcomes)
    axes = parcoords.ParallelAxes(limits)

    for problem_formulation, outcomes in problem_formulations_dict.items():
        axes.plot(
            outcomes,
            color=color_mapping[problem_formulation],
            label=problem_formulation,
            linewidth=1.5,
            alpha=1.0
        )

    axes.legend()
    plt.show()

    if saving:
        if file_name is None:
            file_name = 'optimal_policies'
        file_name += '.png'
        sub_folder = 'optimalpolicies'
        save_own_figure(axes.fig, file_name, sub_folder)


def plot_single_parallel_axis_plot(
        df,
        gray_df=None,
        limits=None,
        problem_formulation=None,
        sub_folder=None,
        saving=False,
        file_name=None
):
    """
    Plots a parallel axis plots with all levers as axes and color-coded by problem formulation.
    @param df: DataFrame
    @param gray_df: DataFrame
    @param limits: DataFrame
    @param problem_formulation: ProblemFormulation
    @param sub_folder: String
    @param saving: Boolean: whether to save the figure
    @param file_name: String

    """
    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    sns.set(rc={'figure.figsize': (5, 5)})

    minimize_list = [
        'damages',
        'welfare\nloss',
        'atmospheric\ntemperature',
        'industrial\nemission',
        'temperature\novershoots',
        "distance to\nconsumption\nthreshold",
        "population below\nconsumption\nthreshold",
        "distance to\ndamage\nthreshold",
        "population above\ndamage\nthreshold",
        'highest damage\nper capita',
        "consumption gini",
        "damage gini",
    ]

    problem_formulations = [
        ProblemFormulation.UTILITARIAN_AGGREGATED.name,
        ProblemFormulation.UTILITARIAN_DISAGGREGATED.name,
        ProblemFormulation.SUFFICIENTARIAN_AGGREGATED.name,
        ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED.name,
        ProblemFormulation.EGALITARIAN_AGGREGATED.name,
        ProblemFormulation.EGALITARIAN_DISAGGREGATED.name,
        ProblemFormulation.PRIORITARIAN_AGGREGATED.name,
        ProblemFormulation.PRIORITARIAN_DISAGGREGATED.name,
    ]

    if problem_formulation is None:
        problem_formulation = ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED.name

    df = get_formatted_column_names(df)
    if gray_df is not None:
        gray_df = get_formatted_column_names(gray_df)

    if limits is None:
        if gray_df is not None:
            limits = parcoords.get_limits(gray_df)
        else:
            limits = parcoords.get_limits(df)

    limits = get_formatted_column_names(limits)
    axes = parcoords.ParallelAxes(limits)

    # Problem formulations and colors
    color_mapping = {}
    for _, (pf, color) in enumerate(zip(problem_formulations, sns.color_palette('Paired'))):
        color_mapping[pf] = color

    if gray_df is not None:
        axes.plot(
            gray_df,
            color=to_rgb((230/256, 230/256, 230/256)),
            linewidth=0.8,
            alpha=0.5
        )

    axes.plot(
        df,
        color=color_mapping[problem_formulation],
        label=problem_formulation,
        linewidth=0.5,
        alpha=0.4,
        # linewidth=2.0,
        # alpha=1.0
    )

    # Invert axes where necessary
    for o in minimize_list:
        if o in df.columns:
            axes.invert_axis(o)

    # axes.legend()
    plt.show()

    if saving:
        if file_name is None:
            file_name = f'trade-off_{problem_formulation}'

        if sub_folder is None:
            sub_folder = 'tradeoffs'
        save_own_figure(axes.fig, file_name, sub_folder)


def get_formatted_column_names(df):
    """
    Reformat names of objectives to shorter names.
    @param df: DataFrame
    @return new_df: DataFrame
    """
    new_df = df.rename(columns={
        'Damages 2105': 'damages',
        'Disutility 2105': 'welfare\nloss',
        'Atmospheric Temperature 2105': 'atmospheric\ntemperature',
        'Industrial Emission 2105': 'industrial\nemission',
        'Temperature overshoot 2105': 'temperature\novershoots',
        "Distance to consumption threshold 2105": "distance to\nconsumption\nthreshold",
        "Population below consumption threshold 2105": "population below\nconsumption\nthreshold",
        "Distance to damage threshold 2105": "distance to\ndamage\nthreshold",
        "Population above damage threshold 2105": "population above\ndamage\nthreshold",
        'Number of regions below consumption threshold 2105': 'number of regions\nbelow consumptionthreshold',
        'Number of regions above damage threshold 2105': 'number of regions\nabove damage threshold',
        'Highest damage per capita 2105': 'highest damage\nper capita',
        "Intratemporal consumption Gini 2105": "consumption gini",
        "Intratemporal damage Gini 2105": "damage gini",
        'Utility 2105': 'welfare',
        'Lowest income per capita 2105': 'lowest income\nper capita',
        'Total Output 2105': 'total output',

        'sr': 'savings rate (%)',
        'miu': 'net-zero\nemission target\n(year)',
        'irstp_consumption': 'initial rate of social\ntime preference\nfor consumption',
        'irstp_damage': 'initial rate of social\ntime preference\nfor damage',

    })

    return new_df


def get_limits_from_several_sources(input_variable):
    """
    Compute the limits for the parallel axis plot (ema workbench) from several dataframes, such that the limits can be
    used to better compare the plots.
    @param input_variable: list or dict with DataFrames
    @return limits: DataFrame
    """
    if isinstance(input_variable, list):
        all_dfs = pd.concat(input_variable)
        all_dfs = get_formatted_column_names(all_dfs)
        limits = parcoords.get_limits(all_dfs)
    elif isinstance(input_variable, dict):
        # Handling limits (such that all limits are considered)
        all_dfs = None
        for _, df in input_variable.items():

            if all_dfs is None:
                all_dfs = df
            else:
                all_dfs = pd.concat([all_dfs, df])
        all_dfs = get_formatted_column_names(all_dfs)
        limits = parcoords.get_limits(all_dfs)
    else:
        raise ValueError('The input has to be of type dict or list with DataFrames!')
    return limits


def plot_robustness(robustness_dataframe, pf, legend=False, saving=False, file_name=None):
    """
    Plot the robustness of some KPIs on a parallel axis plot.
    @param robustness_dataframe: DataFrame
    @param pf: String (problem formulation)
    @param legend: boolean
    @param saving: Boolean: whether to save the figure
    @param file_name: String
    """

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    sns.set(rc={'figure.figsize': (5, 5)})

    # Colors
    unique_problem_formulations = robustness_dataframe.loc[:, 'Problem Formulation'].unique()
    problem_formulations = [
        ProblemFormulation.UTILITARIAN_AGGREGATED,
        ProblemFormulation.UTILITARIAN_DISAGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_AGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED,
        ProblemFormulation.EGALITARIAN_AGGREGATED,
        ProblemFormulation.EGALITARIAN_DISAGGREGATED,
        ProblemFormulation.PRIORITARIAN_AGGREGATED,
        ProblemFormulation.PRIORITARIAN_DISAGGREGATED,
    ]
    color_mapping = {}
    for _, (problem_formulation, color) in enumerate(zip(problem_formulations, sns.color_palette('Paired'))):
        color_mapping[problem_formulation.name] = color

    # Handling limits (such that all limits are considered)
    all_policies = None
    for problem_formulation in unique_problem_formulations:

        relevant_policies = robustness_dataframe[robustness_dataframe['Problem Formulation'] == problem_formulation]
        relevant_policies = relevant_policies.drop(columns=['Problem Formulation', 'Policy'])

        if all_policies is None:
            all_policies = relevant_policies
        else:
            all_policies = pd.concat([all_policies, relevant_policies])

    limits = parcoords.get_limits(all_policies)
    axes = parcoords.ParallelAxes(limits)

    for problem_formulation in unique_problem_formulations:

        if problem_formulation == pf:

            relevant_policies = robustness_dataframe[robustness_dataframe['Problem Formulation'] == problem_formulation]
            relevant_policies = relevant_policies.drop(columns=['Problem Formulation', 'Policy'])

            if axes is None:
                axes = parcoords.ParallelAxes(limits)
            axes.plot(relevant_policies, color=color_mapping[problem_formulation], label=problem_formulation)

    if legend:
        axes.legend()
    plt.show()

    if saving:
        if file_name is None:
            file_name = 'robustness'
        sub_folder = 'optimalpolicies'
        save_own_figure(axes.fig, file_name, sub_folder)


def plot_boxplots(dict_list, outcome_names, year=2105, saving=False, file_name=None):
    """
    Takes in a number of dictionaries with experimental data, and creates box plots.
    Each row represents one metric and consists of 3 subplots.
    Each subplot contains a boxplot for each problem formulation.
    @param outcome_names: list of Strings
    @param dict_list: list with dictionary
    @param year: int (anything in [2005, 2305] with steps of 10)
    @param saving: Boolean
    @param file_name: String
    """

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")

    nrows = len(outcome_names)
    ncols = 3

    figsize = (ncols * 9, nrows * 6)
    # figsize = (ncols * 4, nrows * 6)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        tight_layout=True,
        sharey='row',
        sharex='col'
    )
    plt.subplots_adjust(
        left=None,
        bottom=None,
        right=None,
        top=None,
        wspace=0.5,
        hspace=0.8
    )

    mapping = {
        0: '4 reference scenarios',
        1: '50 bad scenarios',
        2: '50 random scenarios',
    }

    delft_blue = (0/256, 166/256, 214/256)  # TU Deflt Blue
    delft_green = (0/256, 155/256, 119/256)  # TU Deflt Green
    delft_colors = (delft_blue, delft_green)

    y_labels = get_y_labels_dict()

    for outcome_idx, outcome_name in enumerate(outcome_names):
        for dict_idx, problem_formulations in enumerate(dict_list):
            all_outcomes = None
            for problem_formulation, outcomes in problem_formulations.items():
                terms = problem_formulation.split('_')
                outcomes['problem formulation'] = r'$%s_{%s}$' % (terms[0][0], terms[1][0])
                if all_outcomes is None:
                    all_outcomes = outcomes
                else:
                    all_outcomes = pd.concat([all_outcomes, outcomes])

            var_min = all_outcomes[f'{outcome_name} {year}'].min()
            var_max = all_outcomes[f'{outcome_name} {year}'].max()
            # Boxplots
            sns.boxplot(
                x='problem formulation',
                y=f'{outcome_name} {year}',
                data=all_outcomes,
                ax=axes[outcome_idx, dict_idx],
                palette=sns.color_palette('Paired'),
                # palette=delft_colors,

            )

            if outcome_idx == 0:
                axes[outcome_idx, dict_idx].set_title(mapping[dict_idx], pad=30, fontsize=24)
            axes[outcome_idx, dict_idx].set_ylabel(y_labels[outcome_name])
            # axes[outcome_idx, dict_idx] .set_xlabel('Time in years', fontsize=axes_font_size)
            # axes[outcome_idx, dict_idx] .set_ylabel(name, fontsize=axes_font_size)

    # Show labels although sharing axes
    for ax in axes.flatten():
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)

    if saving:
        if file_name is None:
            file_name = 'boxplots_3_ways'
        sub_folder = 'boxplots'
        save_own_figure(fig, file_name, sub_folder)

    plt.show()


def compute_medians(problem_formulations_dict, kpi, year):
    """
    Compute the medians per problem formulations for a given KPI.
    @param problem_formulations_dict: dictionary: {problem formulation: outcomes DataFrame}
    @param kpi: string
    @param year: int
    @return:
        pf_medians_dict: dictionary: {problem formulation: median}
    """

    column_name = f'{kpi} {year}'

    pf_medians_dict = {}

    for pf, outcomes in problem_formulations_dict.items():

        kpi_median = outcomes.loc[:, column_name].median()

        pf_medians_dict[pf] = kpi_median

    return pf_medians_dict


def compute_relative_values(medians_dict):
    """
    Compute relative differences between medians of problem formulations.
    @param medians_dict:
    @return:
        data: np.array (2d)
    """
    data = np.zeros((len(medians_dict), len(medians_dict)))

    # maximum difference
    values = sorted(list(medians_dict.values()))
    max_difference = abs(values[-1] - values[0])

    for row_idx, (_, median_1) in enumerate(medians_dict.items()):
        for col_idx, (_, median_2) in enumerate(medians_dict.items()):
            dif = (median_1 - median_2) / max_difference
            data[row_idx, col_idx] = dif

    return data


def plot_pf_median_relations(problem_formulations_dict, kpi, year, maximize, saving=False, file_name=None):
    """
    Plot a grid with colored cells. A cell represents the relation between two problem formulations.
    @param problem_formulations_dict: dictionary: {problem formulation: outcomes DataFrame}
    @param kpi: string
    @param year: int
    @param maximize: Boolean: should kpi be maximized?
    @param saving: Boolean
    @param file_name: String
    """

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 12))

    if maximize:
        cmap = sns.diverging_palette(20, 145, as_cmap=True)
    else:
        cmap = sns.diverging_palette(145, 20, as_cmap=True)

    medians_dict = compute_medians(problem_formulations_dict, kpi, year)

    data = compute_relative_values(medians_dict)

    plt.pcolormesh(data, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_label('relative performance', labelpad=15, fontsize=30)

    x_labels = ax.get_xticks().tolist()
    y_labels = ax.get_yticks().tolist()

    # Shorten problem formulation names
    pf_strings = []
    for pf in problem_formulations_dict.keys():
        terms = pf.split('_')
        short_pf = r'$%s_{%s}$' % (terms[0][0], terms[1][0])
        pf_strings.append(short_pf)

    # Create new lables
    new_labels = []
    for pf in pf_strings:
        # new_labels.append('')
        new_labels.append(pf)
    new_labels.append('')

    # Shift ticks
    SHIFT = -0.5  # Data coordinates
    for label in ax.xaxis.get_majorticklabels():
        label.customShiftValue = SHIFT
        label.set_x = types.MethodType(lambda self, x: matplotlib.text.Text.set_x(self, x - self.customShiftValue),
                                       label)

    SHIFT = -0.5  # Data coordinates
    for label in ax.yaxis.get_majorticklabels():
        label.customShiftValue = SHIFT
        label.set_y = types.MethodType(lambda self, y: matplotlib.text.Text.set_y(self, y - self.customShiftValue),
                                       label)

    plt.xticks(x_labels, new_labels, fontsize=40)
    plt.yticks(y_labels, new_labels, fontsize=40)

    name = get_flat_y_labels_dict()[kpi]
    # plt.title(f"{name}")

    if saving:
        if file_name is None:
            file_name = f'kpi_medians_{name}'
        sub_folder = 'relativemedians'
        save_own_figure(fig, file_name, sub_folder)

    plt.show()
