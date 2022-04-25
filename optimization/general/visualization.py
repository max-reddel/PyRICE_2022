"""
This module contains functions to visualize data, hypervolume, etc.
"""

from ema_workbench.util.utilities import load_results
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


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

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 24), tight_layout=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.8)

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
            ax.plot(years, row.iloc[:], linewidth=linewidth, alpha=alpha, color='forestgreen')

        ax.set_title(name)
        ax.set_xlabel('Time in years')
        ax.set_ylabel(name)

    axes[-1, -1].axis('off')
    plt.show()

    if saving:

        visualization_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/outputimages/'
        if file_name is None:
            file_name = "open_exploration_pathways"
        file_name += ".png"
        fig.savefig(visualization_folder + file_name, dpi=200, pad_inches=0.2)


if __name__ == '__main__':

    directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/exploration/data/'
    results = load_results(file_name=directory + 'results_open_exploration_10')
    experiments, outcomes = results
    outcomes = pd.DataFrame(outcomes)

    outcome_names = [
        'Distance to consumption threshold',
        'Distance to damage threshold',
        'Population below consumption threshold',
        'Population above damage threshold',
        'Utility',
        'Disutility'
    ]

    plot_pathways(outcomes, outcome_names, saving=True, file_name='results_testing')
