"""
This module contains helper functions to perform experiments.
"""
import pandas as pd
from ema_workbench import ScalarOutcome
from ema_workbench.util.utilities import load_results

import matplotlib.pyplot as plt
import seaborn as sns
import os


def prepare_info_outcomes(args):
    """
    Given specific outcome names, attach relevant years to the names and create ScalarOutcome variables for those.
    @param args: string: names of outcome variables, e.g., 'Utility'
    @return:
        outcomes_list: list of ScalarOutcomes
    """

    outcomes_list = []
    years = list(range(2005, 2310, 10))
    for name in args:
        outcomes1 = [ScalarOutcome(name + ' ' + str(x)) for x in years]
        outcomes_list.extend(outcomes1)

    return outcomes_list


def get_columns_by_outcome_prefix(outcomes_df, prefix):
    """

    @param outcomes_df:
    @param prefix:
    @return:
    """
    df = outcomes_df.filter(regex=prefix, axis=1)
    return df


def plot_pathways(outcomes_df, args, saving=False, file_name=None):
    """
    Currently not super stable. Might break easily because of length of args.
    @param outcomes_df:
    @param args:
    @param saving: Booelean
    @param file_name: String: file name for saving
    @return:
    """

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")

    if len(args) <= 4:
        nrows = 2
        ncols = 2
    elif len(args) <= 6:
        nrows = 3
        ncols = 2
    elif len(args) <= 9:
        nrows = 3
        ncols = 3
    elif len(args) <= 12:
        nrows = 3
        ncols = 4
    elif len(args) <= 16:
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
        if i >= len(args):
            break
        name = args[i]
        df = get_columns_by_outcome_prefix(outcomes_df, prefix=name)
        for idx, row in df.iterrows():
            ax.plot(years, row.iloc[:], linewidth=linewidth, alpha=alpha, color='b')

        ax.set_title(name)
        ax.set_xlabel('Time in years')
        ax.set_ylabel(name)

    plt.show()

    if saving:
        directory = os.getcwd()
        root_directory = os.path.dirname(directory)
        visualization_folder = root_directory + '/optimization/outputimages/'
        if file_name is None:
            file_name = "pathways_experiments"
        file_name += ".png"
        fig.savefig(visualization_folder + file_name, dpi=200, pad_inches=0.2)


if __name__ == '__main__':

    # Loading results
    directory = os.getcwd() + '/results/'
    results = load_results(file_name=directory + 'test_results')
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

    plot_pathways(outcomes, outcome_names)
