"""
This module contains functions to compute and visualize convergence data.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns


def visualize_epsilon_progress(saving=False):
    """
    Load and plot epsilon progress of the four problem formulations..
    """

    directory = os.getcwd()

    # Load data
    df_con1 = pd.read_csv(directory + '/results/NORDHAUS_UTILITARIAN_200000_convergence.csv')
    df_con2 = pd.read_csv(directory + '/results/NORDHAUS_SUFFICIENTARIAN_200000_convergence.csv')
    df_con3 = pd.read_csv(directory + '/results/WEITZMAN_UTILITARIAN_200000_convergence.csv')
    df_con4 = pd.read_csv(directory + '/results/WEITZMAN_SUFFICIENTARIAN_200000_convergence.csv')

    # Set parameters
    x_label = '$\epsilon$-progress'
    y_label = 'nfe'

    # Create subplots
    sns.set(font_scale=1.35)
    sns.set_style("whitegrid")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex='all', figsize=(20, 4), tight_layout=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    fig.patch.set_facecolor('white')

    ax1.plot(df_con1.nfe, df_con1.epsilon_progress)
    ax1.set_ylabel(x_label)
    ax1.set_title('Nordhaus + Utilitarian', fontsize=22)
    ax1.set_xlabel(y_label)

    ax2.plot(df_con3.nfe, df_con3.epsilon_progress)
    ax2.set_ylabel(x_label)
    ax2.set_title('Weitzman + Utilitarian', fontsize=22)
    ax2.set_xlabel(y_label)

    ax3.plot(df_con2.nfe, df_con2.epsilon_progress)
    ax3.set_ylabel(x_label)
    ax3.set_title('Nordhaus + Sufficientarian', fontsize=22)
    ax3.set_xlabel(y_label)

    ax4.plot(df_con4.nfe, df_con4.epsilon_progress)
    ax4.set_ylabel(x_label)
    ax4.set_title('Weitzman + Sufficientarian', fontsize=22)
    ax4.set_xlabel(y_label)

    # sns.despine()
    plt.show()

    if saving:
        directory = os.getcwd()
        # root_directory = os.path.dirname(directory)
        visualization_folder = directory + '/outputimages/'
        fig.savefig(visualization_folder + "convergence_epsilon_progress.png", dpi=200, pad_inches=0.2)
