"""
This module contains functions to compute the silhouette widths for some given objectives.

1. load data from open exploration
2. calculate distances with CID per objective
3. apply agglomerative clustering on the distances and compute silhouette widths

After computing the silhouette widths, we will do:
1. determine # of clusters per objective via visual inspection of graph (of silhouette widths)
2. identify worst cluster per objective (via visual inspection of pathway plot of objective per cluster)
merge worst-off scenarios

We will need these silhouette widths, such that we can find the worst-off scenarios.
"""

from ema_workbench import load_results
import itertools
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ema_workbench.analysis.clusterer import apply_agglomerative_clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from optimization.general.visualization import plot_pathways
from optimization.general.xlm_constants_epsilons import get_all_outcome_names


def get_outcomes_reshaped(outcomes_df, objective_names):
    """
    Reshape outcomes such that it summarizes the objectives by year. Instead of 'Utility 2005', etc., there will be
    keys such as 'Utility', etc.
    @param outcomes_df: DataFrame
    @param objective_names: list with Strings
    @return:
        outcomes_reshaped: dictionary
    """
    outcomes_reshaped = {}
    for name in objective_names:
        df = outcomes_df.filter(regex=name, axis=1)
        column_names = list(df.columns)
        outcomes_reshaped[name] = np.stack([df[x] for x in column_names], axis=-1)

    return outcomes_reshaped


def compute_silhouette_widths(results, objective_names=None, max_cluster=10):
    """
    Computes the slihouette widths for some given objectives.
    @param results: DataFrame, dictionary (data from perform_experiments)
    @param objective_names: list with Strings
    @param max_cluster: maximal number of clusters to consider
    @return
        widths_df: DataFrame
        distances: 2d numpy array
    """

    experiments, outcomes = results
    outcomes_df = pd.DataFrame(outcomes)

    # In case, the number of experiments is too small (mostly for testing purposes)
    if len(experiments) > max_cluster:
        max_cluster = len(experiments)

    if objective_names is None:
        objective_names = get_all_outcome_names()

    # Dictionary for saving the widths per objective
    widths_dict = {}

    # Dictionary for saving the clusters per objective+cluster combination
    cluster_dict = {}

    outcomes_reshaped = get_outcomes_reshaped(outcomes_df, objective_names)

    # Clustering
    cluster_numbers = list(range(2, max_cluster+1))

    for idx, objective in enumerate(objective_names):

        print(f'\nComputing objective #{idx+1}/{len(objective_names)}')

        # Compute distances
        data = outcomes_reshaped[objective]
        distances = calculate_cid(data)

        # Compute silhouette widths
        widths = []
        for k in cluster_numbers:

            print(f'\tcluster #{k}/{max_cluster}')

            clusterers = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage="complete")
            cluster_labels = clusterers.fit_predict(distances)
            silhouette_avg = silhouette_score(distances, cluster_labels, metric="precomputed")
            widths.append(silhouette_avg)

            # apply agglomerative clustering on distances and create appropriate csv files
            clusters = apply_agglomerative_clustering(distances, n_clusters=k)

            cluster_dict[objective + '_' + str(k)] = clusters

        widths_dict[objective] = widths

    # Save resulting data

    widths_df = pd.DataFrame(widths_dict, index=cluster_numbers)
    cluster_df = pd.DataFrame(cluster_dict)

    target_directory = os.getcwd() + '/data/'
    # Save silhouette widths
    file_name = f'silhouette_widths_{len(experiments)}.csv'
    # noinspection PyTypeChecker
    widths_df.to_csv(target_directory + file_name, index=cluster_numbers)

    # Save clusters
    file_name = f'clusters_{len(experiments)}.csv'
    # noinspection PyTypeChecker
    cluster_df.to_csv(target_directory + file_name, index=cluster_numbers)

    return widths_df


def calculate_cid(data):
    """calculate the complex invariant distance between all rows

    Remark: ema_workbench.analysis.clusterer.calculate_cid has the same implementation, however, the helper function
    runs into an issue of dividing by zero. That's why I copied the code and made a small adjustment.

    @param data : 2d ndarray
    @return:
        distances
            a 2D ndarray with the distances between all time series
    """
    ce = np.sqrt(np.sum(np.diff(data, axis=1) ** 2, axis=1))

    indices = np.arange(0, data.shape[0])
    cid = np.zeros((data.shape[0], data.shape[0]))

    for i, j in itertools.combinations(indices, 2):
        xi = data[i, :]
        xj = data[j, :]
        ce_i = ce[i]
        ce_j = ce[j]

        distance = _calculate_cid(xi, xj, ce_i, ce_j)
        cid[i, j] = distance
        cid[j, i] = distance

    return cid


def _calculate_cid(xi, xj, ce_i, ce_j):
    """
    @param xi:
    @param xj:
    @param ce_i:
    @param ce_j:
    @return:
    """
    return np.linalg.norm(xi - xj) * (max(ce_i, ce_j) / max(0.001, min(ce_i, ce_j)))  # avoiding to divide by zero


def plot_silhouette_widths(widths, saving=False, file_name=None):
    """
    Plot silhouette widths for all objectives.
    @param widths: Dataframe
    @param saving: Boolean: whether to save the resulting figure
    @param file_name: String: title of resulting figure
    """
    sns.set(font_scale=1.6)
    sns.set_style("whitegrid")

    objectives = list(widths.columns)

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(36, 26), tight_layout=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.8)

    # Figures
    for i, ax in enumerate(axes.flat):
        if i >= len(objectives):
            break

        objective = objectives[i]
        data = widths.loc[:, objective]
        ax.plot(data, linewidth=5.0, alpha=1.0, color='forestgreen', marker='o', markersize=20)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ax.set_title(objective, fontsize=30)
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Average silhouette width')

    axes[-1, -1].axis('off')
    plt.show()

    if saving:

        visualization_folder = os.path.dirname(os.path.dirname(os.getcwd())) + '/outputimages/'
        if file_name is None:
            file_name = "scenario_discovery_time_series_clustering_silhouette_widths"
        file_name += ".png"
        fig.savefig(visualization_folder + file_name, dpi=200, pad_inches=0.2)


def get_experiments_with_clusters(objective, cluster_number, results_name='results_open_exploration_100'):
    """
    Get the experiments dataframe with the corresponding clusters-column
    @param objective: String
    @param cluster_number: int
    @param results_name: String: name of file from open exploration
    @return
        x: DataFrame: experiments with extra column (clusters)
    """

    # Loading data
    target_directory = os.path.dirname(os.path.dirname(os.getcwd())) + '/exploration/data/'
    results = load_results(file_name=target_directory + results_name)

    experiments, outcomes = results

    # Load cluster data
    target_directory = os.getcwd() + '/data/'
    file_name = 'clusters.csv'
    clusters_df = pd.read_csv(target_directory + file_name)

    # Get specific clusters for objective and cluster number
    objective_name = f'{objective}_{cluster_number}'
    clusters = clusters_df[objective_name]

    # Save into dataframe
    x = experiments.copy()
    x['clusters'] = clusters.astype('int')

    return x


if __name__ == '__main__':

    print('Starting...\n')

    n_scenarios = 30000

    # Loading data
    target_directory = os.path.dirname(os.path.dirname(os.getcwd())) + '/exploration/data/'
    file_name = f'results_open_exploration_{n_scenarios}'
    results = load_results(file_name=target_directory + file_name)

    print('\n############ Computing silhouette widths... ############')
    # Computing silhouette widths
    widths = compute_silhouette_widths(results)

    print('\n############ Plotting silhouette widths... ############')
    # Plotting silhouette widths
    plot_silhouette_widths(widths, saving=True)

    print('\n############ Plotting open exploration data... ############')
    # Plotting open exploration data
    _, outcomes = results
    outcomes_df = pd.DataFrame(outcomes)
    outcome_names = get_all_outcome_names()
    plot_pathways(outcomes_df, outcome_names, saving=True, file_name=f'pathways_open_exploration_{n_scenarios}')

    print('\n############ Done! ############')
