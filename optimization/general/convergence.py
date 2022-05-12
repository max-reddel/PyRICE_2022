"""
This module contains functions to load and plot convergence behavior.
"""

from model.enumerations import ProblemFormulation
import os
from platypus import Solution, Problem
import pandas as pd
from platypus import Hypervolume
import matplotlib.pyplot as plt
import seaborn as sns


def plot_epsilon_progress(data_folder_path, searchover, nfe=100000, saving=False):
    """
    Plot the epsilon progress of all eight problem formulations.
    @param data_folder_path: String: path to the data folder in which all relevant data is saved
    @param searchover: String: {'levers', 'uncertainties'}
    @param nfe: int: this indicates what the desired nfe is (has to exist in the results)
    @param saving: Boolean: whether to save image
    """

    problem_formulations = [pf.name for pf in ProblemFormulation.get_8_problem_formulations()]

    # Set parameters
    x_label = "$\epsilon$-progress"
    y_label = "nfe"

    # Create subplots
    sns.set(font_scale=1.35)
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(28, 14), tight_layout=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    fig.patch.set_facecolor("white")

    # Load epsilon values for each problem formulation
    for idx, ax in enumerate(axes.flat):
        problem_formulation = problem_formulations[idx]
        directory = os.path.join(data_folder_path, f'{problem_formulation}_{searchover}_{nfe}')
        df = pd.read_csv(os.path.join(directory, 'epsilon_progress.csv'))

        ax.plot(df.nfe, df.epsilon_progress)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylabel(x_label)
        ax.set_title(problem_formulation, fontsize=22)
        ax.set_xlabel(y_label)

    plt.show()

    if saving:
        directory = os.path.join(os.getcwd(), 'data', 'convergence_epsilon_progress.png')
        fig.savefig(directory, dpi=200, pad_inches=0.2)


def _define_problem(problem_formulation):
    """
    @param problem_formulation: String: combination of damage function and welfare function
    @return:
        problem: Platypus Problem
        n_objs: int: number of relevant objectives
        n_decision_vars: int: number of decision variables
    """

    n_decision_vars = 4

    if problem_formulation.name == 'UTILITARIAN_AGGREGATED':
        n_objs = 2
        problem = Problem(n_decision_vars, n_objs)
        problem.directions[0] = Problem.MAXIMIZE  # welfare of utility 2105
        problem.directions[1] = Problem.MINIMIZE  # temperature overshoot 2105

    elif problem_formulation.name == 'UTILITARIAN_DISAGGREGATED':
        n_objs = 3
        problem = Problem(n_decision_vars, n_objs)
        problem.directions[0] = Problem.MAXIMIZE  # welfare of utility 2105
        problem.directions[1] = Problem.MINIMIZE  # welfare of disutility 2105
        problem.directions[2] = Problem.MINIMIZE  # temperature overshoot 2105

    elif problem_formulation.name == 'EGALITARIAN_AGGREGATED':
        n_objs = 3
        problem = Problem(n_decision_vars, n_objs)
        problem.directions[0] = Problem.MAXIMIZE  # welfare of utility 2105
        problem.directions[1] = Problem.MINIMIZE  # intratemporal consumption Gini 2105
        problem.directions[2] = Problem.MINIMIZE  # temperature overshoot 2105

    elif problem_formulation.name == 'EGALITARIAN_DISAGGREGATED':
        n_objs = 5
        problem = Problem(n_decision_vars, n_objs)
        problem.directions[0] = Problem.MAXIMIZE  # welfare of utility 2105
        problem.directions[1] = Problem.MINIMIZE  # welfare of disutility 2105
        problem.directions[2] = Problem.MINIMIZE  # intratemporal consumption Gini 2105
        problem.directions[3] = Problem.MINIMIZE  # intratemporal damage Gini 2105
        problem.directions[4] = Problem.MINIMIZE  # temperature overshoot 2105

    elif problem_formulation.name == 'SUFFICIENTARIAN_AGGREGATED':
        n_objs = 4
        problem = Problem(n_decision_vars, n_objs)
        problem.directions[0] = Problem.MAXIMIZE  # welfare of utility 2105
        problem.directions[1] = Problem.MINIMIZE  # distance to consumption threshold 2105
        problem.directions[2] = Problem.MINIMIZE  # population below consumption 2105
        problem.directions[3] = Problem.MINIMIZE  # temperature overshoot 2105

    elif problem_formulation.name == 'SUFFICIENTARIAN_DISAGGREGATED':
        n_objs = 7
        problem = Problem(n_decision_vars, n_objs)
        problem.directions[0] = Problem.MAXIMIZE  # welfare of utility 2105
        problem.directions[1] = Problem.MINIMIZE  # welfare of disutility 2105
        problem.directions[2] = Problem.MINIMIZE  # distance to consumption threshold 2105
        problem.directions[3] = Problem.MINIMIZE  # population below consumption 2105
        problem.directions[4] = Problem.MINIMIZE  # distance to damage threshold 2105
        problem.directions[5] = Problem.MINIMIZE  # population above damage 2105
        problem.directions[6] = Problem.MINIMIZE  # temperature overshoot 2105

    elif problem_formulation.name == 'PRIORITARIAN_AGGREGATED':
        n_objs = 3
        problem = Problem(n_decision_vars, n_objs)
        problem.directions[0] = Problem.MAXIMIZE  # welfare of utility 2105
        problem.directions[1] = Problem.MAXIMIZE  # worst-off consumption per capita 2105
        problem.directions[2] = Problem.MINIMIZE  # temperature overshoot 2105

    elif problem_formulation.name == 'PRIORITARIAN_DISAGGREGATED':
        n_objs = 5
        problem = Problem(n_decision_vars, n_objs)
        problem.directions[0] = Problem.MAXIMIZE  # welfare of utility 2105
        problem.directions[1] = Problem.MAXIMIZE  # worst-off consumption per capita 2105
        problem.directions[2] = Problem.MINIMIZE  # welfare of disutility 2105
        problem.directions[3] = Problem.MINIMIZE  # worst-off damage per capita 2105
        problem.directions[4] = Problem.MINIMIZE  # temperature overshoot 2105

    else:
        raise ValueError('The function has not received a proper ProblemFormulation!')
        # problem = 0
        # n_objs = 0
        # n_decision_vars = 0

    return problem, n_objs, n_decision_vars


def _create_a_reference_set(problem, n_decision_vars, n_objs, hypervolume_folder, id):
    """
    Creates a reference set for hypervolume calculation
    @param problem: Platypus Problem
    @param n_decision_vars: int: number of decision variables
    @param n_objs: int: number of relevant objectives
    @param hypervolume_folder: string
    @param id: int: number of last saved archives
    @return:
        ref_set: list
    """
    directory = os.path.join(hypervolume_folder, f'archive_{id}.csv')
    data = pd.read_csv(directory, index_col=0).iloc[:, : n_decision_vars + n_objs]
    ref_set = []
    for i, row in data.iterrows():
        solution = Solution(problem)
        solution.objectives = row.values[n_decision_vars::]
        ref_set.append(solution)

    solution = Solution(problem)
    solution.objectives[:] = 0
    ref_set.append(solution)

    return ref_set


def _load_and_merge_archives(problem, n_decision_vars, n_objs, hypervolume_folder, id):
    """
    Loads and merges archive outcomes for a specific problem formulation
    @param problem: Platypus Problem
    @param n_decision_vars: int: number of decision variables
    @param n_objs: int: number of relevant objectives
    @param hypervolume_folder: String
    @param id: int: number of last saved archive
    @return:
        archives_dict: dictionary: merged archives
    """
    archives = []
    for i in range(2, id + 1):
        archive = pd.read_csv(os.path.join(hypervolume_folder, f'archive_{i}.csv'))
        nfe_column = [i * 1000 for _ in range(archive.shape[0])]
        archive.loc[:, "Unnamed: 0"] = nfe_column
        archive.rename(columns={"Unnamed: 0": "nfe"}, inplace=True)
        archives.append(archive)

    archives = pd.concat(archives)
    archives = archives.iloc[:, : n_decision_vars + n_objs + 1]

    archives_dict = {}
    for nfe, generation in archives.groupby("nfe"):
        archive = []
        for i, row in generation.iloc[:, n_decision_vars+1::].iterrows():  # maybe this number should be different
            solution = Solution(problem)
            solution.objectives = row
            archive.append(solution)
        archives_dict[nfe] = archive

    return archives_dict


def _compute_hypervolumes(ref_set, archives_dict):
    """
    @param ref_set: list as reference set
    @param archives_dict: dictionary: merged archives
    @return:
        nfes: int: list with nfes
        hvs: float: list with hypervolumes
    """
    hv = Hypervolume(reference_set=ref_set)
    nfes = []
    hvs = []
    for nfe, archive in archives_dict.items():
        nfes.append(nfe)
        hvs.append(hv.calculate(archive))

    return nfes, hvs


def _wrapper_hypervolume(folder_path, problem_formulation, searchover, nfe):
    """
    Compute hypervolume for one problem formulation
    @param problem_formulation: ProblemFormulation
    @return:
        nfes: int: list with nfes
        hvs: float: list with hypervolumes
    """
    problem, n_objs, n_decision_vars = _define_problem(problem_formulation)

    hypervolume_folder = os.path.join(
        folder_path,
        f'{problem_formulation.name}_{searchover}_{nfe}',
        'hypervolume'
    )

    id = _find_highest_archive_id(hypervolume_folder)

    ref_set = _create_a_reference_set(problem, n_decision_vars, n_objs, hypervolume_folder, id)
    archives = _load_and_merge_archives(problem, n_decision_vars, n_objs, hypervolume_folder, id)

    # print(archives)
    nfes, hvs = _compute_hypervolumes(ref_set, archives)

    # print(hvs)
    return nfes, hvs


def _find_highest_archive_id(hypervolume_folder):
    """
    Find the maximum archive id for a given problem formulation.
    @param hypervolume_folder: String: path to specific hypervolume folder
    @return:
        max_id: int
    """

    files = os.listdir(hypervolume_folder)
    ids = [int(x.split('_')[-1].split('.')[0]) for x in files]
    max_id = max(ids)

    return max_id


def _get_2d_coords(col, idx):
    """
    Transform 1d coordinate into 2d coordinates.
    @param col: int
    @param idx: int
    @return:
        coord: tuple with (x: int, y: int)
    """
    x = int(idx / col)
    y = idx % col
    coord = (x, y)

    return coord


def plot_hypervolumes(data_folder_path, searchover, nfe, saving=False):
    """
    Plot Hypervolume for all eight problem formulations.
    @param data_folder_path: String: path to the data folder in which all relevant data is saved
    @param searchover: String
    @param nfe: int: this indicates what the desired nfe is (has to exist in the results)
    @param saving: Boolean: whether to save image
    """

    # Preparing the plot
    sns.set(font_scale=1.35)
    sns.set_style("whitegrid")
    nrows = 2
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 14), tight_layout=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.8)

    # Computing Hypervolume for each problem formulation
    hypervolume_dict = {}
    for pf in ProblemFormulation.get_8_problem_formulations():
        hypervolume_dict[pf.name] = _wrapper_hypervolume(data_folder_path, pf, searchover, nfe)

    # Plotting
    for idx, (problem_formulation, values) in enumerate(hypervolume_dict.items()):

        coords = _get_2d_coords(ncols, idx)

        axes[coords].plot(*values)
        axes[coords].set_xlabel("nfe")
        axes[coords].set_ylabel("hypervolume")
        axes[coords].set_title(problem_formulation, fontsize=22)

    plt.show()

    # Saving
    if saving:
        directory = os.getcwd()
        root_directory = os.path.dirname(directory)
        visualization_folder = root_directory + "/optimization/outputimages/"
        fig.savefig(visualization_folder + "hypervolume.png", dpi=200, pad_inches=0.2)
