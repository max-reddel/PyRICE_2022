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


def plot_epsilon_progress(
        data_folder_path,
        searchover,
        problem_formulations=None,
        nfe=200000,
        n_references=4,
        n_seeds=4,
        saving=False,
        file_name=None
):
    """
    Plot the epsilon progress of all eight problem formulations.
    @param data_folder_path: String: path to the data folder in which all relevant data is saved
    @param searchover: String: {'levers', 'uncertainties'}
    @param problem_formulations: list with ProblemFormulation.name strings
    @param nfe: int: this indicates what the desired nfe is (has to exist in the results)
    @param n_references: int: how many reference scenarios or policies have been used
    @param n_seeds: int: how many seeds have been used
    @param saving: Boolean: whether to save image
    @param file_name: String
    """

    if problem_formulations is None:
        problem_formulations = [pf.name for pf in ProblemFormulation.get_util_and_suff_problem_formulations()]
    else:
        problem_formulations = [pf.name for pf in problem_formulations]

    # Set parameters
    y_label = "$\epsilon$-progress"
    x_label = "nfe"

    # Create subplots
    sns.set(font_scale=1.35)
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        nrows=len(problem_formulations),
        ncols=n_references,
        figsize=(24, 4*len(problem_formulations)),
        tight_layout=True
    )
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    fig.patch.set_facecolor("white")

    reference_name = 'reference_scenario' if searchover == 'levers' else 'reference_policy'

    # Seeds and colors
    color_mapping = {}
    unique_seeds = list(set(list(range(n_seeds))))
    for _, (seed, color) in enumerate(zip(unique_seeds, sns.color_palette())):
        color_mapping[seed] = color

    axes_font_size = 20

    # Load epsilon values for each problem formulation, seed, and reference
    for pf_idx, problem_formulation in enumerate(problem_formulations):
        for seed_idx in range(n_seeds):
            for reference_idx in range(n_references):

                # Define path name
                directory = os.path.join(
                    data_folder_path,
                    f'{problem_formulation}_{nfe}',
                    f'seed_{seed_idx}',
                    f'{reference_name}_{reference_idx}',
                    'epsilon_progress.csv'
                )

                # If statement because incomplete data for ED
                if not ((reference_idx == 2 or reference_idx == 3) and seed_idx == 1 and problem_formulation == 'EGALITARIAN_DISAGGREGATED'):

                    # load dataframe
                    df = pd.read_csv(directory)

                    # Plotting
                    axes[pf_idx, reference_idx].plot(
                        df.nfe,
                        df.epsilon_progress,
                        color=color_mapping[seed_idx],
                        label=f'seed {seed_idx}'
                    )
                    axes[pf_idx, reference_idx].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    axes[pf_idx, reference_idx].set_ylabel(y_label, fontsize=axes_font_size)
                    clean_reference_name = ' '.join(reference_name.split('_'))
                    fig_title = f'{clean_reference_name} {reference_idx}'
                    axes[pf_idx, reference_idx].set_title(fig_title, fontsize=22)
                    axes[pf_idx, reference_idx].set_xlabel(x_label, fontsize=axes_font_size)

                    axes[pf_idx, reference_idx].tick_params(axis='both', which='minor', labelsize=8)

    # Splitting strings for better readability
    # problem_formulations = ['\n'.join(pf.split('_')) for pf in problem_formulations]
    new_problem_formulations = []
    for pf in problem_formulations:
        terms = pf.split('_')
        new_pf = r'$%s_{%s}$' % (terms[0][0], terms[1][0])
        new_problem_formulations.append(new_pf)

    # Annotation with row names
    for ax, row in zip(axes[:, 0], new_problem_formulations):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    ha='right', va='center', fontsize=25)

    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # fig.suptitle(f'Convergence with {y_label}', y=1.05, fontsize=25)
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 1.02),
        loc='upper center',
        ncol=len(color_mapping)
    )

    plt.show()

    if saving:
        if file_name is None:
            file_name = 'convergence_epsilon_progress'
        file_name += '.png'
        directory = os.path.join(
            os.path.dirname(os.path.dirname(os.getcwd())),
            'outputimages',
            'optimalpolicies',
            file_name
        )
        fig.savefig(directory, dpi=150, pad_inches=0.2, bbox_inches='tight')


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


def _wrapper_hypervolume(hypervolume_path, problem_formulation):
    """
    Compute hypervolume for one problem formulation
    @param problem_formulation: ProblemFormulation
    @return:
        nfes: int: list with nfes
        hvs: float: list with hypervolumes
    """
    problem, n_objs, n_decision_vars = _define_problem(problem_formulation)

    id = _find_highest_archive_id(hypervolume_path)

    ref_set = _create_a_reference_set(problem, n_decision_vars, n_objs, hypervolume_path, id)
    archives = _load_and_merge_archives(problem, n_decision_vars, n_objs, hypervolume_path, id)

    nfes, hvs = _compute_hypervolumes(ref_set, archives)

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


def plot_hypervolumes(
        data_folder_path,
        searchover,
        nfe,
        problem_formulations=None,
        n_seeds=1,
        n_references=1,
        saving=False,
        file_name=None
):
    """
    Plot Hypervolume for all eight problem formulations.
    @param data_folder_path: String: path to the data folder in which all relevant data is saved
    @param searchover: String
    @param nfe: int: this indicates what the desired nfe is (has to exist in the results)
    @param problem_formulations: list with ProblemFormulation objects
    @param n_seeds: int: how many seeds have been used
    @param n_references: int: how many reference scenarios have been used
    @param saving: Boolean: whether to save image
    @param file_name: String
    """

    if problem_formulations is None:
        problem_formulations = ProblemFormulation.get_util_and_suff_problem_formulations()

    # Preparing the plot
    sns.set(font_scale=1.35)
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(
        nrows=len(problem_formulations),
        ncols=n_references,
        figsize=(24, 4*len(problem_formulations)),
        tight_layout=True
    )
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.8)
    fig.suptitle('Convergence with Hypervolume', y=1.05, fontsize=25)

    # Splitting strings for better readability
    clean_problem_formulations = ['\n'.join(pf.name.split('_')) for pf in problem_formulations]

    # Annotation with row names
    for ax, row in zip(axes[:, 0], clean_problem_formulations):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    reference_name = 'reference_scenario' if searchover == 'levers' else 'reference_policy'

    # Seeds and colors
    color_mapping = {}
    unique_seeds = list(set(list(range(n_seeds))))
    for _, (seed, color) in enumerate(zip(unique_seeds, sns.color_palette())):
        color_mapping[seed] = color

    # Load epsilon values for each problem formulation, seed, and reference
    for pf_idx, problem_formulation in enumerate(problem_formulations):
        for seed_idx in range(n_seeds):
            for reference_idx in range(n_references):

                # Define path name
                directory = os.path.join(
                    data_folder_path,
                    f'{problem_formulation.name}_{nfe}',
                    f'seed_{seed_idx}',
                    f'{reference_name}_{reference_idx}',
                    'hypervolume'
                )

                # Compute single hypervolume
                values = _wrapper_hypervolume(directory, problem_formulation)

                # Plotting
                axes[pf_idx, reference_idx].plot(
                    *values,
                    color=color_mapping[seed_idx],
                    label=f'seed {seed_idx}'
                )
                axes[pf_idx, reference_idx].set_xlabel('nfe')
                axes[pf_idx, reference_idx].set_ylabel('hypervolume')
                clean_reference_name = ' '.join(reference_name.split('_'))
                fig_title = f'{clean_reference_name} {reference_idx}'
                axes[pf_idx, reference_idx].set_title(fig_title, fontsize=22)

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 1.0),
        loc='upper center',
        ncol=len(color_mapping)
    )

    plt.show()

    # Saving
    if saving:
        directory = os.getcwd()
        root_directory = os.path.dirname(directory)
        visualization_folder = os.path.join(root_directory, 'dmdu', 'outputimages')
        if file_name is None:
            file_name = 'hypervolume'
        file_name += '.png'
        fig.savefig(os.path.join(visualization_folder, file_name), dpi=200, pad_inches=0.2)
