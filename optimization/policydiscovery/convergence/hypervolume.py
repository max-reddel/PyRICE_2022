"""
Functions to load and plot hypervolume convergence.
"""

import os
from platypus import Solution, Problem
import pandas as pd
from platypus import Hypervolume
import matplotlib.pyplot as plt
import seaborn as sns

directory = os.getcwd()
directory_optimization = os.path.dirname(directory)
os.chdir(directory_optimization)


def define_problem(problem_formulation):
    """
    @param problem_formulation: String: combination of damage function and welfare function
    @return:
        problem: Platypus Problem
        n_objs: int: number of relevant objectives
        n_decision_vars: int: number of decision variables
    """
    if "SUFFICIENTARIAN" in problem_formulation:
        n_objs = 2
        n_decision_vars = 3
        problem = Problem(n_decision_vars, n_objs)

        problem.directions[0] = Problem.MAXIMIZE  # Total aggregated utility
        problem.directions[1] = Problem.MINIMIZE  # Distance to threshold 2035

    elif "UTILITARIAN" in problem_formulation:
        n_objs = 4
        n_decision_vars = 3

        problem = Problem(n_decision_vars, n_objs)
        problem.directions[0] = Problem.MAXIMIZE  # Total aggregated utility
        problem.directions[1] = Problem.MAXIMIZE  # Utility 2035
        problem.directions[2] = Problem.MAXIMIZE  # Utility 2055
        problem.directions[3] = Problem.MAXIMIZE  # Utility 2075

    else:
        problem = 0
        n_objs = 0
        n_decision_vars = 0

    return problem, n_objs, n_decision_vars


def create_a_reference_set(
    problem,
    n_decision_vars,
    n_objs,
    problem_formulation="WEITZMAN/SUFFICIENTARIAN",
    id=198,
):
    """
    Creates a reference set for hypervolume calculation
    @param problem: Platypus Problem
    @param n_decision_vars: int: number of decision variables
    @param n_objs: int: number of relevant objectives
    @param problem_formulation: string: combination of damage function and welfare function
    @param id: int: number of last saved archives
    @return:
        ref_set: list
    """
    data = pd.read_csv(
        directory_optimization
        + f"/results/hypervolume/{problem_formulation}/archive_{id}.csv",
        index_col=0,
    ).iloc[:, : n_decision_vars + n_objs]

    ref_set = []
    for i, row in data.iterrows():
        solution = Solution(problem)
        solution.objectives = row.values[n_decision_vars::]
        ref_set.append(solution)

    solution = Solution(problem)
    solution.objectives[:] = 0
    ref_set.append(solution)

    return ref_set


def load_and_merge_archives(
    problem,
    n_decision_vars,
    n_objs,
    problem_formulation="WEITZMAN/SUFFICIENTARIAN",
    id=198,
):
    """
    Loads and merges archive outcomes for a specific problem formulation
    @param problem: Platypus Problem
    @param n_decision_vars: int: number of decision variables
    @param n_objs: int: number of relevant objectives
    @param problem_formulation: string: combination of damage function and welfare function
    @param id: int: number of last saved archive
    @return:
        archives_dict: dictionary: merged archives
    """
    archives = []
    for i in range(2, id + 1):
        archive = pd.read_csv(
            f"./results/hypervolume/{problem_formulation}/archive_{i}.csv"
        )
        nfe_column = [i * 1000 for _ in range(archive.shape[0])]
        archive.loc[:, "Unnamed: 0"] = nfe_column
        archive.rename(columns={"Unnamed: 0": "nfe"}, inplace=True)
        archives.append(archive)

    archives = pd.concat(archives)
    archives = archives.iloc[:, : n_decision_vars + n_objs + 1]

    archives_dict = {}
    for nfe, generation in archives.groupby("nfe"):
        archive = []
        for i, row in generation.iloc[:, 4::].iterrows():
            solution = Solution(problem)
            solution.objectives = row
            archive.append(solution)
        archives_dict[nfe] = archive

    return archives_dict


def compute_hypervolumes(ref_set, archives_dict):
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


def wrapper_hypervolume(problem_formulation="WEITZMAN/SUFFICIENTARIAN", id=198):
    """
    Compute hypervolume for one problem formulation
    @param problem_formulation: string: combination of damage function and welfare function
    @param id: int: number of last saved archive
    @return:
        nfes: int: list with nfes
        hvs: float: list with hypervolumes
    """
    problem, n_objs, n_decision_vars = define_problem(problem_formulation)
    ref_set = create_a_reference_set(
        problem, n_decision_vars, n_objs, problem_formulation, id
    )
    archives = load_and_merge_archives(
        problem, n_decision_vars, n_objs, problem_formulation, id
    )
    nfes, hvs = compute_hypervolumes(ref_set, archives)

    return nfes, hvs


def plot_hypervolumes(hypervolume_dict, saving=False):
    """
    Plots hypervolume for all four problem formulations
    @param hypervolume_dict: dictionary containing tuples with (nfes, hvs)
    @param saving: Boolean: whether to save image
    """
    sns.set(font_scale=1.35)
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        nrows=1, ncols=4, sharex="all", figsize=(20, 4), tight_layout=True
    )
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.8
    )

    idx = 0
    for problem_formulation, values in hypervolume_dict.items():
        axes[idx].plot(*values)
        axes[idx].set_xlabel("nfe")
        axes[idx].set_ylabel("hypervolume")
        axes[idx].set_title(problem_formulation, fontsize=22)
        idx += 1

    # sns.despine()
    plt.show()

    if saving:
        directory = os.getcwd()
        root_directory = os.path.dirname(directory)
        visualization_folder = root_directory + "/optimization/outputimages/"
        fig.savefig(visualization_folder + "hypervolume.png", dpi=200, pad_inches=0.2)
