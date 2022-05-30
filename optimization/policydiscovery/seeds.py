"""
This module is used to run directed policy search with several seeds.
"""


from optimization.general.directed_search import run_optimization
from model.enumerations import *
import random
import numpy as np


if __name__ == "__main__":

    seeds = [9845531, 1644652]

    problem_formulation = ProblemFormulation.PRIORITARIAN_AGGREGATED

    for seed in seeds:

        # Setting seeds
        random.seed(seed)
        np.random.seed(seed)

        run_optimization(
            problem_formulation=problem_formulation,
            nfe=250000,
            searchover='levers',
            saving_results=True,
            with_convergence=True,
        )
