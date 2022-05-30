"""
This module is used to run directed policy search with several seeds.
"""


from optimization.general.directed_search import run_optimization
from optimization.scenariodiscovery.selection.scenario_selection import load_reference_scenarios
from model.enumerations import *
import random
import numpy as np

if __name__ == "__main__":

    seeds = [9845531, 1644652]
    problem_formulations = ProblemFormulation.get_util_and_suff_problem_formulations()
    reference_scenarios = load_reference_scenarios()

    for problem_formulation in problem_formulations:
        for seed_index, seed in enumerate(seeds):
            for reference_index, reference_scenario in enumerate(reference_scenarios):

                # Setting seeds
                random.seed(seed)
                np.random.seed(seed)

                # Run optimizations
                run_optimization(
                    problem_formulation=problem_formulation,
                    nfe=200000,
                    searchover='levers',
                    seed_index=seed_index,
                    reference=(reference_index, reference_scenario),
                    saving_results=True,
                    with_convergence=True,
                )
