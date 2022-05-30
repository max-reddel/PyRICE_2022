"""
This module is used to run directed policy search with several seeds.
"""


from dmdu.general.directed_search import run_optimization
from dmdu.scenariodiscovery.selection.scenario_selection import load_reference_scenarios
from model.enumerations import *
import random
import numpy as np


if __name__ == "__main__":

    seeds = [9845531, 1644652]
    # problem_formulations = ProblemFormulation.get_util_and_suff_problem_formulations()
    problem_formulations = [ProblemFormulation.PRIORITARIAN_DISAGGREGATED]
    reference_scenarios = load_reference_scenarios()

    for problem_formulation in problem_formulations:
        for seed_index, seed in enumerate(seeds):
            for reference_index, reference_scenario in enumerate(reference_scenarios):

                if reference_index == 2:
                    break

                # Setting seeds
                random.seed(seed)
                np.random.seed(seed)

                # Run optimizations
                run_optimization(
                    problem_formulation=problem_formulation,
                    # nfe=200000,
                    nfe=10,
                    searchover='levers',
                    seed_index=seed_index,
                    reference=(reference_index, reference_scenario),
                    saving_results=True,
                    with_convergence=True,
                )
