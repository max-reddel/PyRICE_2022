import os


# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

# Change the current working directory
os.chdir('/Users/palokbiswas/Library/CloudStorage/OneDrive-Personal/AI_NL_PhD_2022/CURRENT_22-23-RICE/PyRICE_Max_New/PyRICE_2022') #change this

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

from dmdu.general.directed_search import run_optimization
from model.enumerations import *
import random
import numpy as np
import pandas as pd
from dmdu.scenariodiscovery.selection.scenario_selection import load_reference_scenarios
from dmdu.general.xlm_constants_epsilons import get_lever_names
from ema_workbench import Policy


if __name__ == "__main__":

    seeds = [9845531, 1644652]
    problem_formulations = [
        ProblemFormulation.UTILITARIAN_AGGREGATED,
        ProblemFormulation.SUFFICIENTARIAN_AGGREGATED
    ]
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
                    nfe=100,  #200000
                    searchover='levers',
                    seed_index=seed_index,
                    reference=(reference_index, reference_scenario),
                    saving_results=True,
                    with_convergence=True
                )
