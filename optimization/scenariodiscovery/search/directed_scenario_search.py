"""
This module is used to run directed scenario search for the process of scenario discovery.
"""


from optimization.general.directed_search import run_optimization
from model.enumerations import *


if __name__ == '__main__':

    run_optimization(
        problem_formulation=ProblemFormulation.UTILITARIAN_AGGREGATED,
        nfe=2,
        searchover='uncertainties',
        saving_results=True,
        with_convergence=True
    )
