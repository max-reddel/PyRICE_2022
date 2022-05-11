"""
This module is used to run directed scenario search for the process of scenario discovery.
"""


from optimization.general.directed_search import run_optimization
from optimization.general.timer import *
from model.enumerations import *


if __name__ == "__main__":

    timer = Timer()

    problem_formulations = ProblemFormulation.get_8_problem_formulations()

    for idx, problem_formulation in enumerate(problem_formulations):

        print(
            f"Running problem formulation {idx+1}/{len(problem_formulations)} ({problem_formulation.name})"
        )

        run_optimization(
            problem_formulation=problem_formulation,
            nfe=200000,
            searchover="uncertainties",
            saving_results=True,
            with_convergence=True,
        )

    timer.stop()
