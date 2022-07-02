"""
This file is used prepare the data input for the pareto.py file. The main function will go through the optimization
solutions, put them into the output folder and give them a name that relates to the problem formulation.
"""


import os
import pandas as pd
from model.enumerations import ProblemFormulation


if __name__ == '__main__':

    # directory where optimization results have been saved
    results_directory = os.path.join(
        os.path.dirname(os.getcwd()),
        'data',
    )

    # directory where those results will be renamed and saved to
    saving_directory = os.path.join(
        os.getcwd(),
        'data',
        'input'
    )

    problem_formulations = [
        # ProblemFormulation.SUFFICIENTARIAN_AGGREGATED,
        # ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED,
        # ProblemFormulation.UTILITARIAN_AGGREGATED,
        # ProblemFormulation.UTILITARIAN_DISAGGREGATED
        # ProblemFormulation.EGALITARIAN_AGGREGATED,
        # ProblemFormulation.EGALITARIAN_DISAGGREGATED,
        ProblemFormulation.PRIORITARIAN_AGGREGATED,
        ProblemFormulation.PRIORITARIAN_DISAGGREGATED
    ]

    # Parameters that refer to the optimization
    nfe = 200000
    n_seeds = 2
    n_references = 4

    for problem_formulation in problem_formulations:
        idx = 0  # for naming each file

        for reference_idx in range(n_references):

            df_main = None

            for seed_idx in range(n_seeds):

                # if reference_idx == 2 and seed_idx == 1 and problem_formulation == ProblemFormulation.EGALITARIAN_DISAGGREGATED:
                #     raise StopIteration

                problem_folder = f'{problem_formulation.name}_{nfe}'
                seed_folder = f'seed_{seed_idx}'
                reference_folder = f'reference_scenario_{reference_idx}'

                current_directory = os.path.join(
                    results_directory,
                    problem_folder,
                    seed_folder,
                    reference_folder,
                    'results.csv'
                )

                df = pd.read_csv(current_directory)

                if df_main is None:
                    df_main = df
                else:
                    df_main = pd.concat([df_main, df])

            df_main.index = list(range(len(df_main)))
            df_main = df_main.drop(columns=['Unnamed: 0'])
            df_main.to_csv(os.path.join(saving_directory, f'{problem_formulation.name}_{reference_idx}.csv'))
            idx += 1
