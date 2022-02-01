# Problem Formulations for Optimization


## Outcomes and Epsilons
This folder contains a script `outcomes_and_epsilons.py` which contains a function to define the relevant outcomes and its epsilon-values for optimization. 


## Problem Formulations

The file `problem_formulation.py` contains the function `run_optimization` which can be used to run an optimization. Furthermore, the folder `problemformulations` contains four alternative problem formulations – each file name consist of the name of the damage function plus the ethical principle:

- `nordhaus_utilitarian.py`
- `nordhaus_sufficientarian.py`
- `weitzman_utilitarian.py`
- `weitzman_sufficientarian.py`

Run any of these scripts to run the corresponding optimization and decide whether to save the results and/or convergence data.

## Model Results

The results of the optimization processes are saved into the folder `results`. The csv files have a naming convention that combines information about the optimization with the following order:

- damage function name
- welfare function name
- results or convergence
- used number of function evaluations (nfe)


E.g., running the optimization with:

- `damage_function = DamageFunction.NORDHAUS`
- `welfare_function = WelfareFunction.UTILITARIAN`
- `nfe = 100000`
- `with_convergence = True`

This would result in two files:

- `NORDHAUS_UTILITARIAN_100000_results.csv`
- `NORDHAUS_UTILITARIAN_100000_convergence.csv`


## Visualizations

The folder `visualizations` contains several scripts and notebooks to visualize some results. E.g., `pathways.ipynb` is used to visualize the various pathways that result from the optimized policies. `convergence.ipynb` is used to visualize convergence data. 