"""
Example script to run the PyRICE model with the Nordhaus policy.
"""

# Imports

from model.pyrice import PyRICE
from model.enumerations import *


# Specification of parameters

model_specification = ModelSpec.STANDARD
damage_function = DamageFunction.NORDHAUS
welfare_function = WelfareFunction.UTILITARIAN


# Instantiate model

model = PyRICE(
    model_specification=model_specification,
    damage_function=damage_function,
    welfare_function=welfare_function,
)

# Run model

# Specification of the policies
sr = 0.248
miu = 2135
irstp = 0.015

results = model(sr=sr, miu=miu, irstp_consumption=irstp)

# View results_formatted

# `results_formatted` is a dictionary containing all outcomes (including time series outcomes)
# This format is used to make it easier to run further optimizations.
# But there is a better way to view results_formatted from a simple simulation

# Better formated results_formatted
better_results = model.get_better_formatted_results()

# Highly Aggregated variables
print(f"aggregated_utility_gini: \t{better_results.aggregated_consumption_gini}")
print(f"aggregated_impact_gini: \t{better_results.aggregated_damage_gini}")
print(f"aggregated_utility: \t\t{better_results.aggregated_utility}")

# Dataframe on spatially aggregated variables of interest
print("\nDataframe on spatially aggregated variables of interest")
print(better_results.df_main.head())

# Dataframe on spatially and temporally disaggregated population
print("\nDataframe on spatially and temporally disaggregated population")
print(better_results.df_population.head())

# Dataframe on spatially disaggregated CPC
print("\nDataframe on spatially disaggregated CPC")
print(better_results.df_cpc.head())

# Dataframe on CPC pre damage
# Cell entries represent quintiles
print("\nDataframe on CPC pre damage")
print(better_results.df_cpc_pre_damage.head())

# Dataframe on CPC post damage
# Cell entries represent quintiles.
print("\nDataframe on CPC post damage")
print(better_results.df_cpc_post_damage.head())
