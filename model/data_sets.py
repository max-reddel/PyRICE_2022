"""
This module contains the class DataSets
"""
import os

import pandas as pd


class DataSets:
    """
    This class loads all the relevant outcomes from the folder inputdata.
    """

    def __init__(self):

        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'inputdata')

        # Load in RICE input parameters for all regions
        self.RICE_DATA = pd.read_excel(os.path.join(directory, "RICE_data.xlsx"))
        self.RICE_PARAMETER = pd.read_excel(os.path.join(directory, "RICE_parameter.xlsx"))
        self.RICE_input = pd.read_excel(os.path.join(directory, "input_data_RICE.xlsx"))
        self.RICE_regional_damage_factor = \
            pd.read_csv(os.path.join(directory, "regional damage frac factor RICE.csv"))
        self.RICE_regional_damage_factor = self.RICE_regional_damage_factor.iloc[:, 1:].to_numpy()

        # import World Bank income shares
        self.RICE_income_shares = pd.read_excel(os.path.join(directory, "RICE_income_shares.xlsx"))
        self.RICE_income_shares = self.RICE_income_shares.iloc[:, 1:6].to_numpy()

        # import dataframes for SSP (IPCC) uncertainty analysis
        self.RICE_GDP_SSP = pd.read_excel(os.path.join(directory, "Y_Gross_ssp.xlsx")).to_numpy()
        self.POP_ssp = pd.read_excel(os.path.join(directory, "pop_ssp.xlsx"), header=None)
        self.GDP_ssp = pd.read_excel(os.path.join(directory, "Y_Gross_ssp.xlsx"), header=None)
