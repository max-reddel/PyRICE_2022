"""
This module contains the class ModelLimits
"""


class ModelLimits:
    """
    This class contains the limits of the model.
    """
    def __init__(self):

        # Output low (constraints of the model)
        self.y_lo = 0.0
        self.ygross_lo = 0.0
        self.i_lo = 0.0
        self.c_lo = 2.0
        self.cpc_lo = 0
        self.k_lo = 1.0

        self.mat_lo = 10.0
        self.mu_lo = 100.0
        self.ml_lo = 1000.0
        self.temp_ocean_up = 20.0
        self.temp_ocean_lo = -1.0
        self.temp_atm_lo = 0.0

        # self.temp_atm_up = 20 or 12 for 2016 version
        self.temp_atm_up = 40.0
