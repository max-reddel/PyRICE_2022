"""
This module contains the carbon cycle part of the PyRICE model.
"""

import numpy as np


class CarbonCycleModel:
    """
    This sub-model describes the carbon cycle part of the PyRICE model.
    """

    def __init__(self, steps, limits):
        """
        @param steps: int (31)
        @param limits: ModelLimits
        """

        self.limits = limits

        # RICE2010 INPUTS
        # Initial concentration in atmosphere 2000 [GtC]
        self.mat0 = 787
        # Initial concentration in atmosphere 2010 [GtC]
        self.mat1 = 829
        # Initial concentration in upper strata [GtC]
        self.mu0 = 1600.0  # 1600 in excel
        # Initial concentration in lower strata [GtC]
        self.ml0 = 10010.0
        # Equilibrium concentration in atmosphere [GtC]
        self.mateq = 588.0
        # Equilibrium concentration in upper strata [GtC]
        self.mueq = 1500.0
        # Equilibrium concentration in lower strata [GtC]
        self.mleq = 10000.0

        self.b11 = 0.088
        self.b23 = 0.00500
        self.b12 = 1 - self.b11
        self.b21 = self.b11 * self.mateq / self.mueq
        self.b22 = 1 - self.b21 - self.b23
        self.b32 = self.b23 * self.mueq / self.mleq
        self.b33 = 1 - self.b32

        # 2000 forcings of non-CO2 greenhouse gases (GHG) [Wm-2]
        self.fex0 = -0.06
        # 2100 forcings of non-CO2 GHG [Wm-2]
        self.fex1 = 0.30
        # Forcings of equilibrium CO2 doubling [Wm-2]
        self.fco22x = 3.8

        self.mat = np.zeros((steps,))
        self.mu = np.zeros((steps,))
        self.ml = np.zeros((steps,))
        self.forcoth = np.zeros((steps,))
        self.forc = np.zeros((steps,))

        # Carbon pools
        self.mat[0] = self.mat0
        self.mat[1] = self.mat1

        if self.mat[0] < self.limits.mat_lo:
            self.mat[0] = self.limits.mat_lo

        self.mu[0] = self.mu0
        if self.mu[0] < self.limits.mu_lo:
            self.mu[0] = self.limits.mu_lo

        self.ml[0] = self.ml0
        if self.ml[0] < self.limits.ml_lo:
            self.ml[0] = self.limits.ml_lo

        # Radiative forcing
        self.forcoth[0] = self.fex0
        self.forc[0] = (
            self.fco22x
            * (np.log(((self.mat[0] + self.mat[1]) / 2) / 596.40) / np.log(2.0))
            + self.forcoth[0]
        )

    def run(self, t, E):
        """
        @param t: int: time step
        @param E: numpy array (12, 31): emissions
        @return:
            self.fco22x: float: forcings of equilibrium CO2 doubling
            self.forc: numpy array (31,): forcing
            E_worldwilde_per_year: numpy array (31,): global annual emissions

        """
        # Carbon concentration increase in atmosphere [GtC from 1750]
        E_worldwilde_per_year = E.sum(axis=0)

        # calculate concentration in bioshpere and upper oceans
        self.mu[t] = (
            12 / 100 * self.mat[t - 1]
            + 94.796 / 100 * self.mu[t - 1]
            + 0.075 / 100 * self.ml[t - 1]
        )

        # set lower constraint for shallow ocean concentration
        if self.mu[t] < self.limits.mu_lo:
            self.mu[t] = self.limits.mu_lo

        # Carbon concentration increase in lower oceans [GtC from 1750]
        self.ml[t] = 99.925 / 100 * self.ml[t - 1] + 0.5 / 100 * self.mu[t - 1]

        # set lower constraint for shallow ocean concentration
        if self.ml[t] < self.limits.ml_lo:
            self.ml[t] = self.limits.ml_lo

        # calculate concentration in atmosphere for t + 1 (because of averaging in forcing formula
        if t < 30:
            self.mat[t + 1] = (
                88 / 100 * self.mat[t]
                + 4.704 / 100 * self.mu[t]
                + E_worldwilde_per_year[t] * 10
            )

            # set lower constraint for atmospheric concentration
            if self.mat[t + 1] < self.limits.mat_lo:
                self.mat[t + 1] = self.limits.mat_lo

        # Radiative forcing
        # Exogenous forcings from other GHG
        # rises linearly from 2010 to 2100 from -0.060 to 0.3 then becomes stable in RICE -  UPDATE FOR DICE2016R

        exo_forcing_2000 = -0.060
        exo_forcing_2100 = 0.3000

        if t < 11:
            self.forcoth[t] = (
                self.fex0 + 0.1 * (exo_forcing_2100 - exo_forcing_2000) * t
            )
        else:
            self.forcoth[t] = exo_forcing_2100

        # Increase in radiative forcing [Wm-2 from 1900]
        # forcing = constant * Log2( current concentration / concentration of forcing in 1900 at a
        # doubling of CO2 (η)[◦C/2xCO2] ) + external forcing
        if t < 30:
            self.forc[t] = (
                self.fco22x
                * (
                    np.log(((self.mat[t] + self.mat[t + 1]) / 2) / (280 * 2.13))
                    / np.log(2.0)
                )
                + self.forcoth[t]
            )
        if t == 30:
            self.forc[t] = (
                self.fco22x * (np.log((self.mat[t]) / (280 * 2.13)) / np.log(2.0))
                + self.forcoth[t]
            )

        return self.fco22x, self.forc, E_worldwilde_per_year
