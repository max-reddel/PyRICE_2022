"""
This module contains the climate part of the PyRICE model.
"""


import numpy as np


class ClimateModel:
    """
    This sub-model describes the carbon cycle part of the PyRICE model.
    """

    def __init__(self, steps, limits, regions_list):
        """
        @param steps: int (31)
        @param limits: ModelLimits
        @param regions_list: list with 12 regions as strings
        """

        self.limits = limits

        # Initial lower stratum temperature change [dC from 1900]
        self.tocean0 = 0.0068
        # Initial atmospheric temperature change [dC from 1900]
        self.tatm0 = 0.83

        # Climate equation coefficient for upper level
        self.c1 = 0.208
        # Transfer coefficient upper to lower stratum
        self.c3 = 0.310
        # Transfer coefficient for lower level
        self.c4 = 0.05
        # Climate model parameter
        # self.lam = self.carbon_model.fco22x / self.t2xco2

        # Increase temperature of atmosphere [dC from 1900]
        self.temp_atm = np.zeros(steps)
        # Increase temperature of lower oceans [dC from 1900]
        self.temp_ocean = np.zeros(steps)

        # Atmospheric temperature
        self.temp_atm[0] = self.tatm0
        self.temp_atm[1] = 0.980

        if self.temp_atm[0] < self.limits.temp_atm_lo:
            self.temp_atm[0] = self.limits.temp_atm_lo
        if self.temp_atm[0] > self.limits.temp_atm_up:
            self.temp_atm[0] = self.limits.temp_atm_up

        # Oceanic temperature
        self.temp_ocean[0] = 0.007

        if self.temp_ocean[0] < self.limits.temp_ocean_lo:
            self.temp_ocean[0] = self.limits.temp_ocean_lo
        if self.temp_ocean[0] > self.limits.temp_ocean_up:
            self.temp_ocean[0] = self.limits.temp_ocean_up

        # SLR parameters

        self.SLRTHERM = np.zeros(steps)
        self.THERMEQUIL = np.zeros(steps)

        self.GSICREMAIN = np.zeros(steps)
        self.GSICCUM = np.zeros(steps)
        self.GSICMELTRATE = np.zeros(steps)
        self.GISREMAIN = np.zeros(steps)
        self.GISMELTRATE = np.zeros(steps)
        self.GISEXPONENT = np.zeros(steps)
        self.GISCUM = np.zeros(steps)
        self.AISREMAIN = np.zeros(steps)
        self.AISMELTRATE = np.zeros(steps)
        self.AISCUM = np.zeros(steps)
        self.TOTALSLR = np.zeros(steps)

        # inputs
        self.therm0 = 0.092066694
        self.thermadj = 0.024076141
        self.thermeq = 0.5

        self.gsictotal = 0.26
        self.gsicmelt = 0.0008
        self.gsicexp = 1
        self.gsieq = -1

        self.gis0 = 7.3
        self.gismelt0 = 0.6
        self.gismeltabove = 1.118600816
        self.gismineq = 0
        self.gisexp = 1

        self.aismelt0 = 0.21
        self.aismeltlow = -0.600407185
        self.aismeltup = 2.225420209
        self.aisratio = 1.3
        self.aisinflection = 0
        self.aisintercept = 0.770332789
        self.aiswais = 5
        self.aisother = 51.6

        self.THERMEQUIL[0] = self.temp_atm[0] * self.thermeq
        self.SLRTHERM[0] = self.therm0 + self.thermadj * (
            self.THERMEQUIL[0] - self.therm0
        )

        self.GSICREMAIN[0] = self.gsictotal

        self.GSICMELTRATE[0] = (
            self.gsicmelt
            * 10
            * (self.GSICREMAIN[0] / self.gsictotal) ** self.gsicexp
            * (self.temp_atm[0] - self.gsieq)
        )
        self.GSICCUM[0] = self.GSICMELTRATE[0]
        self.GISREMAIN[0] = self.gis0
        self.GISMELTRATE[0] = self.gismelt0
        self.GISCUM[0] = self.gismelt0 / 100
        self.GISEXPONENT[0] = 1
        self.AISREMAIN[0] = self.aiswais + self.aisother
        self.AISMELTRATE[0] = 0.1225
        self.AISCUM[0] = self.AISMELTRATE[0] / 100

        self.TOTALSLR[0] = (
            self.SLRTHERM[0] + self.GSICCUM[0] + self.GISCUM[0] + self.AISCUM[0]
        )

        self.slrmultiplier = 2
        self.slrelasticity = 4

        self.SLRDAMAGES = np.zeros((len(regions_list), steps))
        self.slrdamlinear = np.array(
            [
                0,
                0.00452,
                0.00053,
                0,
                0.00011,
                0.01172,
                0,
                0.00138,
                0.00351,
                0,
                0.00616,
                0,
            ]
        )
        self.slrdamquadratic = np.array(
            [
                0.000255,
                0,
                0.000053,
                0.000042,
                0,
                0.000001,
                0.000255,
                0,
                0,
                0.000071,
                0,
                0.001239,
            ]
        )

        self.SLRDAMAGES[:, 0] = 0

    def run(self, t, fco22x, forc, t2xco2, Y_gross):
        """
        @param t: int: time step
        @param fco22x: float: forcing equilibrium
        @param forc: numpy array (31,): forcing
        @param t2xco2: float
        @param Y_gross: numpy array (12, 31): gross GDP
        @return:
            temp_atm: numpy array (31,: atmospheric temperature
        """
        # heating of oceans and atmospheric according to matrix equations
        if t > 1:
            self.temp_atm[t] = self.temp_atm[t - 1] + self.c1 * (
                (forc[t] - ((fco22x / t2xco2) * self.temp_atm[t - 1]))
                - (self.c3 * (self.temp_atm[t - 1] - self.temp_ocean[t - 1]))
            )

        # setting up lower and upper bound for temperatures
        if self.temp_atm[t] < self.limits.temp_atm_lo:
            self.temp_atm[t] = self.limits.temp_atm_lo

        if self.temp_atm[t] > self.limits.temp_atm_up:
            self.temp_atm[t] = self.limits.temp_atm_up

        self.temp_ocean[t] = self.temp_ocean[t - 1] + self.c4 * (
            self.temp_atm[t - 1] - self.temp_ocean[t - 1]
        )

        # setting up lower and upper bound for temperatures
        if self.temp_ocean[t] < self.limits.temp_ocean_lo:
            self.temp_ocean[t] = self.limits.temp_ocean_lo

        if self.temp_ocean[t] > self.limits.temp_ocean_up:
            self.temp_ocean[t] = self.limits.temp_ocean_up

        # thermal expansion
        self.THERMEQUIL[t] = self.temp_atm[t] * self.thermeq

        self.SLRTHERM[t] = self.SLRTHERM[t - 1] + self.thermadj * (
            self.THERMEQUIL[t] - self.SLRTHERM[t - 1]
        )

        # glacier ice cap
        self.GSICREMAIN[t] = self.gsictotal - self.GSICCUM[t - 1]

        self.GSICMELTRATE[t] = (
            self.gsicmelt
            * 10
            * (self.GSICREMAIN[t] / self.gsictotal) ** self.gsicexp
            * self.temp_atm[t]
        )

        self.GSICCUM[t] = self.GSICCUM[t - 1] + self.GSICMELTRATE[t]

        # greenland
        self.GISREMAIN[t] = self.GISREMAIN[t - 1] - (self.GISMELTRATE[t - 1] / 100)

        if t > 1:
            self.GISMELTRATE[t] = (
                self.gismeltabove * (self.temp_atm[t] - self.gismineq) + self.gismelt0
            ) * self.GISEXPONENT[t - 1]
        else:
            self.GISMELTRATE[1] = 0.60

        self.GISCUM[t] = self.GISCUM[t - 1] + self.GISMELTRATE[t] / 100

        if t > 1:
            self.GISEXPONENT[t] = 1 - (self.GISCUM[t] / self.gis0) ** self.gisexp
        else:
            self.GISEXPONENT[t] = 1

        # antartica ice cap
        if t <= 11:
            if self.temp_atm[t] < 3:
                self.AISMELTRATE[t] = (
                    self.aismeltlow * self.temp_atm[t] * self.aisratio
                    + self.aisintercept
                )
            else:
                self.AISMELTRATE[t] = (
                    self.aisinflection * self.aismeltlow
                    + self.aismeltup * (self.temp_atm[t] - 3.0)
                    + self.aisintercept
                )
        else:
            if self.temp_atm[t] < 3:
                self.AISMELTRATE[t] = (
                    self.aismeltlow * self.temp_atm[t] * self.aisratio + self.aismelt0
                )
            else:
                self.AISMELTRATE[t] = (
                    self.aisinflection * self.aismeltlow
                    + self.aismeltup * (self.temp_atm[t] - 3)
                    + self.aismelt0
                )

        self.AISCUM[t] = self.AISCUM[t - 1] + self.AISMELTRATE[t] / 100

        self.AISREMAIN[t] = self.AISREMAIN[0] - self.AISCUM[t]

        self.TOTALSLR[t] = (
            self.SLRTHERM[t] + self.GSICCUM[t] + self.GISCUM[t] + self.AISCUM[t]
        )

        self.SLRDAMAGES[:, t] = (
            100
            * self.slrmultiplier
            * (
                self.TOTALSLR[t - 1] * self.slrdamlinear
                + (self.TOTALSLR[t - 1] ** 2) * self.slrdamquadratic
            )
            * (Y_gross[:, t - 1] / Y_gross[:, 0]) ** (1 / self.slrelasticity)
        )

        return self.temp_atm
