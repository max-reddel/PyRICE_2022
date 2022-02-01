"""
This module contains the utility model class and the results class.
"""


import numpy as np
from model.enumerations import WelfareFunction
import pandas as pd


class UtilityModel:
    """
    This sub-model describes the utility part of the PyRICE model.
    """

    def __init__(self, steps, data_sets, regions_list):
        """
        @param steps: int (31)
        @param data_sets: DataSets
        @param regions_list: list with 12 regions
        """

        self.n_regions = len(regions_list)
        self.regions_list = regions_list

        # output metrics
        self.util_sdr = np.zeros((self.n_regions, steps))
        self.inst_util = np.zeros((self.n_regions, steps))
        self.per_util = np.zeros((self.n_regions, steps))

        self.cum_util = np.zeros((self.n_regions, steps))
        self.reg_cum_util = np.zeros((self.n_regions, steps))
        self.reg_util = np.zeros(self.n_regions)
        self.util = np.zeros((self.n_regions, steps))

        self.per_util_ww = np.zeros((self.n_regions, steps))
        self.cum_per_util = np.zeros((self.n_regions, steps))
        self.inst_util_ww = np.zeros((self.n_regions, steps))

        # alternative WelfareFunction output arrays
        self.sufficientarian_threshold = np.zeros(steps)
        self.inst_util_tres = np.zeros(steps)
        self.inst_util_tres_ww = np.zeros((self.n_regions, steps))

        # Elasticity of marginal utility of consumption (1.45)
        self.elasmu = 1.50

        # Get scaling data_dict for welfare weights and aggregated utility
        self.Alpha_data = data_sets.RICE_DATA.iloc[357:369, 1:60].to_numpy()
        self.additative_scaling_weights = data_sets.RICE_DATA.iloc[167:179, 14:17].to_numpy()
        self.multiplutacive_scaling_weights = data_sets.RICE_DATA.iloc[232:244, 1:2].to_numpy() / 1000

        # Ooutcome variables

        # Dictionaries for quintile outputs
        self.quintile_inst_util = {}
        self.quintile_inst_util_ww = {}
        self.quintile_inst_util_concave = {}
        self.quintile_per_util_ww = {}

        # Utilitarian outputs
        self.global_damages = np.zeros(steps)
        self.global_ouput = np.zeros(steps)
        self.global_per_util_ww = np.zeros(steps)
        self.regional_cum_util = np.zeros(steps)

        # Prioritarian outputs
        self.inst_util_worst_off = np.zeros((self.n_regions, steps))
        self.inst_util_worst_off_condition = np.zeros((self.n_regions, steps))
        self.worst_off_income_class = np.zeros(steps)
        self.worst_off_income_class_index = np.zeros(steps)
        self.worst_off_climate_impact = np.zeros(steps)
        self.worst_off_climate_impact_index = np.zeros(steps)
        # self.climate_impact_relative_to_capita = {}

        # Sufficientarian outputs
        self.average_world_CPC = np.zeros(steps)
        self.average_growth_CPC = np.zeros(steps)
        self.sufficientarian_threshold = np.zeros(steps)
        self.inst_util_tres = np.zeros(steps)
        self.inst_util_tres_ww = np.zeros((self.n_regions, steps))
        self.quintile_inst_util = {}
        self.quintile_inst_util_ww = {}
        self.population_under_threshold = np.zeros(steps)
        self.utility_distance_threshold = np.zeros((self.n_regions, steps))
        self.max_utility_distance_threshold = np.zeros(steps)
        self.regions_under_threshold = [None] * steps
        self.largest_distance_under_threshold = np.zeros(steps)
        self.growth_frontier = np.zeros(steps)

        # Egalitarian outputs
        self.CPC_intra_gini = np.zeros(steps)
        self.average_world_CPC = np.zeros(steps)
        self.average_regional_impact = np.zeros(steps)
        self.climate_impact_per_dollar_consumption = np.zeros((self.n_regions, steps))
        self.climate_impact_per_dollar_gini = np.zeros(steps)

    def set_up_utility(self, regions_list, ini_suf_threshold, climate_impact_relative_to_capita, CPC_post_damage, CPC,
                       region_pop, damages, Y):
        """
        @param regions_list: list with 12 regions
        @param ini_suf_threshold: float
        @param climate_impact_relative_to_capita: dictionary
        @param CPC_post_damage: dictionary
        @param CPC: numpy array (12, 31)
        @param region_pop: numpy array (12, 31)
        @param damages: numpy array (12, 31)
        @param Y: numpy array (12, 31)
        """

        self.regions_list = regions_list

        # Initial rate of social time preference per year
        self.util_sdr[:, 0] = 1

        # Instantaneous utility function equation
        self.inst_util[:, 0] = ((1 / (1 - self.elasmu)) * (CPC[:, 0]) ** (1 - self.elasmu) + 1)

        # CEMU period utility
        self.per_util[:, 0] = self.inst_util[:, 0] * region_pop[:, 0] * self.util_sdr[:, 0]

        # Cummulative period utility without WW
        self.cum_per_util[:, 0] = self.per_util[:, 0]

        # Instantaneous utility function with welfare weights
        self.inst_util_ww[:, 0] = self.inst_util[:, 0] * self.Alpha_data[:, 0]

        # Period utility with welfare weights
        self.per_util_ww[:, 0] = self.inst_util_ww[:, 0] * region_pop[:, 0] * self.util_sdr[:, 0]

        # cummulative utility with ww
        self.reg_cum_util[:, 0] = self.per_util[:, 0]
        self.global_per_util_ww[0] = self.per_util_ww[:, 0].sum(axis=0)

        # initialise objectives for principles
        # objective for the worst-off region in terms of consumption per capita
        self.worst_off_income_class[0] = CPC_post_damage[2005][0].min()

        array_worst_off_income = CPC_post_damage[2005][0]
        self.worst_off_income_class_index[0] = np.argmin(array_worst_off_income)

        # objective for the worst-off region in terms of climate impact
        self.worst_off_climate_impact[0] = climate_impact_relative_to_capita[2005][0].min()

        array_worst_off_share = climate_impact_relative_to_capita[2005][0]
        self.worst_off_climate_impact_index[0] = np.argmin(array_worst_off_share)

        # objectives sufficientarian
        self.average_world_CPC[0] = CPC[:, 0].sum() / self.n_regions
        self.average_growth_CPC[0] = 0.250  # average growth over 10 years World Bank Data

        # calculate instantaneous welfare equivalent of minimum capita per head
        self.sufficientarian_threshold[0] = ini_suf_threshold  # specified in consumption per capita thousand/year

        self.inst_util_tres[0] = ((1 / (1 - self.elasmu)) * (self.sufficientarian_threshold[0]) ** (1 - self.elasmu) + 1)

        # calculate instantaneous welfare equivalent of minimum capita per head with PPP
        self.inst_util_tres_ww[:, 0] = self.inst_util_tres[0] * self.Alpha_data[:, 0]

        # calculate utility equivalent for every income quintile and scale with welfare weights for comparison
        self.quintile_inst_util[2005] = (
                (1 / (1 - self.elasmu)) * (CPC_post_damage[2005]) ** (1 - self.elasmu) + 1)
        self.quintile_inst_util_ww[2005] = self.quintile_inst_util[2005] * self.Alpha_data[:, 0]

        utility_per_income_share = self.quintile_inst_util_ww[2005]

        list_timestep = []

        for quintile in range(0, 5):
            for region in range(0, self.n_regions):
                if utility_per_income_share[quintile, region] < self.inst_util_tres_ww[region, 0]:
                    self.population_under_threshold[0] = self.population_under_threshold[0] + \
                                                         region_pop[region, 0] * 1 / 5
                    self.utility_distance_threshold[region, 0] = self.inst_util_tres_ww[region, 0] - \
                                                                 utility_per_income_share[quintile, region]

                    list_timestep.append(regions_list[region])

        self.regions_under_threshold[0] = list_timestep
        self.max_utility_distance_threshold[0] = self.utility_distance_threshold[:, 0].max()

        # calculate gini as measure of current inequality in consumption (intragenerational)
        input_gini = CPC[:, 0]

        diffsum = 0
        for i, xi in enumerate(input_gini[:-1], 1):
            diffsum += np.sum(np.abs(xi - input_gini[i:]))

        self.CPC_intra_gini[0] = diffsum / ((len(input_gini) ** 2) * np.mean(input_gini))

        # calculate gini as measure of current inequality in climate impact (per dollar consumption)
        # (intragenerational)
        self.climate_impact_per_dollar_consumption[:, 0] = damages[:, 0] / CPC[:, 0]
        input_gini = self.climate_impact_per_dollar_consumption[:, 0]

        diffsum = 0
        for i, xi in enumerate(input_gini[:-1], 1):
            diffsum += np.sum(np.abs(xi - input_gini[i:]))

        self.climate_impact_per_dollar_gini[0] = diffsum / ((len(input_gini) ** 2) * np.mean(input_gini))

        self.global_damages[0] = damages[:, 0].sum(axis=0)
        self.global_ouput[0] = Y[:, 0].sum(axis=0)

    def run(self, t, year, welfare_function, irstp, tstep, growth_factor_prio, growth_factor_suf,
            sufficientarian_discounting, egalitarian_discounting, prioritarian_discounting, regions_list,
            CPC, region_pop, damages, Y, CPC_lo, climate_impact_relative_to_capita, CPC_post_damage):
        """
        @param t: int
        @param year: int
        @param welfare_function: WelfareFunction
        @param irstp: float
        @param tstep: int (10)
        @param growth_factor_prio: int
        @param growth_factor_suf: int
        @param sufficientarian_discounting: int
        @param egalitarian_discounting: int
        @param prioritarian_discounting: int
        @param regions_list: list with 12 regions
        @param CPC: numpy array (12, 31)
        @param region_pop: numpy array (12, 31)
        @param damages: numpy array (12, 31)
        @param Y: numpy array (12, 31)
        @param CPC_lo: float
        @param climate_impact_relative_to_capita: dictionary
        @param CPC_post_damage: dictionary
        @return:
            CPC: numpy array (12, 31)
            CPC_post_damage: dictionary

        """

        self.welfare_function = welfare_function

        if self.welfare_function.__eq__(WelfareFunction.UTILITARIAN):

            self. run_utilitarian(t, year, irstp, tstep, CPC, region_pop, damages, Y, CPC_lo,
                                  climate_impact_relative_to_capita, CPC_post_damage)

        elif self.welfare_function.__eq__(WelfareFunction.PRIORITARIAN):

            self.run_prioritarian(t, year, irstp, tstep, growth_factor_prio, prioritarian_discounting, CPC, region_pop,
                                  damages, Y, CPC_lo, climate_impact_relative_to_capita, CPC_post_damage)

        elif self.welfare_function.__eq__(WelfareFunction.SUFFICIENTARIAN):

            self.run_sufficientarian(t, year, irstp, tstep, growth_factor_suf, sufficientarian_discounting,
                                     CPC, region_pop, damages, Y, CPC_lo, climate_impact_relative_to_capita,
                                     CPC_post_damage)

        elif self.welfare_function.__eq__(WelfareFunction.EGALITARIAN):

            self.run_egalitarian(t, year, irstp, tstep, egalitarian_discounting, CPC, region_pop, damages, Y, CPC_lo,
                                 climate_impact_relative_to_capita, CPC_post_damage)

        else:
            raise ValueError('Oh, no! Apparently, your welfare function is unknown!')

        return CPC, CPC_post_damage

    def run_utilitarian(self, t, year, irstp, tstep, CPC, region_pop, damages, Y, CPC_lo,
                        climate_impact_relative_to_capita, CPC_post_damage):
        """
        @param t: int
        @param year: int
        @param irstp: float
        @param tstep: int
        @param CPC: numpy array (12, 31)
        @param region_pop: numpy array (12, 31)
        @param damages:numpy array (12, 31)
        @param Y: numpy array (12, 31)
        @param CPC_lo: float
        @param climate_impact_relative_to_capita: dictionary
        @param CPC_post_damage: dictionary
        """

        self.set_up_weights_related(t, irstp, tstep, CPC, region_pop)

        # period utility with welfare weights
        self.per_util_ww[:, t] = self.inst_util_ww[:, t] * region_pop[:, t] * self.util_sdr[:, t]
        self.regional_cum_util[t] = self.reg_cum_util[:, t].sum()

        self.calculate_alternative_principles_objectives(t, year, CPC, damages, CPC_post_damage, CPC_lo, region_pop,
                                                         self.welfare_function, climate_impact_relative_to_capita, Y)

    def run_prioritarian(self, t, year, irstp, tstep, growth_factor_prio, prioritarian_discounting, CPC, region_pop,
                         damages, Y, CPC_lo, climate_impact_relative_to_capita, CPC_post_damage):
        """
        @param t: int
        @param year: int
        @param irstp: float
        @param tstep: int
        @param growth_factor_prio: int
        @param prioritarian_discounting: int
        @param CPC: numpy array (12, 31)
        @param region_pop: numpy array (12, 31)
        @param damages: numpy array (12, 31)
        @param Y: numpy array (12, 31)
        @param CPC_lo: float
        @param climate_impact_relative_to_capita: dictionary
        @param CPC_post_damage: dictionary
        """

        self.set_up_weights_related(t, irstp, tstep, CPC, region_pop)

        # specify growth factor for conditional discounting
        self.growth_factor = growth_factor_prio ** 10
        self.prioritarian_discounting = prioritarian_discounting

        # check for discounting prioritarian

        # no discounting used
        if self.prioritarian_discounting == 0:
            self.per_util_ww[:, t] = self.inst_util_ww[:, t] * region_pop[:, t]

        # only execute discounting when the lowest income groups experience consumption level growth
        if self.prioritarian_discounting == 1:
            # utility worst-off
            self.inst_util_worst_off[:, t] = ((1 / (1 - self.elasmu)) * (CPC_post_damage[year][0]) **
                                              (1 - self.elasmu) + 1)

            self.inst_util_worst_off_condition[:, t] = ((1 / (1 - self.elasmu)) * (
                    CPC_post_damage[year - 10][0] * self.growth_factor) ** (1 - self.elasmu) + 1)

            # apply discounting when all regions experience enough growth

            for region in range(0, self.n_regions):
                if self.inst_util_worst_off[region, t] >= self.inst_util_worst_off_condition[region, t]:
                    self.per_util_ww[region, t] = self.inst_util_ww[region, t] * region_pop[region, t] * \
                                                  self.util_sdr[region, t]

                # no discounting when lowest income groups do not experience enough growth
                else:
                    self.per_util_ww[region, t] = self.inst_util_ww[region, t] * region_pop[region, t]

        self.calculate_alternative_principles_objectives(t, year, CPC, damages, CPC_post_damage, CPC_lo, region_pop,
                                                         self.welfare_function, climate_impact_relative_to_capita, Y)

    def run_sufficientarian(self, t, year, irstp, tstep, growth_factor_suf, sufficientarian_discounting, CPC,
                            region_pop, damages, Y, CPC_lo, climate_impact_relative_to_capita, CPC_post_damage):
        """
        @param t: int
        @param year: int
        @param irstp: float
        @param tstep: int
        @param growth_factor_suf: int
        @param sufficientarian_discounting: int
        @param CPC: numpy array (12, 31)
        @param region_pop: numpy array (12, 31)
        @param damages: numpy array (12, 31)
        @param Y: numpy array (12, 31)
        @param CPC_lo: float
        @param climate_impact_relative_to_capita: dictionary
        @param CPC_post_damage: dictionary
        """

        self.set_up_weights_related(t, irstp, tstep, CPC, region_pop)

        # sufficientarian controls
        self.sufficientarian_discounting = sufficientarian_discounting

        # ten year growth factor to be met to discount
        self.temporal_growth_factor = growth_factor_suf ** 10

        # growth by technology frontier
        self.growth_frontier[t] = (np.max(CPC[:, t]) - np.max(CPC[:, t - 1])) / np.max(CPC[:, t - 1])

        # sufficientarian discounting
        # only discount when economy situations is as good as timestep before in every region
        if sufficientarian_discounting == 0:
            for region in range(0, self.n_regions):
                if CPC[region, t] < CPC[region, t - 1]:
                    self.per_util_ww[:, t] = self.inst_util_ww[:, t] * region_pop[:, t]
                    break
                else:
                    self.per_util_ww[region, t] = self.inst_util_ww[region, t] * region_pop[region, t] * \
                                                  self.util_sdr[region, t]

        # only discount when next generation experiences certain growth in every region
        if sufficientarian_discounting == 1:
            for region in range(0, self.n_regions):
                if CPC[region, t] < CPC[region, t - 1] * self.temporal_growth_factor:
                    self.per_util_ww[:, t] = self.inst_util_ww[:, t] * region_pop[:, t]
                    break
                else:
                    self.per_util_ww[region, t] = self.inst_util_ww[region, t] * region_pop[region, t] * \
                                                  self.util_sdr[region, t]

        self.global_per_util_ww[t] = self.per_util_ww[:, t].sum(axis=0)

        self.calculate_alternative_principles_objectives(t, year, CPC, damages, CPC_post_damage, CPC_lo, region_pop,
                                                         self.welfare_function, climate_impact_relative_to_capita, Y)

    def run_egalitarian(self, t, year, irstp, tstep, egalitarian_discounting, CPC, region_pop, damages, Y, CPC_lo,
                        climate_impact_relative_to_capita, CPC_post_damage):
        """
        @param t: int
        @param year: int
        @param irstp: float
        @param tstep: int
        @param egalitarian_discounting: int
        @param CPC: numpy array (12, 31)
        @param region_pop: numpy array (12, 31)
        @param damages: numpy array (12, 31)
        @param Y: numpy array (12, 31)
        @param CPC_lo: float
        @param climate_impact_relative_to_capita: dictionary
        @param CPC_post_damage: dictionary
        """

        self.set_up_weights_related(t, irstp, tstep, CPC, region_pop)

        # controls for egalitarian principles
        self.egalitarian_discounting = egalitarian_discounting

        # apply no discounting
        if self.egalitarian_discounting == 1:
            self.per_util_ww[:, t] = self.inst_util_ww[:, t] * region_pop[:, t]
        else:
            self.per_util_ww[:, t] = self.inst_util_ww[:, t] * region_pop[:, t] * self.util_sdr[:, t]

        self.global_per_util_ww[t] = self.per_util_ww[:, t].sum(axis=0)

        self.calculate_alternative_principles_objectives(t, year, CPC, damages, CPC_post_damage, CPC_lo, region_pop,
                                                         self.welfare_function, climate_impact_relative_to_capita, Y)

    def get_outcomes(self, temp_atm, E_worldwilde_per_year, region_pop, CPC_pre_damage, CPC_post_damage, CPC,
                     start_year, end_year, tstep, precision=10):
        """
        Prepare outcome variables in a dictionary and return it.
        @param temp_atm: numpy array (31,)
        @param E_worldwilde_per_year: numpy array (31,)
        @param region_pop: numpy array (12, 31)
        @param CPC_pre_damage: dictionary
        @param CPC_post_damage: dictionary
        @param CPC: numpy array (12, 31)
        @param start_year: int
        @param end_year: int
        @param tstep: int
        @param precision: int
        @return:
            self.data_dict: dictionary
        """

        # egalitarian objectives troubles with NaN
        self.take_care_of_nans()

        objectives_list_timeseries = [self.global_damages, self.global_per_util_ww,
                                      self.worst_off_income_class,
                                      self.worst_off_climate_impact,
                                      self.max_utility_distance_threshold,
                                      self.population_under_threshold,
                                      self.CPC_intra_gini,
                                      self.climate_impact_per_dollar_gini,
                                      temp_atm, E_worldwilde_per_year, self.global_ouput
                                      ]

        objectives_list = [self.intertemporal_utility_gini, self.intertemporal_impact_gini, self.utility,
                           self.regions_under_threshold]

        objectives_list_name = ['Intertemporal utility GINI', 'Intertemporal impact GINI', 'Total Aggregated Utility',
                                'Regions below threshold']

        objectives_list_timeseries_name = ['Damages ', 'Utility ',
                                           'Lowest income per capita ', 'Highest climate impact per capita ',
                                           'Distance to threshold ', 'Population under threshold ',
                                           'Intratemporal utility GINI ', 'Intratemporal impact GINI ',
                                           'Atmospheric Temperature ', 'Industrial Emission ', 'Total Output ']

        supplementary_list_timeseries = [CPC, region_pop]
        supplementary_list_quintile = [CPC_pre_damage, CPC_post_damage]

        supplementary_list_timeseries_name = ['CPC ', 'Population ']
        supplementary_list_quintile_name = ['CPC pre damage ', 'CPC post damage ']

        # set up range of timepoints to save with the needed precision
        self.timepoints_to_save = np.arange(start_year, end_year + precision, precision)

        # setup data_dict dict
        self.data_dict = {}

        # save objectives with timeseries
        for idx, name in enumerate(objectives_list_timeseries_name):
            for year in self.timepoints_to_save:
                name_year = name + str(year)
                timestep = (year - start_year) / tstep
                self.data_dict[name_year] = objectives_list_timeseries[idx][int(timestep)]

        # save objectives with only one aggregate statistic over all timesteps
        for idx, name in enumerate(objectives_list_name):
            self.data_dict[name] = objectives_list[idx]

        # save additional data_dict in timeseries format split within regions
        for idx, name in enumerate(supplementary_list_timeseries_name):
            for year in self.timepoints_to_save:
                name_year = name + str(year)
                timestep = (year - start_year) / tstep
                timestep_list = []

                for region in range(0, self.n_regions):
                    timestep_list.append(supplementary_list_timeseries[idx][region][int(timestep)])
                self.data_dict[name_year] = timestep_list

        # save additional data_dict in quintile 5 X 12 per year format
        for idx, name in enumerate(supplementary_list_quintile_name):
            for year in self.timepoints_to_save:
                name_year = name + str(year)
                self.data_dict[name_year] = supplementary_list_quintile[idx][year].tolist()

        self.data = Results(self.data_dict, self.timepoints_to_save, self.regions_list)

        return self.data_dict

    def take_care_of_nans(self):
        """
        Take care of NaNs for egalitarian objectives.
        """

        if np.isnan(self.CPC_intra_gini[5]):
            self.CPC_intra_gini[5] = 0

        if np.isnan(self.climate_impact_per_dollar_gini[5]):
            self.climate_impact_per_dollar_gini[5] = 0

        if np.isnan(self.CPC_intra_gini[10]):
            self.CPC_intra_gini[10] = 0

        if np.isnan(self.climate_impact_per_dollar_gini[10]):
            self.climate_impact_per_dollar_gini[10] = 0

        if np.isnan(self.CPC_intra_gini[15]):
            self.CPC_intra_gini[15] = 0

        if np.isnan(self.climate_impact_per_dollar_gini[15]):
            self.climate_impact_per_dollar_gini[15] = 0

        if np.isnan(self.CPC_intra_gini[20]):
            self.CPC_intra_gini[20] = 0

        if np.isnan(self.climate_impact_per_dollar_gini[20]):
            self.climate_impact_per_dollar_gini[20] = 0

        if np.isnan(self.CPC_intra_gini[30]):
            self.CPC_intra_gini[30] = 0

        if np.isnan(self.climate_impact_per_dollar_gini[30]):
            self.climate_impact_per_dollar_gini[30] = 0

        if np.isnan(self.intertemporal_utility_gini):
            self.intertemporal_utility_gini = 0

        if np.isnan(self.intertemporal_impact_gini):
            self.intertemporal_impact_gini = 40

    def get_elasmu(self):
        """
        @return:
            self.elasmu: float
        """
        return self.elasmu

    # Helper methods for calculating utilities for social welfare functions

    def set_up_weights_related(self, t, irstp, tstep, CPC, region_pop):
        """
        @param t: int
        @param irstp: float
        @param tstep: int
        @param CPC: numpy array (12, 31)
        @param region_pop: numpy array (12, 31)
        """

        # irstp: Initial rate of social time preference per year
        self.util_sdr[:, t] = 1 / ((1 + irstp) ** (tstep * t))

        # instantaneous welfare without welfare weights
        self.inst_util[:, t] = ((1 / (1 - self.elasmu)) * (CPC[:, t]) ** (1 - self.elasmu) + 1)

        # period utility
        self.per_util[:, t] = self.inst_util[:, t] * region_pop[:, t] * self.util_sdr[:, t]

        # cumulative period utilty without welfare weights
        self.cum_per_util[:, 0] = self.cum_per_util[:, t - 1] + self.per_util[:, t]

        # Instantaneous utility function with welfare weights
        self.inst_util_ww[:, t] = self.inst_util[:, t] * self.Alpha_data[:, t]

    def calculate_alternative_principles_objectives(self, t, year, CPC, damages, CPC_post_damage, CPC_lo, region_pop,
                                                    welfare_function, climate_impact_relative_to_capita, Y):
        """
        @param t: int
        @param year: int
        @param CPC: numpy array (12, 31)
        @param damages: numpy array (12, 31)
        @param CPC_post_damage: dictionary
        @param CPC_lo: float
        @param region_pop: numpy array (12, 31)
        @param welfare_function: WelfareFunction
        @param climate_impact_relative_to_capita: dictionary
        @param Y: numpy array (12, 31)
        """

        # cummulative utility with ww
        self.reg_cum_util[:, t] = self.reg_cum_util[:, t - 1] + self.per_util_ww[:, t]

        # scale utility with weights derived from the excel
        if t == 30:
            self.reg_util = 10 * self.multiplutacive_scaling_weights[:, 0] * self.reg_cum_util[:, t] + \
                            self.additative_scaling_weights[:, 0] - self.additative_scaling_weights[:, 2]

            # calculate worldwide utility
        self.utility = self.reg_util.sum()

        # ###### GINI calculations INTERTEMPORAL #########
        # CPC is floored on minimum value
        CPC[:, t] = np.where(CPC[:, t] > CPC_lo, CPC[:, t], CPC_lo)

        self.average_world_CPC[0] = (CPC[:, 0].sum(axis=0) / self.n_regions)

        self.average_world_CPC[t] = (CPC[:, t].sum(axis=0) / self.n_regions)

        if t == 30:
            input_gini_inter_cpc = self.average_world_CPC

            diffsum = 0
            for i, xi in enumerate(input_gini_inter_cpc[:-1], 1):
                diffsum += np.sum(np.abs(xi - input_gini_inter_cpc[i:]))

            self.intertemporal_utility_gini = diffsum / (
                    (len(input_gini_inter_cpc) ** 2) * np.mean(input_gini_inter_cpc))

        # intertemporal climate impact GINI
        self.average_regional_impact[t] = (damages[:, t].sum(axis=0) / self.n_regions)

        # Impact is floored on minimum value
        if t == 30:
            input_gini_inter = self.average_regional_impact

            diffsum_inter = 0

            for i, xi in enumerate(input_gini_inter[:-1], 1):
                diffsum_inter += np.sum(np.abs(xi - input_gini_inter[i:]))

            self.intertemporal_impact_gini = diffsum_inter / (
                    (len(input_gini_inter) ** 2) * np.mean(input_gini_inter))

        # ###### GINI calculations INTRATEMPORAL #########
        # calculate gini as measure of current inequality in welfare (intragenerational)
        # CPC is floored on minimum value
        input_gini_intra = CPC[:, t]

        diffsum = 0
        for i, xi in enumerate(input_gini_intra[:-1], 1):
            diffsum += np.sum(np.abs(xi - input_gini_intra[i:]))

        self.CPC_intra_gini[t] = diffsum / ((len(input_gini_intra) ** 2) * np.mean(input_gini_intra))

        # calculate gini as measure of current inequality in climate impact (per dollar consumption)
        # (intragenerational)
        self.climate_impact_per_dollar_consumption[:, t] = np.where(damages[:, t] < 0.001, CPC[:, t],
                                                                    damages[:, t] / CPC[:, t])

        input_gini_intra_impact = self.climate_impact_per_dollar_consumption[:, t]

        diffsum = 0
        for i, xi in enumerate(input_gini_intra_impact[:-1], 1):
            diffsum += np.sum(np.abs(xi - input_gini_intra_impact[i:]))

        self.climate_impact_per_dollar_gini[t] = diffsum / (
                (len(input_gini_intra_impact) ** 2) * np.mean(input_gini_intra_impact))

        # sufficientarian objectives
        # growth by the world
        self.average_world_CPC[t] = CPC[:, t].sum() / self.n_regions
        self.average_growth_CPC[t] = (self.average_world_CPC[t] - self.average_world_CPC[t - 1]) / (
            self.average_world_CPC[t - 1])

        # sufficientarian threshold adjusted by the growth of the average world economy
        self.sufficientarian_threshold[t] = self.sufficientarian_threshold[t - 1] * (
                1 + self.average_growth_CPC[t])

        # calculate instantaneous welfare equivalent of minimum capita per head
        self.inst_util_tres[t] = ((1 / (1 - self.elasmu)) * (self.sufficientarian_threshold[t]) ** (1 - self.elasmu) + 1)

        # calculate instantaneous welfare equivalent of threshold
        self.inst_util_tres_ww[:, t] = self.inst_util_tres[t] * self.Alpha_data[:, t]

        # calculate utility equivalent for every income quintile and scale with welfare weights for comparison
        self.quintile_inst_util[year] = ((1 / (1 - self.elasmu)) * (CPC_post_damage[year]) ** (1 - self.elasmu) + 1)
        self.quintile_inst_util_ww[year] = self.quintile_inst_util[year] * self.Alpha_data[:, t]

        utility_per_income_share = self.quintile_inst_util_ww[year]

        list_timestep = []

        for quintile in range(0, 5):
            for region in range(0, self.n_regions):
                if utility_per_income_share[quintile, region] < self.inst_util_tres_ww[region, t]:
                    self.population_under_threshold[t] = \
                        self.population_under_threshold[t] + region_pop[region, t] * 1 / 5
                    self.utility_distance_threshold[region, t] = self.inst_util_tres_ww[region, t] - \
                                                                 utility_per_income_share[quintile, region]

                    list_timestep.append(self.regions_list[region])

        if welfare_function.__eq__(WelfareFunction.SUFFICIENTARIAN):
            self.regions_under_threshold[t] = list_timestep

        # minimize max distance to threshold
        self.max_utility_distance_threshold[t] = self.utility_distance_threshold[:, t].max()

        # prioritarian objectives
        self.worst_off_income_class[t] = CPC_post_damage[year][0].min()
        self.worst_off_climate_impact[t] = climate_impact_relative_to_capita[year][0].max()

        # Utilitarian objectives
        self.global_damages[t] = damages[:, t].sum(axis=0)
        self.global_ouput[t] = Y[:, t].sum(axis=0)
        self.global_per_util_ww[t] = self.per_util_ww[:, t].sum(axis=0)


class Results:
    """
    Results of the PyRICE model in a better formatted way.
    """

    def __init__(self, data_dict, tperiod, regions_list):
        """
        @param data_dict: dictionary
        @param tperiod: numpy array (31,)
        @param regions_list: list with 12 regions
        """

        self.data_dict = data_dict
        years = tperiod

        # Create one big dataframe
        damages = self.get_values_for_specific_prefix("Damages")
        utility = self.get_values_for_specific_prefix("Utility")
        lowest = self.get_values_for_specific_prefix("Lowest income per capita")
        highest = self.get_values_for_specific_prefix("Highest climate impact per capita")
        distance = self.get_values_for_specific_prefix("Distance to threshold")
        population = self.get_values_for_specific_prefix("Population under threshold")
        utility_gini = self.get_values_for_specific_prefix("Intratemporal utility GINI")
        impact_gini = self.get_values_for_specific_prefix("Intratemporal impact GINI")
        temp = self.get_values_for_specific_prefix("Atmospheric Temperature")
        emission = self.get_values_for_specific_prefix("Industrial Emission")
        output = self.get_values_for_specific_prefix("Total Output")
        regions_under_threshold = self.data_dict["Regions below threshold"]

        columns = ['Damges', 'Utility', 'Lowest income per capita', 'Highest climate impact per capita',
                   'Distance to threshold', 'Population under threshold', 'Intratemporal utility GINI',
                   'Intratemporal impact GINI', 'Atmospheric temperature', 'Industrial emission', 'Total output',
                   'Regions below threshold']

        self.df_main = pd.DataFrame(list(zip(damages, utility, lowest, highest, distance, population, utility_gini,
                                         impact_gini, temp, emission, output, regions_under_threshold)),
                                   index=years,
                                   columns=columns)

        # Highly aggregated variables
        self.aggregated_utility_gini = self.data_dict["Intertemporal utility GINI"]
        self.aggregated_impact_gini = self.data_dict["Intertemporal impact GINI"]
        self.aggregated_utility = self.data_dict["Total Aggregated Utility"]

        # CPC dataframe
        cpc = self.get_values_for_specific_prefix("CPC 2")
        columns = regions_list
        self.df_cpc = pd.DataFrame(cpc, index=years, columns=columns)

        # Population dataframe
        population = self.get_values_for_specific_prefix("Population 2")
        columns = regions_list
        self.df_population = pd.DataFrame(population, index=years, columns=columns)

        # CPC pre damage dataframe
        cpc_pre = self.get_values_for_specific_prefix("CPC pre")
        cpc_pre = np.array(cpc_pre)
        cpc_pre = np.transpose(cpc_pre, (2, 0, 1))
        columns = regions_list
        self.df_cpc_pre_damage = pd.DataFrame(list(zip(*cpc_pre)), index=years, columns=columns)

        # CPC post damage dataframe
        cpc_post = self.get_values_for_specific_prefix("CPC post")
        cpc_post = np.array(cpc_post)
        cpc_post = np.transpose(cpc_post, (2, 0, 1))
        columns = regions_list
        self.df_cpc_post_damage = pd.DataFrame(list(zip(*cpc_post)), index=years, columns=columns)

    def get_values_for_specific_prefix(self, prefix="Damages"):
        """
        Find the values for a specific key in self.data_dict where the key string starts with the given prefix.
        @param prefix: string
        @return:
            values: list with floats
        """

        # Get all relvant key-value pairs
        sub_dict = {}

        for key, value in self.data_dict.items():
            if prefix in key:
                sub_dict[key] = value

        # Save items as sorted list by year
        items = list(sub_dict.items())

        items.sort(key=lambda x: x[0])

        # Get values
        values = [x[1] for x in items]

        return values
