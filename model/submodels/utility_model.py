"""
This module contains the utility model class and the results_formatted class.
"""

import numpy as np
from model.enumerations import WelfareFunction
import pandas as pd


class UtilityModel:
    """
    This sub-model describes the utility part of the PyRICE model.
    """

    def __init__(
        self,
        steps,
        data_sets,
        regions_list,
        limits,
        welfare_function,
        damage_function,
        emdd,
    ):
        """
        @param steps: int (31)
        @param data_sets: DataSets
        @param regions_list: list with 12 regions
        @param limits: ModelLimits
        @param welfare_function: WelfareFunction
        @param damage_function: DamageFunction
        @param emdd: elasticity of marginal disutility of damages
        """

        self.welfare_function = welfare_function
        self.damage_function = damage_function

        self.data_sets = data_sets
        self.regions_list = regions_list
        self.n_regions = len(regions_list)
        self.regions_list = regions_list

        # limits
        self.dpc_lo = limits.dpc_lo
        self.sdr_dam_lo = limits.sdr_dam_lo
        self.inst_disutil_lo = limits.inst_disutil_lo

        # output metrics
        self.discount_factors_utility = np.zeros((self.n_regions, steps))
        self.period_utilities = np.zeros((self.n_regions, steps))
        self.utility_welfares = np.zeros((self.n_regions, steps))

        self.cum_util = np.zeros((self.n_regions, steps))
        self.reg_cum_util = np.zeros((self.n_regions, steps))
        self.reg_util = np.zeros(self.n_regions)
        self.util = np.zeros((self.n_regions, steps))

        self.per_util_ww = np.zeros((self.n_regions, steps))
        self.cum_utility_welfares = np.zeros((self.n_regions, steps))
        self.inst_util_ww = np.zeros((self.n_regions, steps))

        # Elasticity of marginal utility of consumption and damages
        self.emcu = 1.50
        self.emdd = emdd

        # Get scaling data_dict for welfare weights and aggregated utility
        self.Alpha_data = data_sets.RICE_DATA.iloc[357:369, 1:60].to_numpy()
        self.additative_scaling_weights = data_sets.RICE_DATA.iloc[
            167:179, 14:17
        ].to_numpy()
        self.multiplutacive_scaling_weights = (
            data_sets.RICE_DATA.iloc[232:244, 1:2].to_numpy() / 1000
        )

        # Outcome variables

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
        self.worst_off_damage = np.zeros(steps)
        self.worst_off_climate_impact_index = np.zeros(steps)

        # Sufficientarian outputs
        self.average_world_CPC = np.zeros(steps)
        self.average_growth_CPC = np.zeros(steps)

        # Sufficientarian outputs (consumption)
        self.sufficientarian_consumption_threshold = np.zeros(steps)
        self.inst_util_thres = np.zeros(steps)
        self.inst_util_thres_ww = np.zeros((self.n_regions, steps))
        self.quintile_inst_util = {}
        self.quintile_inst_util_ww = {}
        self.population_below_consumption_threshold = np.zeros(steps)
        self.utility_distance_threshold = np.zeros((self.n_regions, steps))
        self.max_utility_distance_threshold = np.zeros(steps)
        self.regions_below_consumption_threshold = []
        self.largest_distance_below_consumption_threshold = np.zeros(steps)
        self.growth_frontier = np.zeros(steps)

        # Sufficientarian outputs (damages)
        self.sufficientarian_damage_threshold = np.zeros(steps)
        self.inst_disutil_thres = np.zeros(steps)
        self.inst_disutil_thres_ww = np.zeros((self.n_regions, steps))
        self.quintile_inst_disutil = {}
        self.quintile_inst_disutil_ww = {}
        self.population_above_damage_threshold = np.zeros(steps)
        self.disutility_distance_threshold = np.zeros((self.n_regions, steps))
        self.max_disutility_distance_threshold = np.zeros(steps)
        self.regions_above_damage_threshold = []

        # Egalitarian outputs
        self.CPC_intra_gini = np.zeros(steps)
        self.average_regional_impact = np.zeros(steps)
        self.climate_impact_per_dollar_consumption = np.zeros((self.n_regions, steps))
        self.climate_impact_per_dollar_gini = np.zeros(steps)

        # Disutility-related outputs
        self.dpc = np.zeros((self.n_regions, steps)) + 0.000000001
        self.period_disutilities = np.zeros((self.n_regions, steps))
        self.inst_disutility_ww = np.zeros((self.n_regions, steps))
        self.dam_g = np.zeros((self.n_regions, steps))
        self.rho = np.zeros((self.n_regions, steps))
        self.discount_factors_disutility = np.zeros((self.n_regions, steps))
        self.per_disutility_ww = np.zeros((self.n_regions, steps))
        self.global_per_disutility_ww = np.zeros(steps)
        self.reg_cum_disutil = np.zeros((self.n_regions, steps))

    def set_up_utility(
        self,
        ini_suf_threshold_consumption,
        relative_damage_threshold,
        climate_impact_relative_to_capita,
        CPC_post_damage,
        CPC,
        region_pop,
        damages,
        Y,
    ):
        """
        Sets up most variables with their initial values.
        @param ini_suf_threshold_consumption: float
        @param relative_damage_threshold: float: percentage of how high damage can be compared to consumption
        @param climate_impact_relative_to_capita: dictionary
        @param CPC_post_damage: dictionary
        @param CPC: numpy array (12, 31)
        @param region_pop: numpy array (12, 31)
        @param damages: numpy array (12, 31)
        @param Y: numpy array (12, 31)
        """

        self.relative_damage_threshold = relative_damage_threshold

        # Initial rate of social time preference per year
        self.discount_factors_utility[:, 0] = 1
        self.discount_factors_disutility[:, 0] = 1

        # Instantaneous utility function equation
        self.period_utilities[:, 0] = (1 / (1 - self.emcu)) * (CPC[:, 0]) ** (
            1 - self.emcu
        ) + 1
        self.period_disutilities[:, 0] = self.dpc[:, 0] ** (1 - self.emdd) / (
            1 - self.emdd
        )

        # CEMU period utility
        self.utility_welfares[:, 0] = (
            self.period_utilities[:, 0]
            * region_pop[:, 0]
            * self.discount_factors_utility[:, 0]
        )

        # Cummulative period utility without WW
        self.cum_utility_welfares[:, 0] = self.utility_welfares[:, 0]

        # Instantaneous utility function with welfare weights
        self.inst_util_ww[:, 0] = self.period_utilities[:, 0] * self.Alpha_data[:, 0]
        self.inst_disutility_ww[:, 0] = (
            self.period_disutilities[:, 0] * self.Alpha_data[:, 0]
        )

        # Period utility with welfare weights
        self.per_util_ww[:, 0] = (
            self.inst_util_ww[:, 0]
            * region_pop[:, 0]
            * self.discount_factors_utility[:, 0]
        )
        self.per_disutility_ww[:, 0] = (
            self.inst_disutility_ww[:, 0]
            * region_pop[:, 0]
            * self.discount_factors_disutility[:, 0]
        )

        # cummulative utility with ww
        self.reg_cum_util[:, 0] = self.utility_welfares[:, 0]
        self.reg_cum_disutil[:, 0] = self.per_disutility_ww[:, 0]
        self.global_per_util_ww[0] = self.per_util_ww[:, 0].sum(axis=0)

        # initialise objectives for principles
        # objective for the worst-off region in terms of consumption per capita
        self.worst_off_income_class[0] = CPC_post_damage[2005][0].min()

        array_worst_off_income = CPC_post_damage[2005][0]
        self.worst_off_income_class_index[0] = np.argmin(array_worst_off_income)

        # objective for the worst-off region in terms of climate impact
        self.worst_off_damage[0] = climate_impact_relative_to_capita[2005][0].min()

        array_worst_off_share = climate_impact_relative_to_capita[2005][0]
        self.worst_off_climate_impact_index[0] = np.argmin(array_worst_off_share)

        # objectives sufficientarian
        self.average_world_CPC[0] = CPC[:, 0].sum() / self.n_regions
        self.average_growth_CPC[
            0
        ] = 0.250  # average growth over 10 years World Bank Data

        # calculate instantaneous welfare equivalent of minimum capita per head
        # specified in consumption per capita thousand/year
        self.sufficientarian_consumption_threshold[0] = ini_suf_threshold_consumption

        self.inst_util_thres[0] = (1 / (1 - self.emcu)) * (
            self.sufficientarian_consumption_threshold[0]
        ) ** (1 - self.emcu) + 1

        # calculate instantaneous welfare equivalent of minimum capita per head with PPP
        self.inst_util_thres_ww[:, 0] = self.inst_util_thres[0] * self.Alpha_data[:, 0]

        # calculate utility equivalent for every income quintile and scale with welfare weights for comparison
        self.quintile_inst_util[2005] = (1 / (1 - self.emcu)) * (
            CPC_post_damage[2005]
        ) ** (1 - self.emcu) + 1
        self.quintile_inst_util_ww[2005] = (
            self.quintile_inst_util[2005] * self.Alpha_data[:, 0]
        )

        utility_per_income_share = self.quintile_inst_util_ww[2005]

        list_timestep = []

        for quintile in range(0, 5):
            for region in range(0, self.n_regions):
                if (
                    utility_per_income_share[quintile, region]
                    < self.inst_util_thres_ww[region, 0]
                ):
                    self.population_below_consumption_threshold[0] = (
                        self.population_below_consumption_threshold[0]
                        + region_pop[region, 0] * 1 / 5
                    )
                    self.utility_distance_threshold[region, 0] = (
                        self.inst_util_thres_ww[region, 0]
                        - utility_per_income_share[quintile, region]
                    )

                    list_timestep.append(self.regions_list[region])

        self.regions_below_consumption_threshold.append(list_timestep)

        self.max_utility_distance_threshold[0] = self.utility_distance_threshold[
            :, 0
        ].max()

        # Set up sufficentarian objectives for damages (first entry)
        list_timestep = []

        relative_damage = np.divide(
            damages, CPC, out=np.zeros_like(damages), where=(CPC != 0)
        )
        for region in range(self.n_regions):
            if relative_damage[region, 0] > relative_damage_threshold:
                self.population_above_damage_threshold[0] = (
                    self.population_above_damage_threshold[0] + region_pop[region, 0]
                )
                self.disutility_distance_threshold[region, 0] = np.transpose(
                    (damages[region, 0] - CPC[region, 0] * relative_damage_threshold)
                )
                list_timestep.append(self.regions_list[region])

        self.regions_above_damage_threshold.append(list_timestep)
        self.max_disutility_distance_threshold[0] = self.disutility_distance_threshold[
            :, 0
        ].max()

        # calculate gini as measure of current inequality in consumption (intragenerational)
        input_gini = CPC[:, 0]

        diffsum = 0
        for i, xi in enumerate(input_gini[:-1], 1):
            diffsum += np.sum(np.abs(xi - input_gini[i:]))

        self.CPC_intra_gini[0] = diffsum / (
            (len(input_gini) ** 2) * np.mean(input_gini)
        )

        # calculate gini as measure of current inequality in climate impact (per dollar consumption)
        # (intragenerational)
        self.climate_impact_per_dollar_consumption[:, 0] = damages[:, 0] / CPC[:, 0]
        input_gini = self.climate_impact_per_dollar_consumption[:, 0]

        diffsum = 0
        for i, xi in enumerate(input_gini[:-1], 1):
            diffsum += np.sum(np.abs(xi - input_gini[i:]))

        self.climate_impact_per_dollar_gini[0] = diffsum / (
            (len(input_gini) ** 2) * np.mean(input_gini)
        )

        self.global_damages[0] = damages[:, 0].sum(axis=0)
        self.global_ouput[0] = Y[:, 0].sum(axis=0)

        # Setting up for disutilities
        self.previous_dpc = np.zeros(region_pop.shape)

    def run(
        self,
        t,
        year,
        irstp_consumption,
        irstp_damage,
        tstep,
        growth_factor_prio,
        growth_factor_suf,
        sufficientarian_discounting,
        egalitarian_discounting,
        prioritarian_discounting,
        CPC,
        region_pop,
        damages,
        Y,
        CPC_lo,
        climate_impact_relative_to_capita,
        CPC_post_damage,
        emdd,
    ):
        """
        @param t: int
        @param year: int
        @param irstp_consumption: float
        @param irstp_damage: float
        @param tstep: int (10)
        @param growth_factor_prio: int
        @param growth_factor_suf: int
        @param sufficientarian_discounting: int
        @param egalitarian_discounting: int
        @param prioritarian_discounting: int
        @param CPC: numpy array (12, 31)
        @param region_pop: numpy array (12, 31)
        @param damages: numpy array (12, 31)
        @param Y: numpy array (12, 31)
        @param CPC_lo: float
        @param climate_impact_relative_to_capita: dictionary
        @param CPC_post_damage: dictionary
        @param emdd: elasticity of marginal disutility of damages: float
        @return:
            CPC: numpy array (12, 31)
            CPC_post_damage: dictionary
        """

        self.region_pop = region_pop
        self.compute_welfare_disutility(damages, emdd, tstep, t, irstp_damage)

        if self.welfare_function == WelfareFunction.UTILITARIAN:
            self.run_utilitarian(
                t,
                year,
                irstp_consumption,
                tstep,
                CPC,
                damages,
                Y,
                CPC_lo,
                climate_impact_relative_to_capita,
                CPC_post_damage,
            )

        elif self.welfare_function == WelfareFunction.PRIORITARIAN:
            self.run_prioritarian(
                t,
                year,
                irstp_consumption,
                tstep,
                growth_factor_prio,
                prioritarian_discounting,
                CPC,
                damages,
                Y,
                CPC_lo,
                climate_impact_relative_to_capita,
                CPC_post_damage,
            )

        elif self.welfare_function == WelfareFunction.SUFFICIENTARIAN:
            self.run_sufficientarian(
                t,
                year,
                irstp_consumption,
                tstep,
                growth_factor_suf,
                sufficientarian_discounting,
                CPC,
                damages,
                Y,
                CPC_lo,
                climate_impact_relative_to_capita,
                CPC_post_damage,
            )

        elif self.welfare_function == WelfareFunction.EGALITARIAN:
            self.run_egalitarian(
                t,
                year,
                irstp_consumption,
                tstep,
                egalitarian_discounting,
                CPC,
                damages,
                Y,
                CPC_lo,
                climate_impact_relative_to_capita,
                CPC_post_damage,
            )

        else:
            raise ValueError("Oh, no! Apparently, your welfare function is unknown!")

        return CPC, CPC_post_damage

    def run_utilitarian(
        self,
        t,
        year,
        irstp_consumption,
        tstep,
        CPC,
        damages,
        Y,
        CPC_lo,
        climate_impact_relative_to_capita,
        CPC_post_damage,
    ):
        """
        @param t: int
        @param year: int
        @param irstp_consumption: float
        @param tstep: int
        @param CPC: numpy array (12, 31)
        @param damages:numpy array (12, 31)
        @param Y: numpy array (12, 31)
        @param CPC_lo: float
        @param climate_impact_relative_to_capita: dictionary
        @param CPC_post_damage: dictionary
        """

        self.set_up_weights_related(t, irstp_consumption, tstep, CPC)

        # period utility with welfare weights
        self.per_util_ww[:, t] = (
            self.inst_util_ww[:, t]
            * self.region_pop[:, t]
            * self.discount_factors_utility[:, t]
        )
        self.regional_cum_util[t] = self.reg_cum_util[:, t].sum()

        self.compute_alternative_principles_objectives(
            t,
            year,
            CPC,
            damages,
            CPC_post_damage,
            CPC_lo,
            climate_impact_relative_to_capita,
            Y,
        )

    def run_prioritarian(
        self,
        t,
        year,
        irstp_consumption,
        tstep,
        growth_factor_prio,
        prioritarian_discounting,
        CPC,
        damages,
        Y,
        CPC_lo,
        climate_impact_relative_to_capita,
        CPC_post_damage,
    ):
        """
        @param t: int
        @param year: int
        @param irstp_consumption: float
        @param tstep: int
        @param growth_factor_prio: int
        @param prioritarian_discounting: int
        @param CPC: numpy array (12, 31)
        @param damages: numpy array (12, 31)
        @param Y: numpy array (12, 31)
        @param CPC_lo: float
        @param climate_impact_relative_to_capita: dictionary
        @param CPC_post_damage: dictionary
        """

        self.set_up_weights_related(t, irstp_consumption, tstep, CPC)

        # specify growth factor for conditional discounting
        self.growth_factor = growth_factor_prio ** 10
        self.prioritarian_discounting = prioritarian_discounting

        # check for discounting prioritarian

        # no discounting used
        if self.prioritarian_discounting == 0:
            self.per_util_ww[:, t] = self.inst_util_ww[:, t] * self.region_pop[:, t]

        # only execute discounting when the lowest income groups experience consumption level growth
        if self.prioritarian_discounting == 1:
            # utility worst-off
            self.inst_util_worst_off[:, t] = (1 / (1 - self.emcu)) * (
                CPC_post_damage[year][0]
            ) ** (1 - self.emcu) + 1

            self.inst_util_worst_off_condition[:, t] = (1 / (1 - self.emcu)) * (
                CPC_post_damage[year - 10][0] * self.growth_factor
            ) ** (1 - self.emcu) + 1

            # apply discounting when all regions experience enough growth

            for region in range(0, self.n_regions):
                if (
                    self.inst_util_worst_off[region, t]
                    >= self.inst_util_worst_off_condition[region, t]
                ):
                    self.per_util_ww[region, t] = (
                        self.inst_util_ww[region, t]
                        * self.region_pop[region, t]
                        * self.discount_factors_utility[region, t]
                    )

                # no discounting when lowest income groups do not experience enough growth
                else:
                    self.per_util_ww[region, t] = (
                        self.inst_util_ww[region, t] * self.region_pop[region, t]
                    )

        self.compute_alternative_principles_objectives(
            t,
            year,
            CPC,
            damages,
            CPC_post_damage,
            CPC_lo,
            climate_impact_relative_to_capita,
            Y,
        )

    def run_sufficientarian(
        self,
        t,
        year,
        irstp_consumption,
        tstep,
        growth_factor_suf,
        sufficientarian_discounting,
        CPC,
        damages,
        Y,
        CPC_lo,
        climate_impact_relative_to_capita,
        CPC_post_damage,
    ):
        """
        @param t: int
        @param year: int
        @param irstp_consumption: float
        @param tstep: int
        @param growth_factor_suf: int
        @param sufficientarian_discounting: int
        @param CPC: numpy array (12, 31)
        @param damages: numpy array (12, 31)
        @param Y: numpy array (12, 31)
        @param CPC_lo: float
        @param climate_impact_relative_to_capita: dictionary
        @param CPC_post_damage: dictionary
        """

        self.set_up_weights_related(t, irstp_consumption, tstep, CPC)

        # sufficientarian controls
        self.sufficientarian_discounting = sufficientarian_discounting

        # ten year growth factor to be met to discount
        self.temporal_growth_factor = growth_factor_suf ** 10

        # growth by technology frontier
        self.growth_frontier[t] = (np.max(CPC[:, t]) - np.max(CPC[:, t - 1])) / np.max(
            CPC[:, t - 1]
        )

        # sufficientarian discounting
        # only discount when economy situations is as good as timestep before in every region
        if sufficientarian_discounting == 0:
            for region in range(0, self.n_regions):
                if CPC[region, t] < CPC[region, t - 1]:
                    self.per_util_ww[:, t] = (
                        self.inst_util_ww[:, t] * self.region_pop[:, t]
                    )
                    break
                else:
                    self.per_util_ww[region, t] = (
                        self.inst_util_ww[region, t]
                        * self.region_pop[region, t]
                        * self.discount_factors_utility[region, t]
                    )

        # only discount when next generation experiences certain growth in every region
        if sufficientarian_discounting == 1:
            for region in range(0, self.n_regions):
                if CPC[region, t] < CPC[region, t - 1] * self.temporal_growth_factor:
                    self.per_util_ww[:, t] = (
                        self.inst_util_ww[:, t] * self.region_pop[:, t]
                    )
                    break
                else:
                    self.per_util_ww[region, t] = (
                        self.inst_util_ww[region, t]
                        * self.region_pop[region, t]
                        * self.discount_factors_utility[region, t]
                    )

        self.global_per_util_ww[t] = self.per_util_ww[:, t].sum(axis=0)

        self.compute_alternative_principles_objectives(
            t,
            year,
            CPC,
            damages,
            CPC_post_damage,
            CPC_lo,
            climate_impact_relative_to_capita,
            Y,
        )

    def run_egalitarian(
        self,
        t,
        year,
        irstp_consumption,
        tstep,
        egalitarian_discounting,
        CPC,
        damages,
        Y,
        CPC_lo,
        climate_impact_relative_to_capita,
        CPC_post_damage,
    ):
        """
        @param t: int
        @param year: int
        @param irstp_consumption: float
        @param tstep: int
        @param egalitarian_discounting: int
        @param CPC: numpy array (12, 31)
        @param damages: numpy array (12, 31)
        @param Y: numpy array (12, 31)
        @param CPC_lo: float
        @param climate_impact_relative_to_capita: dictionary
        @param CPC_post_damage: dictionary
        """

        self.set_up_weights_related(t, irstp_consumption, tstep, CPC)

        # controls for egalitarian principles
        self.egalitarian_discounting = egalitarian_discounting

        # apply no discounting
        if self.egalitarian_discounting == 1:
            self.per_util_ww[:, t] = self.inst_util_ww[:, t] * self.region_pop[:, t]
        else:
            self.per_util_ww[:, t] = (
                self.inst_util_ww[:, t]
                * self.region_pop[:, t]
                * self.discount_factors_utility[:, t]
            )

        self.global_per_util_ww[t] = self.per_util_ww[:, t].sum(axis=0)

        self.compute_alternative_principles_objectives(
            t,
            year,
            CPC,
            damages,
            CPC_post_damage,
            CPC_lo,
            climate_impact_relative_to_capita,
            Y,
        )

    def get_outcomes(
        self,
        temp_atm,
        E_worldwilde_per_year,
        CPC_pre_damage,
        CPC_post_damage,
        CPC,
        start_year,
        end_year,
        tstep,
        costs,
        precision=10,
    ):
        """
        Prepare outcome variables in a dictionary and return it.
        @param temp_atm: numpy array (31,)
        @param E_worldwilde_per_year: numpy array (31,)
        @param CPC_pre_damage: dictionary
        @param CPC_post_damage: dictionary
        @param CPC: numpy array (12, 31)
        @param start_year: int
        @param end_year: int
        @param tstep: int
        @param costs: numpy array (31, ): abatement costs + damages
        @param precision: int
        @return:
            self.data_dict: dictionary
        """

        # egalitarian objectives troubles with NaN
        self.take_care_of_nans()

        temperature_overhoots = self.compute_overshoots(temp_atm)

        # Regions counts
        nr_of_regions_below_consumption_threshold = []
        for value in self.regions_below_consumption_threshold:
            nr_of_regions_below_consumption_threshold.append(float(len(value)))
        nr_of_regions_above_damage_threshold = []
        for value in self.regions_above_damage_threshold:
            nr_of_regions_above_damage_threshold.append(float(len(value)))

        objectives_list_timeseries = [
            self.global_damages,
            self.global_per_util_ww,
            self.global_per_disutility_ww,
            self.worst_off_income_class,
            self.worst_off_damage,
            self.max_utility_distance_threshold,
            self.population_below_consumption_threshold,
            self.max_disutility_distance_threshold,
            self.population_above_damage_threshold,
            self.CPC_intra_gini,
            self.climate_impact_per_dollar_gini,
            temp_atm,
            temperature_overhoots,
            E_worldwilde_per_year,
            self.global_ouput,
            costs,
            nr_of_regions_below_consumption_threshold,
            nr_of_regions_above_damage_threshold
        ]

        # Add regional data
        for row in CPC:
            objectives_list_timeseries.append(row)
        for row in self.dpc:
            objectives_list_timeseries.append(row)

        # Extra aggregated variables
        self.aggregated_costs = costs.sum()

        # Sufficientarian
        self.intertemporal_max_distance_to_consumption_threshold = (
            self.max_utility_distance_threshold.sum()
        )
        self.intertemporal_max_distance_to_damage_threshold = (
            self.max_disutility_distance_threshold.sum()
        )
        self.intertemporal_population_below_consumption_threshold = (
            self.population_below_consumption_threshold.sum()
        )
        self.intertemporal_population_above_damage_threshold = (
            self.population_above_damage_threshold.sum()
        )

        # Prioritarian
        self.intertemporal_worst_off_income_class = self.worst_off_income_class.sum()
        self.intertemporal_worst_off_damage = self.worst_off_damage.sum()

        objectives_list = [
            self.intertemporal_consumption_gini,
            self.intertemporal_damage_gini,
            self.utility,
            self.disutility,
            self.regions_below_consumption_threshold,
            self.regions_above_damage_threshold,
            self.aggregated_costs,
            self.intertemporal_max_distance_to_consumption_threshold,
            self.intertemporal_max_distance_to_damage_threshold,
            self.intertemporal_population_below_consumption_threshold,
            self.intertemporal_population_above_damage_threshold,
            self.intertemporal_worst_off_income_class,
            self.intertemporal_worst_off_damage,
            temperature_overhoots[-1],
        ]

        objectives_list_name = [
            "Intertemporal consumption Gini",
            "Intertemporal damage Gini",
            "Total Aggregated Utility",
            "Total Aggregated Disutility",
            "Regions below consumption threshold",
            "Regions above damage threshold",
            "Total Aggregated Costs",
            "Intertemporal consumption distance",
            "Intertemporal damage distance",
            "Intertemporal consumption population",
            "Intertemporal damage population",
            "Intertemporal lowest income p/c",
            "Intertemporal highest damage p/c",
            "Total temperature overshoot",
        ]

        objectives_list_timeseries_name = [
            "Damages",
            "Utility",
            "Disutility",
            "Lowest income per capita",
            "Highest damage per capita",
            "Distance to consumption threshold",
            "Population below consumption threshold",
            "Distance to damage threshold",
            "Population above damage threshold",
            "Intratemporal consumption Gini",
            "Intratemporal damage Gini",
            "Atmospheric Temperature",
            "Temperature overshoot",
            "Industrial Emission",
            "Total Output",
            "Costs",
            "Number of regions below consumption threshold",
            "Number of regions above damage threshold"
        ]

        # Add names of regions for regional data
        for region in self.regions_list:
            objectives_list_timeseries_name.append(f'Regional CPC {region}')
        for region in self.regions_list:
            objectives_list_timeseries_name.append(f'Regional DPC {region}')

        # Adding space after each timeseries objective
        objectives_list_timeseries_name = [
            x + " " for x in objectives_list_timeseries_name
        ]

        supplementary_list_timeseries = [CPC, self.region_pop]
        supplementary_list_quintile = [CPC_pre_damage, CPC_post_damage]

        supplementary_list_timeseries_name = ["CPC ", "Population "]
        supplementary_list_quintile_name = ["CPC pre damage ", "CPC post damage "]

        # set up range of timepoints to save with the needed precision
        self.timepoints_to_save = np.arange(start_year, end_year + precision, precision)

        # setup data_dict dict
        self.data_dict = {}

        # save objectives with timeseries
        for idx, name in enumerate(objectives_list_timeseries_name):
            for year in self.timepoints_to_save:
                name_year = name + str(year)
                timestep = (year - start_year) / tstep
                self.data_dict[name_year] = objectives_list_timeseries[idx][
                    int(timestep)
                ]

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
                    timestep_list.append(
                        supplementary_list_timeseries[idx][region][int(timestep)]
                    )
                self.data_dict[name_year] = timestep_list

        # save additional data_dict in quintile 5 X 12 per year format
        for idx, name in enumerate(supplementary_list_quintile_name):
            for year in self.timepoints_to_save:
                name_year = name + str(year)
                self.data_dict[name_year] = supplementary_list_quintile[idx][
                    year
                ].tolist()

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

        if np.isnan(self.intertemporal_consumption_gini):
            self.intertemporal_consumption_gini = 0

        if np.isnan(self.intertemporal_damage_gini):
            self.intertemporal_damage_gini = 40

    def get_emcu(self):
        """
        @return:
            self.emcu: float
        """
        return self.emcu

    # Helper methods for calculating utilities for social welfare functions

    def set_up_weights_related(self, t, irstp_consumption, tstep, CPC):
        """
        @param t: int
        @param irstp_consumption: float
        @param tstep: int
        @param CPC: numpy array (12, 31)
        """

        # social discount factor for utility
        self.discount_factors_utility[:, t] = 1 / (
            (1 + irstp_consumption) ** (tstep * t)
        )

        # period utilities = utilities for eac region at each time
        # instantaneous welfare without welfare weights
        self.period_utilities[:, t] = (1 / (1 - self.emcu)) * (CPC[:, t]) ** (
            1 - self.emcu
        ) + 1
        # Should it be -1 in the end because that's what the CRRA equation says

        # welfare for utility for each region and each time
        self.utility_welfares[:, t] = (
            self.period_utilities[:, t]
            * self.region_pop[:, t]
            * self.discount_factors_utility[:, t]
        )

        # cumulative period utilty without welfare weights
        self.cum_utility_welfares[:, 0] = (
            self.cum_utility_welfares[:, t - 1] + self.utility_welfares[:, t]
        )

        # Instantaneous utility function with welfare weights
        self.inst_util_ww[:, t] = self.period_utilities[:, t] * self.Alpha_data[:, t]

    def compute_alternative_principles_objectives(
        self,
        t,
        year,
        CPC,
        damages,
        CPC_post_damage,
        CPC_lo,
        climate_impact_relative_to_capita,
        Y,
    ):
        """
        @param t: int
        @param year: int
        @param CPC: numpy array (12, 31)
        @param damages: numpy array (12, 31)
        @param CPC_post_damage: dictionary
        @param CPC_lo: float
        @param climate_impact_relative_to_capita: dictionary
        @param Y: numpy array (12, 31)
        """

        # cummulative utility with ww
        self.reg_cum_util[:, t] = self.reg_cum_util[:, t - 1] + self.per_util_ww[:, t]

        # scale utility with weights derived from the Excel
        if t == 30:
            self.reg_util = (
                10 * self.multiplutacive_scaling_weights[:, 0] * self.reg_cum_util[:, t]
                + self.additative_scaling_weights[:, 0]
                - self.additative_scaling_weights[:, 2]
            )

        # calculate worldwide utility
        self.utility = self.reg_util.sum()

        # ###### Gini calculations INTERTEMPORAL #########
        # CPC is floored on minimum value
        CPC[:, t] = np.where(CPC[:, t] > CPC_lo, CPC[:, t], CPC_lo)

        self.average_world_CPC[t] = CPC[:, t].sum(axis=0) / self.n_regions

        if t == 30:
            input_gini_inter_cpc = self.average_world_CPC

            diffsum = 0
            for i, xi in enumerate(input_gini_inter_cpc[:-1], 1):
                diffsum += np.sum(np.abs(xi - input_gini_inter_cpc[i:]))

            self.intertemporal_consumption_gini = diffsum / (
                (len(input_gini_inter_cpc) ** 2) * np.mean(input_gini_inter_cpc)
            )

        # intertemporal climate impact Gini
        self.average_regional_impact[t] = damages[:, t].sum(axis=0) / self.n_regions

        # Impact is floored on minimum value
        if t == 30:
            input_gini_inter = self.average_regional_impact

            diffsum_inter = 0

            for i, xi in enumerate(input_gini_inter[:-1], 1):
                diffsum_inter += np.sum(np.abs(xi - input_gini_inter[i:]))

            self.intertemporal_damage_gini = diffsum_inter / (
                (len(input_gini_inter) ** 2) * np.mean(input_gini_inter)
            )

        # ###### Gini calculations INTRATEMPORAL #########
        # calculate Cini as measure of current inequality in welfare (intragenerational)

        # CPC is floored on minimum value
        input_gini_intra = CPC[:, t]

        diffsum = 0
        for i, xi in enumerate(input_gini_intra[:-1], 1):
            diffsum += np.sum(np.abs(xi - input_gini_intra[i:]))

        self.CPC_intra_gini[t] = diffsum / (
            (len(input_gini_intra) ** 2) * np.mean(input_gini_intra)
        )

        # calculate Gini as measure of current inequality in climate impact (per dollar consumption)
        # (intragenerational)
        # self.climate_impact_per_dollar_consumption[:, t] = np.where(
        #     damages[:, t] < 0.001, CPC[:, t], damages[:, t] / CPC[:, t]
        # )

        self.climate_impact_per_dollar_consumption[:, t] = damages[:, t] / CPC[:, t]

        input_gini_intra_impact = self.climate_impact_per_dollar_consumption[:, t]

        diffsum = 0
        for i, xi in enumerate(input_gini_intra_impact[:-1], 1):
            diffsum += np.sum(np.abs(xi - input_gini_intra_impact[i:]))

        self.climate_impact_per_dollar_gini[t] = \
            diffsum / ((len(input_gini_intra_impact) ** 2) * np.mean(input_gini_intra_impact))

        # sufficientarian objectives
        # growth by the world
        self.average_world_CPC[t] = CPC[:, t].sum() / self.n_regions
        self.average_growth_CPC[t] = (
            self.average_world_CPC[t] - self.average_world_CPC[t - 1]
        ) / (self.average_world_CPC[t - 1])

        # sufficientarian threshold adjusted by the growth of the average world economy
        self.sufficientarian_consumption_threshold[
            t
        ] = self.sufficientarian_consumption_threshold[t - 1] * (
            1 + self.average_growth_CPC[t]
        )

        # calculate instantaneous welfare equivalent of minimum capita per head
        self.inst_util_thres[t] = (1 / (1 - self.emcu)) * (
            self.sufficientarian_consumption_threshold[t]
        ) ** (1 - self.emcu) + 1

        # calculate instantaneous welfare equivalent of threshold
        self.inst_util_thres_ww[:, t] = self.inst_util_thres[t] * self.Alpha_data[:, t]

        # calculate utility equivalent for every income quintile and scale with welfare weights for comparison
        self.quintile_inst_util[year] = (1 / (1 - self.emcu)) * (
            CPC_post_damage[year]
        ) ** (1 - self.emcu) + 1
        self.quintile_inst_util_ww[year] = (
            self.quintile_inst_util[year] * self.Alpha_data[:, t]
        )

        utility_per_income_share = self.quintile_inst_util_ww[year]
        list_timestep = []

        for quintile in range(0, 5):
            for region in range(0, self.n_regions):
                if utility_per_income_share[quintile, region] < self.inst_util_thres_ww[region, t]:
                    self.population_below_consumption_threshold[t] = (
                        self.population_below_consumption_threshold[t]
                        + self.region_pop[region, t] * 1 / 5
                    )
                    self.utility_distance_threshold[region, t] = (
                        self.inst_util_thres_ww[region, t]
                        - utility_per_income_share[quintile, region]
                    )

                    list_timestep.append(self.regions_list[region])

        self.regions_below_consumption_threshold.append(list_timestep)

        # minimize max distance to threshold
        self.max_utility_distance_threshold[t] = self.utility_distance_threshold[
            :, t
        ].max()

        self.compute_sufficentarian_damage_objectives(t, damages, CPC)

        # prioritarian objectives
        self.worst_off_income_class[t] = CPC_post_damage[year][0].min()
        self.worst_off_damage[t] = climate_impact_relative_to_capita[year][0].max()

        # Utilitarian objectives
        self.global_damages[t] = damages[:, t].sum(axis=0)
        self.global_ouput[t] = Y[:, t].sum(axis=0)
        self.global_per_util_ww[t] = self.per_util_ww[:, t].sum(axis=0)

    def compute_welfare_disutility(self, damages, emdd, tstep, t, irstp_damage):
        """
        Takes damages to compute damages per capita, disutility of damages, and welfare of disutility.
        @param damages: numpy array (12, 31)
        @param irstp_damage: initial rate of social time preference for damages
        @param emdd: float: risk aversion for damages (an uncertainty factor)
        @param tstep: int
        @param t: int: time
        """

        # damages per capita
        self.dpc[:, t] = damages[:, t] * 1000 / self.region_pop[:, t]
        self.dpc[:, t] = np.where(
            self.dpc[:, t] > self.dpc_lo, self.dpc[:, t], self.dpc_lo
        )

        # unweighted and undiscounted diutilities
        if emdd == 1.00:
            self.period_disutilities[:, t] = np.log(self.dpc[:, t])
        else:
            self.period_disutilities[:, t] = self.dpc[:, t] ** (1 - emdd) / (1 - emdd)
        self.period_disutilities[:, t] = np.where(
            self.period_disutilities[:, t] > self.inst_disutil_lo,
            self.period_disutilities[:, t],
            self.inst_disutil_lo,
        )

        # disutility with welfare weights
        self.inst_disutility_ww[:, t] = (
            self.period_disutilities[:, t] * self.Alpha_data[:, t]
        )

        # rate of change of damage
        if t == 0:
            previous_dpc = np.zeros((self.region_pop.shape[0], 1)) + 0.000000001
        else:
            previous_dpc = self.dpc[:, t - 1]
        self.dam_g[:, t] = (self.dpc[:, t] - previous_dpc) / previous_dpc

        # endogenous rate social rate of damage
        self.rho[:, t] = irstp_damage + emdd * self.dam_g[:, t]

        # discount factor for disutility
        self.discount_factors_disutility[:, t] = 1 / (1 + self.rho[:, t] ** (tstep * t))
        self.discount_factors_disutility[:, t] = np.where(
            self.discount_factors_disutility[:, t] > self.sdr_dam_lo,
            self.discount_factors_disutility[:, t],
            self.sdr_dam_lo,
        )

        # welfare = discounted disutility
        self.per_disutility_ww[:, t] = (
            self.inst_disutility_ww[:, t]
            * self.region_pop[:, t]
            * self.discount_factors_disutility[:, t]
        )

        # spatially aggregated disutility
        self.global_per_disutility_ww = self.per_disutility_ww.sum(axis=0) / 1000

        # Totally aggregated disutility

        # cummulative disutility with ww
        if t >= 1:
            self.reg_cum_disutil[:, t] = (
                self.reg_cum_disutil[:, t - 1] + self.per_disutility_ww[:, t]
            )

        # scale utility with weights derived from the Excel
        if t == 30:
            self.reg_disutil = (
                self.multiplutacive_scaling_weights[:, 0] * self.reg_cum_disutil[:, t]
                + self.additative_scaling_weights[:, 0]
                - self.additative_scaling_weights[:, 2]
            )

            # calculate worldwide disutility
            self.disutility = self.reg_disutil.sum()

    def compute_sufficentarian_damage_objectives(self, t, damages, CPC):
        """
        Computes the sufficientarian objectives that are based on damages and disutilities.
        """

        list_timestep = []

        relative_damage = np.divide(damages, CPC, out=np.zeros_like(damages), where=(CPC != 0))

        for region in range(self.n_regions):
            if relative_damage[region, t] > self.relative_damage_threshold:
                self.population_above_damage_threshold[t] = (
                    self.population_above_damage_threshold[t]
                    + self.region_pop[region, t]
                )
                self.disutility_distance_threshold[region, t] = np.transpose(
                    damages[region, t] - CPC[region, t] * self.relative_damage_threshold
                )
                list_timestep.append(self.regions_list[region])

        self.regions_above_damage_threshold.append(list_timestep)
        self.max_disutility_distance_threshold[t] = self.disutility_distance_threshold[:, t].max()

    @staticmethod
    def compute_overshoots(temp_atm):
        """
        Compute for each time step how many previous time steps have had an atmospheric temperature increase of more
        than 2 degrees Celsius.
        @param temp_atm: numpy array (31, ): atmospheric temperature increase
        @return
            above_2_degree_timesteps: numpy array (31, )
        """

        non_cummulative_version = temp_atm > 2.0
        non_cummulative_version = non_cummulative_version.astype(float)
        above_2_degree_timesteps = np.cumsum(non_cummulative_version)
        return above_2_degree_timesteps


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
        damages = self.get_values_for_specific_prefix("Damages 2")
        utility = self.get_values_for_specific_prefix("Utility 2")
        disutility = self.get_values_for_specific_prefix("Disutility 2")
        lowest = self.get_values_for_specific_prefix("Lowest income per capita")
        highest = self.get_values_for_specific_prefix("Highest damage per capita")
        distance_consumption = self.get_values_for_specific_prefix("Distance to consumption threshold")
        population_consumption = self.get_values_for_specific_prefix("Population below consumption threshold")
        distance_damage = self.get_values_for_specific_prefix("Distance to damage threshold")
        population_damage = self.get_values_for_specific_prefix("Population above damage threshold")
        consumption_gini = self.get_values_for_specific_prefix("Intratemporal consumption Gini")
        damage_gini = self.get_values_for_specific_prefix("Intratemporal damage Gini")
        temp = self.get_values_for_specific_prefix("Atmospheric Temperature")
        emission = self.get_values_for_specific_prefix("Industrial Emission")
        output = self.get_values_for_specific_prefix("Total Output")
        regions_below_consumption_threshold = self.data_dict["Regions below consumption threshold"]
        regions_above_damage_threshold = self.data_dict["Regions above damage threshold"]
        costs = self.get_values_for_specific_prefix("Costs 2")
        above_2_degree_timesteps = self.get_values_for_specific_prefix("Temperature overshoot")

        columns = [
            "Damages",
            "Utility",
            "Disutility",
            "Lowest income per capita",
            "Highest climate impact per capita",
            "Distance to consumption threshold",
            "Population below consumption threshold",
            "Distance to damage threshold",
            "Population above damage threshold",
            "Intratemporal consumption Gini",
            "Intratemporal damage Gini",
            "Atmospheric temperature",
            "Temperature overshoot",
            "Industrial emission",
            "Total output",
            "Regions below consumption threshold",
            "Regions above damage threshold",
            "Costs",
        ]

        values = list(
            zip(
                damages,
                utility,
                disutility,
                lowest,
                highest,
                distance_consumption,
                population_consumption,
                distance_damage,
                population_damage,
                consumption_gini,
                damage_gini,
                temp,
                above_2_degree_timesteps,
                emission,
                output,
                regions_below_consumption_threshold,
                regions_above_damage_threshold,
                costs,
            )
        )

        self.df_main = pd.DataFrame(data=values, index=years, columns=columns)

        # Highly aggregated variables
        self.aggregated_consumption_gini = self.data_dict[
            "Intertemporal consumption Gini"
        ]
        self.aggregated_damage_gini = self.data_dict["Intertemporal damage Gini"]
        self.aggregated_utility = self.data_dict["Total Aggregated Utility"]
        self.aggregated_disutility = self.data_dict["Total Aggregated Disutility"]
        self.aggregated_costs = self.data_dict["Total Aggregated Costs"]
        self.aggregated_overshoots = self.data_dict["Total temperature overshoot"]

        # CPC dataframe
        cpc = self.get_values_for_specific_prefix("CPC 2")
        columns = regions_list
        self.df_cpc = pd.DataFrame(cpc, index=years, columns=columns)

        # Population dataframe
        population_consumption = self.get_values_for_specific_prefix("Population 2")
        columns = regions_list
        self.df_population = pd.DataFrame(
            population_consumption, index=years, columns=columns
        )

        # CPC pre damage dataframe
        cpc_pre = self.get_values_for_specific_prefix("CPC pre")
        cpc_pre = np.array(cpc_pre)
        cpc_pre = np.transpose(cpc_pre, (2, 0, 1))
        columns = regions_list
        self.df_cpc_pre_damage = pd.DataFrame(
            list(zip(*cpc_pre)), index=years, columns=columns
        )

        # CPC post damage dataframe
        cpc_post = self.get_values_for_specific_prefix("CPC post")
        cpc_post = np.array(cpc_post)
        cpc_post = np.transpose(cpc_post, (2, 0, 1))
        columns = regions_list
        self.df_cpc_post_damage = pd.DataFrame(
            list(zip(*cpc_post)), index=years, columns=columns
        )

    def get_values_for_specific_prefix(self, prefix="Damages 2"):
        """
        Find the values for a specific key in self.data_dict where the key string starts with the given d_type.
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
