"""
This module contains the economic part of the PyRICE model.
"""


from model.enumerations import ModelSpec, DamageFunction
import numpy as np


class EconomyModel:
    """
    This sub-model describes the neoclassical economic part of the PyRICE model.
    """

    def __init__(self, data_sets, steps, scenario_cback, regions_list):
        """
        @param data_sets: DataSets
        @param steps: int (31)
        @param scenario_cback: int: scenario that specifies the cost for backstop technology
        @param regions_list: list with 12 regions as strings
        """

        self.data_sets = data_sets
        self.regions_list = regions_list
        self.n_regions = len(regions_list)

        # Population parameters
        self.region_pop_gr = data_sets.RICE_input.iloc[0:12, 1:].to_numpy()

        # Get population data_dict for 2005
        self.population2005 = data_sets.RICE_DATA.iloc[19:31, 0].to_numpy()
        self.region_pop = np.zeros((self.n_regions, steps))

        # get regional series for factor productivity growth
        self.tfpgr_region = data_sets.RICE_DATA.iloc[52:64, 1:32].to_numpy()

        # get initial values for various parameters
        self.initails_par = data_sets.RICE_PARAMETER.iloc[33:40, 5:17].to_numpy()
        self.initials_par = self.initails_par.transpose()

        # setting up total factor productivity
        self.tfp_2005 = self.initials_par[:, 5]
        self.tfp_region = np.zeros((self.n_regions, steps))

        # setting up Capital Stock parameters
        self.k_2005 = self.initials_par[:, 4]
        self.k_region = np.zeros((self.n_regions, steps))
        self.dk = 0.1
        self.gama = 0.3

        # setting up total output dataframes
        self.Y_gross = np.zeros((self.n_regions, steps))
        self.ynet = np.zeros((self.n_regions, steps))
        self.damages = np.zeros((self.n_regions, steps))
        self.dam_frac = np.zeros((self.n_regions, steps))

        # Dataframes for emissions, economy and utility
        self.Eind = np.zeros((self.n_regions, steps))
        self.E = np.zeros((self.n_regions, steps))
        self.Etree = np.zeros((self.n_regions, steps))
        self.cumetree = np.zeros((self.n_regions, steps))
        self.CCA = np.zeros((self.n_regions, steps))
        self.CCA_tot = np.zeros((self.n_regions, steps))
        self.Abatement_cost = np.zeros((self.n_regions, steps))
        self.Abatement_cost_RATIO = np.zeros((self.n_regions, steps))
        self.Mabatement_cost = np.zeros((self.n_regions, steps))
        self.CPRICE = np.zeros((self.n_regions, steps))

        # economy parameters per region
        self.Y = np.zeros((self.n_regions, steps))
        self.I = np.zeros((self.n_regions, steps))
        self.C = np.zeros((self.n_regions, steps))
        self.CPC = np.zeros((self.n_regions, steps))

        # Disaggregated consumption tallys
        self.CPC_post_damage = {}
        self.CPC_pre_damage = {}
        self.pre_damage_total__region_consumption = np.zeros((self.n_regions, steps))

        self.climate_impact_relative_to_capita = {}

        # Output-to-Emission
        # Change in sigma: the cumulative improvement in energy efficiency)
        self.sigma_growth_data = data_sets.RICE_DATA.iloc[70:82, 1:6].to_numpy()
        self.Emissions_parameter = (
            data_sets.RICE_PARAMETER.iloc[65:70, 5:17].to_numpy().transpose()
        )

        # set up dataframe for saving_results CO2 to output ratio
        self.Sigma_gr = np.zeros((self.n_regions, steps))
        self.Sigma_gr_RICE = np.zeros((self.n_regions, steps))

        # CO2-equivalent-emissions growth to output ratio in 2005
        self.Sigma_gr[:, 0] = self.sigma_growth_data[:, 0]

        # Fraction of emissions under control based on the Paris Agreement
        # US withdrawal would change the value to 0.7086
        # https://climateanalytics.org/briefings/ratification-tracker/ (0.8875)
        self.partfract2005 = 1

        # Fraction of emissions under control at full time
        self.partfractfull = 1.0

        # Decline rate of decarbonization (per period)
        self.decl_sigma_gr = -0.001

        # Carbon emissions from land 2010 [GtCO2 per year]
        self.eland0 = 1.6
        # Decline rate of land emissions (per period) CHECKED
        self.ecl_land = 0.2

        # Emission data_dict
        self.emission_factor = data_sets.RICE_DATA.iloc[87:99, 6].to_numpy()
        self.Eland0 = 1.6  # (RICE2010 OPT)

        # Cost of abatement
        self.abatement_data = (
            data_sets.RICE_PARAMETER.iloc[56:60, 5:17].to_numpy().transpose()
        )
        self.pbacktime = np.zeros((self.n_regions, steps))
        self.cost1 = np.zeros((self.n_regions, steps))

        # CO2 to economy ratio
        self.sigma_region = np.zeros((self.n_regions, steps))
        self.sigma_region[:, 0] = self.Emissions_parameter[:, 2]

        # Cback per region
        ratio_backstop_world = np.array(
            ([0.9, 1.4, 1.4, 0.6, 0.6, 0.7, 1.1, 1.0, 1.1, 1.3, 1.1, 1.2])
        )

        if scenario_cback == 0:  # SSP LOW SCENARIO
            self.cback = 1.260

        if scenario_cback == 1:  # SSP HIGH SCENARIO
            self.cback = 1.260 * 1.5

        self.cback_region = self.cback * ratio_backstop_world

        # Constations for backstop costs
        self.ratio_asymptotic = self.abatement_data[:, 2]
        self.decl_back_gr = self.abatement_data[:, 3]
        self.expcost2 = 2.8  # RICE 2010 OPT

    def init_economic_parameters(
        self, damage_function, damage_parameters, temp_atm, dam_frac_global, miu
    ):
        """
        @param damage_function: DamageFunction
        @param damage_parameters: numpy array (12, 8)
        @param temp_atm: numpy array (31,): atmospheric temperature
        @param dam_frac_global: numpy array
        @param miu: numpy array (12, 31)
        """

        # Insert population at 2005 for all regions
        self.region_pop[:, 0] = self.population2005

        # total factor production at 2005
        self.tfp_region[:, 0] = self.tfp_2005

        # initial capital in 2005
        self.k_region[:, 0] = self.k_2005

        # Gama: Capital elasticity in production function
        self.Y_gross[:, 0] = (
            self.tfp_region[:, 0]
            * ((self.region_pop[:, 0] / 1000) ** (1 - self.gama))
            * (self.k_region[:, 0] ** self.gama)
        )

        # original RICE parameters dam_frac with SLR
        if damage_function == DamageFunction.NORDHAUS:
            self.dam_frac[:, 0] = (
                damage_parameters[:, 0] * temp_atm[0]
                + damage_parameters[:, 1] * (temp_atm[0] ** damage_parameters[:, 2])
            ) * 0.01

        # Damage function Newbold & Daigneault
        elif damage_function == DamageFunction.NEWBOLD:
            dam_frac_global[0] = 1 - (np.exp(-0.0025 * temp_atm[0] * 2.45))

            # translate global damage frac to regional damage frac with factor as used in RICE
            self.dam_frac[:, 0] = (
                dam_frac_global[0] * self.data_sets.RICE_regional_damage_factor[:, 0]
            )

        # Damage function Weitzman
        elif damage_function == DamageFunction.WEITZMAN:
            dam_frac_global[0] = 1 - 1 / (
                1 + 0.0028388 ** 2 + 0.0000050703 * (temp_atm[0] * 6.754)
            )

            # translate global damage frac to regional damage frac with factor as used in RICE
            self.dam_frac[:, 0] = (
                dam_frac_global[0] * self.data_sets.RICE_regional_damage_factor[:, 0]
            )

        # Net output damages
        self.ynet[:, 0] = self.Y_gross[:, 0] / (1.0 + self.dam_frac[:, 0])

        # Damages in 2005
        self.damages[:, 0] = self.Y_gross[:, 0] - self.ynet[:, 0]

        # Cost of backstop
        self.pbacktime[:, 0] = self.cback_region

        # Adjusted cost for backstop
        self.cost1[:, 0] = (
            self.pbacktime[:, 0] * self.sigma_region[:, 0] / self.expcost2
        )

        # Emissions from land change use
        self.Etree[:, 0] = self.Emissions_parameter[:, 3]
        self.cumetree[:, 0] = self.Emissions_parameter[:, 3]

        # industrial emissions 2005
        self.Eind[:, 0] = self.sigma_region[:, 0] * self.Y_gross[:, 0] * (1 - miu[:, 0])

        # initialize initial emissions
        self.E[:, 0] = self.Eind[:, 0] + self.Etree[:, 0]
        self.CCA[:, 0] = self.Eind[:, 0]
        self.CCA_tot[:, 0] = self.CCA[:, 0] + self.cumetree[:, 0]

    def init_net_economy(self, miu, S, elasticity_of_damages):
        """
        @param miu: numpy array (12, 31)
        @param S: numpy array (12, 31): savings rates as numpy array with dimensions (regions, steps)
        @param elasticity_of_damages: int: damage relation for lower income groups
        @return:
            self.climate_impact_relative_to_capita: dictionary
            self.CPC_post_damage: dictionary
            self.CPC: numpy array (12, 31)
            self.region_pop: numpy array (12, 31)
            self.damages: numpy array (12, 31)
            self.Y: numpy array (12, 31)
        """

        # Cost of climate_model change to economy
        # Abatement cost ratio of output
        self.Abatement_cost_RATIO[:, 0] = self.cost1[:, 0] * (
            miu[:, 0] ** self.expcost2
        )

        # Abatement cost total
        self.Abatement_cost[:, 0] = self.Y_gross[:, 0] * self.Abatement_cost_RATIO[:, 0]

        # Marginal abatement cost
        self.Mabatement_cost[:, 0] = self.pbacktime[:, 0] * miu[:, 0] ** (
            self.expcost2 - 1
        )

        # Resulting carbon_model price
        self.CPRICE[:, 0] = (
            self.pbacktime[:, 0] * 1000 * (miu[:, 0] ** (self.expcost2 - 1))
        )

        # Gross world product (net of abatement and damages)
        self.Y[:, 0] = self.ynet[:, 0] - self.Abatement_cost[:, 0]

        # #############  Investments & Savings  #########################

        # investments per region given the savings rate
        self.I[:, 0] = S[:, 0] * self.Y[:, 0]

        # consumption given the investments
        self.C[:, 0] = self.Y[:, 0] - self.I[:, 0]

        # calculate pre damage consumption aggregated per region
        self.pre_damage_total__region_consumption[:, 0] = (
            self.C[:, 0] + self.damages[:, 0]
        )

        # damage share elasticity function derived from Denig et al 2015
        self.damage_share = self.data_sets.RICE_income_shares ** elasticity_of_damages
        sum_damage = np.sum(self.damage_share, axis=1)

        for i in range(0, self.n_regions):
            self.damage_share[i, :] = self.damage_share[i, :] / sum_damage[i]

        # calculate disaggregated per capita consumption based on income shares BEFORE damages
        self.CPC_pre_damage[2005] = (
            (
                self.pre_damage_total__region_consumption[:, 0]
                * self.data_sets.RICE_income_shares.transpose()
            )
            / (self.region_pop[:, 0] * (1 / 5))
        ) * 1000

        # calculate disaggregated per capita consumption based on income shares AFTER damages
        self.CPC_post_damage[2005] = self.CPC_pre_damage[2005] - (
            (
                (self.damages[:, 0] * self.damage_share.transpose())
                / (self.region_pop[:, 0] * (1 / 5))
            )
            * 1000
        )

        # calculate damage per consumpion in thousands of US dollarsa
        self.climate_impact_relative_to_capita[2005] = (
            (self.damages[:, 0] * self.damage_share.transpose() * 10 ** 12)
            / (0.2 * self.region_pop[:, 0] * 10 ** 6)
        ) / (self.CPC_post_damage[2005] * 1000)

        # consumption per capita
        self.CPC[:, 0] = (1000 * self.C[:, 0]) / self.region_pop[:, 0]

        return (
            self.climate_impact_relative_to_capita,
            self.CPC_post_damage,
            self.CPC,
            self.region_pop,
            self.damages,
            self.Y,
        )

    def run_gross_economy(
        self,
        scenario_pop_gdp,
        tstep,
        t,
        longrun_scenario,
        long_run_nordhaus_pop_gr,
        long_run_nordhaus_tfp_gr,
        long_run_nordhaus_sigma,
        scenario_sigma,
        model_spec,
        miu,
        limmiu,
        miu_period,
        fosslim,
    ):
        """
        @param scenario_pop_gdp: int
        @param tstep: int: time step size
        @param t: int: current time step
        @param longrun_scenario: int
        @param long_run_nordhaus_pop_gr: int
        @param long_run_nordhaus_tfp_gr: int
        @param long_run_nordhaus_sigma: int
        @param scenario_sigma: int
        @param model_spec: ModelSpecification
        @param miu: numpy array (12, 31)
        @param limmiu: int
        @param miu_period: numpy array (12, 1)
        @param fosslim: int
        @return:
            self.Y_gross: numpy array (12, 31)
            self.E: numpy array (12, 31)
        """

        # use SSP population projections if not base with right SSP scenario (SSP1, SSP2 etc.)
        if scenario_pop_gdp != 0:
            # load population and gdp projections from SSP scenarios on first timestep
            if t == 1:
                for region in range(0, self.n_regions):
                    self.region_pop[region, :] = self.data_sets.POP_ssp.iloc[
                        :, (scenario_pop_gdp - 1) + (region * 5)
                    ]

                    self.Y_gross[region, :] = (
                        self.data_sets.GDP_ssp.iloc[
                            :, (scenario_pop_gdp - 1) + (region * 5)
                        ]
                        / 1000
                    )

            self.Y_gross[:, t] = np.where(self.Y_gross[:, t] > 0, self.Y_gross[:, t], 0)

            self.k_region[:, t] = (
                self.k_region[:, t - 1] * ((1 - self.dk) ** tstep)
                + tstep * self.I[:, t - 1]
            )

            # calculate tfp based on GDP projections by SSP's
            self.tfp_region[:, t] = self.Y_gross[:, t] / (
                (self.k_region[:, t] ** self.gama)
                * (self.region_pop[:, t] / 1000) ** (1 - self.gama)
            )

        # Use base projections for population and TFP and sigma growth
        if scenario_pop_gdp == 0 and longrun_scenario == 0:

            # calculate population at time t
            self.region_pop[:, t] = self.region_pop[:, t - 1] * 2.71828 ** (
                self.region_pop_gr[:, t] * 10
            )

            # TOTAL FACTOR PRODUCTIVITY level according to RICE base
            self.tfp_region[:, t] = self.tfp_region[:, t - 1] * 2.71828 ** (
                self.tfpgr_region[:, t] * 10
            )

            # determine capital stock at time t
            self.k_region[:, t] = (
                self.k_region[:, t - 1] * ((1 - self.dk) ** tstep)
                + tstep * self.I[:, t - 1]
            )

            # lower bound capital
            self.k_region[:, t] = np.where(
                self.k_region[:, t] > 1, self.k_region[:, t], 1
            )

            # determine Ygross at time t
            self.Y_gross[:, t] = (
                self.tfp_region[:, t]
                * ((self.region_pop[:, t] / 1000) ** (1 - self.gama))
                * (self.k_region[:, t] ** self.gama)
            )

            # lower bound Y_Gross
            self.Y_gross[:, t] = np.where(self.Y_gross[:, t] > 0, self.Y_gross[:, t], 0)

        # LONG RUN EXPLORATORY ANALAYSIS USING RICE REFERENCE SCENARIO
        if longrun_scenario == 1:
            # calculate population at time t adjust with uncertainty range
            self.region_pop[:, t] = self.region_pop[:, t - 1] * 2.71828 ** (
                self.region_pop_gr[:, t] * long_run_nordhaus_pop_gr * 10
            )

            # TOTAL FACTOR PRODUCTIVITY level according to RICE base adjust with uncertainty range
            self.tfp_region[:, t] = self.tfp_region[:, t - 1] * 2.71828 ** (
                self.tfpgr_region[:, t] * long_run_nordhaus_tfp_gr * 10
            )

            # determine capital stock at time t
            self.k_region[:, t] = (
                self.k_region[:, t - 1] * ((1 - self.dk) ** tstep)
                + tstep * self.I[:, t - 1]
            )

            # lower bound capital
            self.k_region[:, t] = np.where(
                self.k_region[:, t] > 1, self.k_region[:, t], 1
            )

            # determine Ygross at time t
            self.Y_gross[:, t] = (
                self.tfp_region[:, t]
                * ((self.region_pop[:, t] / 1000) ** (1 - self.gama))
                * (self.k_region[:, t] ** self.gama)
            )

            # lower bound Y_Gross
            self.Y_gross[:, t] = np.where(self.Y_gross[:, t] > 0, self.Y_gross[:, t], 0)

            # calculate the sigma growth adjust with uncertainty range and the emission rate development
            if t == 1:
                self.Sigma_gr[:, t] = self.sigma_growth_data[:, 4] + (
                    self.sigma_growth_data[:, 2] - self.sigma_growth_data[:, 4]
                )

                self.sigma_region[:, t] = (
                    self.sigma_region[:, t - 1]
                    * (2.71828 ** (self.Sigma_gr[:, t] * 10))
                    * self.emission_factor
                )

            if t > 1:
                self.Sigma_gr[:, t] = (
                    self.sigma_growth_data[:, 4]
                    + (self.Sigma_gr[:, t - 1] - self.sigma_growth_data[:, 4])
                    * (1 - self.sigma_growth_data[:, 3])
                ) * long_run_nordhaus_sigma

                self.sigma_region[:, t] = self.sigma_region[:, t - 1] * (
                    2.71828 ** (self.Sigma_gr[:, t] * 10)
                )

        if longrun_scenario != 1:
            if scenario_sigma == 0:  # medium SSP AEEI (base RICE)

                # calculate the sigma growth and the emission rate development
                if t == 1:
                    self.Sigma_gr[:, t] = self.sigma_growth_data[:, 4] + (
                        self.sigma_growth_data[:, 2] - self.sigma_growth_data[:, 4]
                    )

                    self.sigma_region[:, t] = (
                        self.sigma_region[:, t - 1]
                        * (2.71828 ** (self.Sigma_gr[:, t] * 10))
                        * self.emission_factor
                    )

                if t > 1:
                    self.Sigma_gr[:, t] = self.sigma_growth_data[:, 4] + (
                        self.Sigma_gr[:, t - 1] - self.sigma_growth_data[:, 4]
                    ) * (1 - self.sigma_growth_data[:, 3])

                    self.sigma_region[:, t] = self.sigma_region[:, t - 1] * (
                        2.71828 ** (self.Sigma_gr[:, t] * 10)
                    )

            if scenario_sigma == 1:  # low SSP AEEI

                # calculate the sigma growth and the emission rate development
                if t == 1:
                    self.Sigma_gr_RICE[:, t] = self.sigma_growth_data[:, 4] + (
                        self.sigma_growth_data[:, 2] - self.sigma_growth_data[:, 4]
                    )

                    self.Sigma_gr[:, t] = self.Sigma_gr_RICE[:, t]

                    self.sigma_region[:, t] = (
                        self.sigma_region[:, t - 1]
                        * (2.71828 ** (self.Sigma_gr[:, t] * 10))
                        * self.emission_factor
                    )

                if t > 1:
                    self.Sigma_gr_RICE[:, t] = self.sigma_growth_data[:, 4] + (
                        self.Sigma_gr_RICE[:, t - 1] - self.sigma_growth_data[:, 4]
                    ) * (1 - self.sigma_growth_data[:, 3])

                    self.Sigma_gr[:, t] = self.Sigma_gr_RICE[:, t] * 0.5

                    self.sigma_region[:, t] = self.sigma_region[:, t - 1] * (
                        2.71828 ** (self.Sigma_gr[:, t] * 10)
                    )

            if scenario_sigma == 2:  # high SSP AEEI
                # calculate the sigma growth and the emission rate development
                if t == 1:
                    self.Sigma_gr_RICE[:, t] = self.sigma_growth_data[:, 4] + (
                        self.sigma_growth_data[:, 2] - self.sigma_growth_data[:, 4]
                    )
                    self.Sigma_gr[:, t] = self.Sigma_gr_RICE[:, t]
                    self.sigma_region[:, t] = (
                        self.sigma_region[:, t - 1]
                        * (2.71828 ** (self.Sigma_gr[:, t] * 10))
                        * self.emission_factor
                    )

                if t > 1:
                    self.Sigma_gr_RICE[:, t] = self.sigma_growth_data[:, 4] + (
                        self.Sigma_gr_RICE[:, t - 1] - self.sigma_growth_data[:, 4]
                    ) * (1 - self.sigma_growth_data[:, 3])

                    self.Sigma_gr[:, t] = self.Sigma_gr_RICE[:, t] * 1.5

                    self.sigma_region[:, t] = self.sigma_region[:, t - 1] * (
                        2.71828 ** (self.Sigma_gr[:, t] * 10)
                    )

        # calculate emission control rate under STANDARD
        if model_spec == ModelSpec.STANDARD:

            # control rate is maximum after target period, otherwise linearly increase towards that point from t[0]
            if t > 1:
                for index in range(0, self.n_regions):
                    calculated_miu = (
                        miu[index, t - 1] + (limmiu - miu[index, 1]) / miu_period[index]
                    )
                    miu[index, t] = min(calculated_miu, limmiu)

        if model_spec == ModelSpec.Validation_2:
            if t > 1:
                for index in range(0, self.n_regions):
                    calculated_miu = (
                        miu[index, t - 1] + (limmiu - miu[index, 1]) / miu_period[index]
                    )
                    miu[index, t] = min(calculated_miu, limmiu)

        # Yearly emissions
        self.Eind[:, t] = self.sigma_region[:, t] * self.Y_gross[:, t] * (1 - miu[:, t])

        # yearly emissions from land change
        self.Etree[:, t] = self.Etree[:, t - 1] * (1 - self.Emissions_parameter[:, 4])

        # yearly combined emissions
        self.E[:, t] = self.Eind[:, t] + self.Etree[:, t]

        # cummulative emissions from land change
        self.cumetree[:, t] = self.cumetree[:, t - 1] + self.Etree[:, t] * 10

        # cummulative emissions from industry
        self.CCA[:, t] = self.CCA[:, t - 1] + self.Eind[:, t] * 10

        self.CCA[:, t] = np.where(self.CCA[:, t] < fosslim, self.CCA[:, t], fosslim)

        # total cummulative emissions
        self.CCA_tot = self.CCA[:, t] + self.cumetree[:, t]

        return self.E, self.Y_gross

    def run_net_economy(
        self,
        t,
        year,
        damage_function,
        damage_parameters,
        temp_atm,
        SLRDAMAGES,
        dam_frac_global,
        miu,
        elasticity_of_damages,
        S,
        model_spec,
        irstp_consumption,
        sr,
        emcu,
    ):
        """
        @param t: int
        @param year: int
        @param damage_function: DamageFunction
        @param damage_parameters: numpy array (12, 8)
        @param temp_atm: numpy array (31,)
        @param SLRDAMAGES: numpy array (12, 31)
        @param dam_frac_global: numpy array (31,)
        @param miu: numpy array (12, 31)
        @param elasticity_of_damages: int
        @param S: numpy array (12, 31)
        @param model_spec: ModelSpecification
        @param irstp_consumption: float
        @param sr: float
        @param emcu: float: elasticity of marginal utility of consumption
        @return:
            self.CPC: numpy rray (12, 31)
            self.region_pop: numpy array (12, 31)
            self.damages: numpy array (12, 31)
            self.Y: numpy array (12, 31)
            self.CPC_lo: float
            self.CPC_pre_damage: dictionary
        """

        # original RICE parameters dam_frac
        if damage_function == DamageFunction.NORDHAUS:
            self.dam_frac[:, t] = (
                damage_parameters[:, 0] * temp_atm[t]
                + damage_parameters[:, 1] * (temp_atm[t] ** damage_parameters[:, 2])
            ) * 0.01

            # Determine total damages
            self.damages[:, t] = self.Y_gross[:, t] * (
                self.dam_frac[:, t] + (SLRDAMAGES[:, t] / 100)
            )

        # Damage function Newbold & Daigneault
        elif damage_function == DamageFunction.NEWBOLD:
            dam_frac_global[t] = 1 - (np.exp(-0.0025 * temp_atm[t] ** 2.45))

            # translate global damage frac to regional damage frac with factor as used in RICE
            self.dam_frac[:, t] = (
                dam_frac_global[t] * self.data_sets.RICE_regional_damage_factor[:, t]
            )

            # calculate damages to economy
            self.damages[:, t] = self.Y_gross[:, t] * self.dam_frac[:, t]

        # Damage function Weitzman
        elif damage_function == DamageFunction.WEITZMAN:
            dam_frac_global[t] = 1 - 1 / (
                1 + 0.0028388 ** 2 + 0.0000050703 * (temp_atm[t] ** 6.754)
            )

            # translate global damage frac to regional damage frac with factor as used in RICE
            self.dam_frac[:, t] = (
                dam_frac_global[t] * self.data_sets.RICE_regional_damage_factor[:, t]
            )

            # calculate damages to economy
            self.damages[:, t] = self.Y_gross[:, t] * self.dam_frac[:, t]

        # determine net output damages with damfrac function chosen in previous step
        self.ynet[:, t] = self.Y_gross[:, t] - self.damages[:, t]

        # Backstop price/cback: cost of backstop
        # decline of backstop competitive year (RICE2010 OPT)
        self.backstopcompetitiveyear = 2250
        if year > self.backstopcompetitiveyear:
            self.pbacktime[:, t] = self.pbacktime[:, t - 1] * 0.5
        else:
            self.pbacktime[:, t] = 0.10 * self.cback_region + (
                self.pbacktime[:, t - 1] - 0.1 * self.cback_region
            ) * (1 - self.decl_back_gr)

        # Adjusted cost for backstop
        self.cost1[:, t] = (
            self.pbacktime[:, t] * self.sigma_region[:, t]
        ) / self.expcost2

        # Abtement cost ratio of output
        self.Abatement_cost_RATIO[:, t] = self.cost1[:, t] * (
            miu[:, t] ** self.expcost2
        )
        self.Abatement_cost[:, t] = self.Y_gross[:, t] * self.Abatement_cost_RATIO[:, t]

        # Marginal abatement cost
        self.Mabatement_cost[:, t] = self.pbacktime[:, t] * (
            miu[:, t] ** (self.expcost2 - 1)
        )

        # Resulting carbon_model price
        self.CPRICE[:, t] = (
            self.pbacktime[:, t] * 1000 * (miu[:, t] ** (self.expcost2 - 1))
        )

        # Gross world product (net of abatement and damages)
        self.Y[:, t] = self.ynet[:, t] - abs(self.Abatement_cost[:, t])

        self.Y[:, t] = np.where(self.Y[:, t] > 0, self.Y[:, t], 0)

        # #############  Investments & Savings  #########################
        if not model_spec == ModelSpec.Validation_1:
            # Optimal long-run savings rate used for transversality --> SEE THESIS SHAJEE
            optlrsav = (
                (self.dk + 0.004)
                / (self.dk + 0.004 * emcu + irstp_consumption)
                * self.gama
            )

            if model_spec == ModelSpec.Validation_2:
                if t > 12:
                    S[:, t] = optlrsav
                else:
                    if t > 1:
                        S[:, t] = (optlrsav - S[:, 1]) * t / 12 + S[:, 1]

            if model_spec == ModelSpec.STANDARD:
                if t > 1:
                    S[:, t] = (sr - S[:, 1]) * t / 12 + S[:, 1]
                if t > 12:
                    S[:, t] = sr

        # investments per region given the savings rate
        self.I[:, t] = S[:, t] * self.Y[:, t]

        # check lower bound investments
        self.I[:, t] = np.where(self.I[:, t] > 0, self.I[:, t], 0)

        # set up constraints
        self.c_lo = 2
        self.CPC_lo = 0.01

        # consumption given the investments
        self.C[:, t] = self.Y[:, t] - self.I[:, t]

        # check for lower bound on C
        self.C[:, t] = np.where(self.C[:, t] > self.c_lo, self.C[:, t], self.c_lo)

        # calculate pre damage consumption aggregated per region
        self.pre_damage_total__region_consumption[:, t] = (
            self.C[:, t] + self.damages[:, t]
        )

        # damage share elasticity function derived from Denig et al 2015
        self.damage_share = self.data_sets.RICE_income_shares ** elasticity_of_damages
        sum_damage = np.sum(self.damage_share, axis=1)

        for i in range(0, self.n_regions):
            self.damage_share[i, :] = self.damage_share[i, :] / sum_damage[i]

            # calculate disaggregated per capita consumption based on income shares BEFORE damages
        self.CPC_pre_damage[year] = (
            (
                self.pre_damage_total__region_consumption[:, t]
                * self.data_sets.RICE_income_shares.transpose()
            )
            / (self.region_pop[:, t] * (1 / 5))
        ) * 1000

        # calculate disaggregated per capita consumption based on income shares AFTER damages
        self.CPC_post_damage[year] = self.CPC_pre_damage[year] - (
            (
                (self.damages[:, t] * self.damage_share.transpose())
                / (self.region_pop[:, t] * (1 / 5))
            )
            * 1000
        )

        # check for lower bound on C
        self.CPC_pre_damage[year] = np.where(
            self.CPC_pre_damage[year] > self.CPC_lo,
            self.CPC_pre_damage[year],
            self.CPC_lo,
        )
        self.CPC_post_damage[year] = np.where(
            self.CPC_post_damage[year] > self.CPC_lo,
            self.CPC_post_damage[year],
            self.CPC_lo,
        )

        # calculate damage per quintile equiv
        self.climate_impact_relative_to_capita[year] = (
            (self.damages[:, t] * self.damage_share.transpose() * 10 ** 12)
            / (0.2 * self.region_pop[:, t] * 10 ** 6)
        ) / (self.CPC_pre_damage[year] * 1000)

        self.climate_impact_relative_to_capita[year] = np.where(
            self.climate_impact_relative_to_capita[year] > 1,
            1,
            self.climate_impact_relative_to_capita[year],
        )

        # average consumption per capita per region
        self.CPC[:, t] = (1000 * self.C[:, t]) / self.region_pop[:, t]
        self.CPC[:, t] = np.where(
            self.CPC[:, t] > self.CPC_lo, self.CPC[:, t], self.CPC_lo
        )

        # overeall costs (damages + abatement
        if t == 30:
            self.costs = self.damages + self.Abatement_cost
            self.costs = self.costs.sum(axis=0)

        return (
            self.CPC,
            self.region_pop,
            self.damages,
            self.Y,
            self.CPC_lo,
            self.CPC_pre_damage,
        )

    def get_costs(self):
        """
        Return undiscounted costs (abatement costs + damages)
        @return: self.costs
        """
        return self.costs

    def get_climate_impact_relative_to_capita(self):
        """
        @return: relative climpate impact: dictionary
        """
        return self.climate_impact_relative_to_capita

    def get_cpc_post_damage(self):
        """
        @return: CPC post damage: dictionary
        """
        return self.CPC_post_damage
