"""
This module contains the PyRICe class and its associated methods.
"""

from scipy.stats import norm, cauchy, lognorm
import json

# Import data_dict sets, own enums and model limits
from model.model_limits import *
from model.data_sets import *

# Import submodels
from model.submodels.economy_model import *
from model.submodels.carbon_cycle_model import *
from model.submodels.climate_model import *
from model.submodels.utility_model import *


class PyRICE(object):
    """
    This is the PyRICE model.
    """

    def __init__(
        self,
        tstep=10,
        model_specification=ModelSpec.STANDARD,
        damage_function=DamageFunction.NORDHAUS,
        welfare_function=WelfareFunction.UTILITARIAN,
        overwrite_f=True,
    ):
        """
        @param tstep: int: time step size in years
        @param model_specification: ModelSpecification
        @param damage_function: DamageFunction
        @param welfare_function: WelfareFunction
        @param overwrite_f: Boolean
        """

        # Time related parameters
        self.tstep = tstep
        self.tperiod = []
        self.start_year = 2005
        self.end_year = 2305
        self.steps = int((self.end_year - self.start_year) / self.tstep + 1)  # = 31

        self.model_spec = model_specification
        self.damage_function = damage_function
        self.welfare_function = welfare_function
        self.overwrite_damage_function = overwrite_f

        # arrange simulation timeline
        for i in range(0, self.steps):
            self.tperiod.append((i * self.tstep) + self.start_year)

        self.samples_t2xco2 = self.set_up_climate_sensitivity_distributions()

    def __call__(
        self,
        growth_factor_prio=1,
        prioritarian_discounting=0,
        sufficientarian_discounting=1,
        growth_factor_suf=1,
        ini_suf_threshold_consumption=1.168,
        relative_damage_threshold=0.05,
        egalitarian_discounting=0,
        t2xco2_index=-1,
        t2xco2_dist=0,
        fosslim=6000,
        damage_function=DamageFunction.NORDHAUS,
        scenario_pop_gdp=0,
        scenario_sigma=0,
        scenario_cback=0,
        scenario_elasticity_of_damages=0,
        scenario_limmiu=0,
        longrun_scenario=1,
        long_run_nordhaus_tfp_gr=1,
        long_run_nordhaus_sigma=1,
        long_run_nordhaus_pop_gr=1,
        sr=0.248,
        miu=2135,
        irstp_consumption=0.015,
        irstp_damage=0.015,
        emdd=0.8,
        precision=10,
        **kwargs,
    ):
        """
        @param growth_factor_prio: int: growth factor when prioritarian (0 = no discounting or 1 = conditional_growth)
        @param prioritarian_discounting: int: how much the worst-off consumpt. needs to grow each tick to allow discou.
        @param sufficientarian_discounting: int: 0 = inheritance discounting or, 1 = sustainable growth discounting
        @param growth_factor_suf: int: growth factor when sufficientarian
        @param ini_suf_threshold_consumption: float: initial sufficientarian threshold (based on poverty line $3.20 p/d)
        @param relative_damage_threshold: float: percentage of how high damage can be compared to consumption
        @param egalitarian_discounting: int: discounting when egalitarian (0 = no discouting  or 1 = normal discounting)
        @param t2xco2_index: int: equilibrium temperature impact
        @param t2xco2_dist: int: total factor productivity growth rate
        @param fosslim: int: availability of fossil fuel
        @param damage_function: DamageFunction
        @param scenario_pop_gdp: int: population growth
        @param scenario_sigma: int: CO2 efficiency development (emission to output growth rate)
        @param scenario_cback: int: cost of the backstop technology
        @param scenario_elasticity_of_damages: int:  damage relation for lower income groups
        @param scenario_limmiu: int: 0 = base RICE or 1 = negative emissions possible
        @param longrun_scenario: int: enables long run scenario (0 = long run uncertainty switch off or 1 = switched on)
        @param long_run_nordhaus_tfp_gr: int: range in DICE [0.07, 0.09] in RICE 0.85 - 1.15
        @param long_run_nordhaus_sigma: int: range in DICE [-0.012, -0.008] 0.75 - 1.25
        @param long_run_nordhaus_pop_gr: int: range in DICE [0.1 0.15]   0.75 - 1.25
        @param sr: float: savings rate
        @param miu: int: global emissions target in which emissions are near zero
        @param irstp_consumption: float: initial rate of social time preference of consumption
        @param irstp_damage: float: initial rate of social time preference of damages
        @param emdd: float: coefficient of relative risk aversion for climate damage
        @param precision: int: precision of outcomes, {10, 20, 30}
        @param kwargs:
        @return:
            self.data_dict: dictionary: all outcomes
        """
        # Set up miu_period
        miu_period = (miu - self.start_year) / 10.0

        # Import data_dict sets
        self.data_sets = DataSets()
        self.regions_list = [
            "US",
            "OECD-Europe",
            "Japan",
            "Russia",
            "Non-Russia Eurasia",
            "China",
            "India",
            "Middle East",
            "Africa",
            "Latin America",
            "OHI",
            "Other non-OECD Asia",
        ]

        # Set up levers
        self.set_up_levers(
            sr,
            irstp_consumption,
            irstp_damage,
            fosslim,
            scenario_limmiu,
            scenario_elasticity_of_damages,
            egalitarian_discounting,
            prioritarian_discounting,
            miu_period,
        )

        # Equilibrium temperature impact [dC per doubling CO2]/(3.2 RICE OPT)
        self.t2xco2 = self.samples_t2xco2[t2xco2_dist][t2xco2_index]

        # Choice of the damage function (structural deep uncertainty)  # derived from Lingerewan (2020)
        if self.overwrite_damage_function:
            self.damage_function = damage_function

        # define growth factor uncertainties for sampling
        self.scenario_pop_gdp = scenario_pop_gdp
        self.scenario_sigma = scenario_sigma
        self.scenario_cback = scenario_cback

        # damage parameters excluding SLR from RICE2010
        self.damage_parameters = (
            self.data_sets.RICE_input.iloc[47:55, 1:13].transpose().to_numpy()
        )

        # damage parameters INCLUDING SLR FIT (Dennig et al.)
        self.dam_frac_global = np.zeros(self.steps)

        # Limits of the model
        self.limits = ModelLimits()

        # Sub-model instantiations

        # Instantiate the Carbon Cycle sub-model
        self.carbon_model = CarbonCycleModel(self.steps, self.limits)

        # Instantiate the Climate sub-model
        self.climate_model = ClimateModel(self.steps, self.limits, self.regions_list)

        # Instantiate the economy sub-model
        self.econ_model = EconomyModel(
            self.data_sets, self.steps, scenario_cback, self.regions_list
        )

        # Instantiate the utility sub-model
        self.utility_model = UtilityModel(
            self.steps,
            self.data_sets,
            self.regions_list,
            self.limits,
            self.welfare_function,
            self.damage_function,
            emdd,
        )

        # Set up economic parameters
        self.econ_model.init_economic_parameters(
            self.damage_function,
            self.damage_parameters,
            self.climate_model.temp_atm,
            self.dam_frac_global,
            self.miu,
        )

        # Instantiate the net economy in economic sub-model
        (
            self.climate_impact_relative_to_capita,
            self.CPC_post_damage,
            self.CPC,
            self.region_pop,
            self.damages,
            self.Y,
        ) = self.econ_model.init_net_economy(
            self.miu, self.S, self.elasticity_of_damages
        )

        # Set up Utility
        self.utility_model.set_up_utility(
            ini_suf_threshold_consumption,
            relative_damage_threshold,
            self.climate_impact_relative_to_capita,
            self.CPC_post_damage,
            self.CPC,
            self.region_pop,
            self.damages,
            self.Y,
        )

        # Run model
        for t in range(1, self.steps):

            # keep track of year per timestep for dicts used
            self.year = self.start_year + self.tstep * t

            # Run gross economy in economic sub-model
            E, Y_gross = self.econ_model.run_gross_economy(
                scenario_pop_gdp,
                self.tstep,
                t,
                longrun_scenario,
                long_run_nordhaus_pop_gr,
                long_run_nordhaus_tfp_gr,
                long_run_nordhaus_sigma,
                scenario_sigma,
                self.model_spec,
                self.miu,
                self.limmiu,
                self.miu_period,
                self.fosslim,
            )

            # Run carbon cycle sub-model
            fco22x, forc, self.E_worldwilde_per_year = self.carbon_model.run(t, E)

            # Run climate sub-model
            self.temp_atm = self.climate_model.run(
                t, fco22x, forc, self.t2xco2, Y_gross
            )

            # Run net economy
            (
                self.CPC,
                self.region_pop,
                self.damages,
                self.Y,
                self.CPC_lo,
                self.CPC_pre_damage,
            ) = self.econ_model.run_net_economy(
                t,
                self.year,
                self.damage_function,
                self.damage_parameters,
                self.temp_atm,
                self.climate_model.SLRDAMAGES,
                self.dam_frac_global,
                self.miu,
                self.elasticity_of_damages,
                self.S,
                self.model_spec,
                self.irstp_consumption,
                self.sr,
                self.utility_model.emcu,
            )

            # Compute Utility
            climate_impact_relative_to_capita = (
                self.econ_model.get_climate_impact_relative_to_capita()
            )
            CPC_post_damage = self.econ_model.get_cpc_post_damage()

            self.CPC, self.CPC_post_damage = self.utility_model.run(
                t,
                self.year,
                self.irstp_consumption,
                self.irstp_damage,
                self.tstep,
                growth_factor_prio,
                growth_factor_suf,
                sufficientarian_discounting,
                egalitarian_discounting,
                prioritarian_discounting,
                self.CPC,
                self.region_pop,
                self.damages,
                self.Y,
                self.CPC_lo,
                climate_impact_relative_to_capita,
                CPC_post_damage,
                emdd,
            )

        # Prepare final outcomes of interest
        costs = self.econ_model.get_costs()
        self.data_dict = self.utility_model.get_outcomes(
            self.temp_atm,
            self.E_worldwilde_per_year,
            self.CPC_pre_damage,
            self.CPC_post_damage,
            self.CPC,
            self.start_year,
            self.end_year,
            self.tstep,
            costs,
            precision=precision,
        )

        # Save alternative format of outcomes
        self.data = self.utility_model.data

        return self.data_dict

    @staticmethod
    def set_up_climate_sensitivity_distributions():
        """
        Setting up three distributions for the climate sensitivity; normal lognormal and cauchy
        @return:
            samples_t2xco2: list with three arrays
        """
        minb = 0
        maxb = 20
        nsamples = 1000

        directory = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(directory, 'outputdata', 'ecs_dist_v5.json')) as f:
            d = json.load(f)

        np.random.seed(10)

        samples_norm = np.zeros((0,))

        while samples_norm.shape[0] < nsamples:
            samples = norm.rvs(d["norm"][0], d["norm"][1], nsamples)
            accepted = samples[(samples >= minb) & (samples <= maxb)]
            samples_norm = np.concatenate((samples_norm, accepted), axis=0)
        samples_norm = samples_norm[:nsamples]

        samples_lognorm = np.zeros((0,))

        while samples_lognorm.shape[0] < nsamples:
            samples = lognorm.rvs(
                d["lognorm"][0], d["lognorm"][1], d["lognorm"][2], nsamples
            )
            accepted = samples[(samples >= minb) & (samples <= maxb)]
            samples_lognorm = np.concatenate((samples_lognorm, accepted), axis=0)
        samples_lognorm = samples_lognorm[:nsamples]

        samples_cauchy = np.zeros((0,))

        while samples_cauchy.shape[0] < nsamples:
            samples = cauchy.rvs(d["cauchy"][0], d["cauchy"][1], nsamples)
            accepted = samples[(samples >= minb) & (samples <= maxb)]
            samples_cauchy = np.concatenate((samples_cauchy, accepted), axis=0)
        samples_cauchy = samples_cauchy[:nsamples]

        # extend array with the deterministic value of the nordhaus
        samples_norm = np.append(samples_norm, 3.2)
        samples_lognorm = np.append(samples_lognorm, 3.2)
        samples_cauchy = np.append(samples_cauchy, 3.2)

        samples_t2xco2 = [samples_norm, samples_lognorm, samples_cauchy]

        return samples_t2xco2

    def set_up_levers(
        self,
        sr,
        irstp,
        irstp_damage,
        fosslim,
        scenario_limmiu,
        scenario_elasticity_of_damages,
        egalitarian_discounting,
        prioritarian_discounting,
        miu_period,
    ):
        """
        Setting up the levers.
        @param sr: float: savings rate
        @param irstp: flaot: initial rate of social time preference of consumption
        @param irstp_damage: flaot: initial rate of social time preference of damages
        @param fosslim: int: availability of fossil fuel
        @param scenario_limmiu: int: availability of negative emissions technology (yes, no)
        @param scenario_elasticity_of_damages: int: damage relation for lower income groups
        @param egalitarian_discounting: int: egalitarian discounting
        @param prioritarian_discounting: int: how much the worst-off cons needs to grow each timestep to allow discou.
        @param miu_period: float ((miu - starting_year) / tstep)
        """
        # Get controls from RICE optimal run
        miu_opt_series = self.data_sets.RICE_input.iloc[15:27, 1:].to_numpy()
        sr_opt_series = self.data_sets.RICE_input.iloc[30:42, 1:].to_numpy()

        # Controls with STANDARD sampling
        if self.model_spec == ModelSpec.STANDARD:
            # create frame for savings rate to be sampled
            self.S = np.zeros((12, self.steps))
            self.miu = np.zeros((12, self.steps))

            # set starting MIU for all runs
            self.miu[:, 0:2] = miu_opt_series[:, 0:2]
            self.S[:, 0:2] = sr_opt_series[:, 0:2]

            self.miu_period = np.full((12, 1), miu_period)
            self.sr = sr

        # Get control from RICE2010 - full RICE2010 replicating run
        elif self.model_spec == ModelSpec.Validation_1:
            # set savings rate and control rate as optimized RICE 2010
            self.S = sr_opt_series

            # set emission control rate for the whole run according to RICE2010 opt.
            self.miu = miu_opt_series
            self.irstp_consumption = irstp
            self.miu_period = np.full((12, 1), miu_period)  # new
            self.sr = sr

        # STANDARD Deterministic controls
        elif self.model_spec == ModelSpec.Validation_2:
            # create dataframes for control rate and savings rate
            self.miu = np.zeros((12, self.steps))
            self.S = np.zeros((12, self.steps))

            # set savings rate and control rate as optimized RICE 2010 for the first two timesteps
            self.miu[:, 0:2] = miu_opt_series[:, 0:2]
            self.S[:, 0:2] = sr_opt_series[:, 0:2]

            # set uncertainties that drive MIU
            self.limmiu = 1
            self.irstp_consumption = irstp
            self.miu_period = [12, 15, 15, 10, 10, 11, 13, 13, 13, 14, 13, 14]
            self.sr = sr

        else:
            raise ValueError("Oh, no! Apparently, your model specification is unknown!")

        # define other uncertainties
        self.irstp_consumption = irstp
        self.irstp_damage = irstp_damage
        self.fosslim = fosslim

        # SSP CCS and negative emissions possibilities
        if scenario_limmiu == 0:
            self.limmiu = 1
        elif scenario_limmiu == 1:
            self.limmiu = 1.2

        # Uncertainty of elasticity of damages to consumption
        if scenario_elasticity_of_damages == 0:
            self.elasticity_of_damages = 1
        elif scenario_elasticity_of_damages == 1:
            self.elasticity_of_damages = 0
        elif scenario_elasticity_of_damages == 2:
            self.elasticity_of_damages = -1
        else:
            self.elasticity_of_damages = 0

        # overwrite IRSTP for non discounting levers
        if self.welfare_function == WelfareFunction.PRIORITARIAN:
            if prioritarian_discounting == 0:
                self.irstp_consumption = 0
        elif self.welfare_function == WelfareFunction.EGALITARIAN:
            if egalitarian_discounting == 0:
                self.irstp_consumption = 0

    def get_better_formatted_results(self):
        """
        Returns the outcome data_dict as an Results object which contains 5 dataframes and 3 highly aggregated float
        variables as attributes.
        @return:
            self.utility_model.outcomes: Results
        """

        return self.utility_model.data

    def view_better_formatted_results(self):
        """
        Prints the attributes of the Results object.
        """

        print(
            f"aggregated_utility_gini: \t{self.utility_model.data.aggregated_consumption_gini}"
        )
        print(
            f"aggregated_impact_gini: \t{self.utility_model.data.aggregated_damage_gini}"
        )
        print(f"aggregated_utility: \t\t{self.utility_model.data.aggregated_utility}")

        print("Dataframe on spatially aggregated variables of interest")
        print(self.utility_model.data.df_main)

        print("Dataframe on spatially and temporally disaggregated population")
        print(self.utility_model.data.df_population)

        print("Dataframe on spatially disaggregated CPC")
        print(self.utility_model.data.df_cpc)

        print("Dataframe on CPC pre damage")
        print(self.utility_model.data.df_cpc_pre_damage)

        print("Dataframe on CPC post damage")
        print(self.utility_model.data.df_cpc_post_damage)


if __name__ == "__main__":

    model = PyRICE(
        model_specification=ModelSpec.STANDARD,
        damage_function=DamageFunction.NORDHAUS,
        welfare_function=WelfareFunction.UTILITARIAN,
    )

    # Default levers are defined by the original Nordhaus policy
    results = model()
    # [print(f'{k}: {v}') for k, v in outcomes.items()]
    # print(outcomes['Temperature overshoot 2105'])
    [print(f"{k}: {v}") for k, v in results.items() if "Temperature overshoot 2" in k]
