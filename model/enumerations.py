"""
This module contains all custom-made enumerations.
"""

from enum import Enum


class ModelSpec(Enum):
    """
    Model Specifications
    """

    Validation_1 = 1
    Validation_2 = 2
    STANDARD = 3


class WelfareFunction(Enum):
    """
    Social Welfare Functions
    """

    UTILITARIAN = 0
    PRIORITARIAN = 1
    SUFFICIENTARIAN = 2
    EGALITARIAN = 3


class DamageFunction(Enum):
    """
    Damage Functions
    """

    NORDHAUS = 0
    NEWBOLD = 1
    WEITZMAN = 2


class ProblemFormulation(Enum):
    """
    Problem formulations. The values are tuples with (WelfareFunction, aggregation)
    """

    ALL_OBJECTIVES = WelfareFunction.UTILITARIAN, True, 0
    UTILITARIAN_COSTS = WelfareFunction.UTILITARIAN, False, 1
    UTILITARIAN_AGGREGATED = WelfareFunction.UTILITARIAN, True, 2
    UTILITARIAN_DISAGGREGATED = WelfareFunction.UTILITARIAN, False, 3
    EGALITARIAN_AGGREGATED = WelfareFunction.EGALITARIAN, True, 4
    EGALITARIAN_DISAGGREGATED = WelfareFunction.EGALITARIAN, False, 5
    SUFFICIENTARIAN_AGGREGATED = WelfareFunction.SUFFICIENTARIAN, True, 6
    SUFFICIENTARIAN_DISAGGREGATED = WelfareFunction.SUFFICIENTARIAN, False, 7
    PRIORITARIAN_AGGREGATED = WelfareFunction.PRIORITARIAN, True, 8
    PRIORITARIAN_DISAGGREGATED = WelfareFunction.PRIORITARIAN, False, 9
    ALL_KPIS = WelfareFunction.UTILITARIAN, True, 10

    @staticmethod
    def get_8_problem_formulations():
        """
        Get all problem formulations but ALL_OBJECTIVES and UTILITARIAN_COSTS
        """
        problem_formulations = [
            ProblemFormulation.UTILITARIAN_AGGREGATED,
            ProblemFormulation.UTILITARIAN_DISAGGREGATED,
            ProblemFormulation.SUFFICIENTARIAN_AGGREGATED,
            ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED,
            ProblemFormulation.PRIORITARIAN_AGGREGATED,
            ProblemFormulation.PRIORITARIAN_DISAGGREGATED,
            ProblemFormulation.EGALITARIAN_AGGREGATED,
            ProblemFormulation.EGALITARIAN_DISAGGREGATED
        ]
        return problem_formulations

    @staticmethod
    def get_util_and_suff_problem_formulations():
        """
        Get four problem formulations that are needed for my analysis.
        """

        pfs = [
            ProblemFormulation.UTILITARIAN_AGGREGATED,
            ProblemFormulation.UTILITARIAN_DISAGGREGATED,
            ProblemFormulation.SUFFICIENTARIAN_AGGREGATED,
            ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED
        ]

        return pfs

    @staticmethod
    def get_util_and_prio_problem_formulations():
        """
        Get four problem formulations that are needed for my analysis.
        """

        pfs = [
            ProblemFormulation.UTILITARIAN_AGGREGATED,
            ProblemFormulation.UTILITARIAN_DISAGGREGATED,
            ProblemFormulation.PRIORITARIAN_AGGREGATED,
            ProblemFormulation.PRIORITARIAN_DISAGGREGATED
        ]

        return pfs

    @staticmethod
    def get_pf_from_name(name):
        """
        Given a name String, return problem formulation.
        @param: name: String
        @param: pf: ProblemFormulation
        """

        pfs = {
            'UTILITARIAN_AGGREGATED': ProblemFormulation.UTILITARIAN_AGGREGATED,
            'UTILITARIAN_DISAGGREGATED': ProblemFormulation.UTILITARIAN_DISAGGREGATED,
            'SUFFICIENTARIAN_AGGREGATED': ProblemFormulation.SUFFICIENTARIAN_AGGREGATED,
            'SUFFICIENTARIAN_DISAGGREGATED': ProblemFormulation.SUFFICIENTARIAN_DISAGGREGATED,
            'PRIORITARIAN_AGGREGATED': ProblemFormulation.PRIORITARIAN_AGGREGATED,
            'PRIORITARIAN_DISAGGREGATED': ProblemFormulation.PRIORITARIAN_DISAGGREGATED,
            'EGALITARIAN_AGGREGATED': ProblemFormulation.EGALITARIAN_AGGREGATED,
            'EGALITARIAN_DISAGGREGATED': ProblemFormulation.EGALITARIAN_DISAGGREGATED,
        }

        pf = pfs[name]

        return pf
