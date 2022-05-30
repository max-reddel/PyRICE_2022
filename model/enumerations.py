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

    @staticmethod
    def get_8_problem_formulations():
        """
        Get all problem formulations but ALL_OBJECTIVES and UTILITARIAN_COSTS
        """
        return list(ProblemFormulation)[2:]

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
