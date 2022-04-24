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
    ALL_OBJECTIVES = WelfareFunction.UTILITARIAN, True
    UTILITARIAN_COSTS = WelfareFunction.UTILITARIAN, False
    UITILITARIAN_AGGREGATED = WelfareFunction.UTILITARIAN, True
    UTILITARIAN_DISAGGREGATED = WelfareFunction.UTILITARIAN, False
    EGALITARIAN_AGGREGATED = WelfareFunction.EGALITARIAN, True
    EGALITARIAN_DISAGGREGATED = WelfareFunction.EGALITARIAN, False
    SUFFICIENTARIAN_AGGREGATED = WelfareFunction.SUFFICIENTARIAN, True
    SUFFICIENTARIAN_DISAGGREGATED = WelfareFunction.SUFFICIENTARIAN, False
    PRIORITARIAN_AGGREGATED = WelfareFunction.PRIORITARIAN, True
    PRIORITARIAN_DISAGGREGATED = WelfareFunction.PRIORITARIAN, False
