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
    Problem formulations
    """

    UITILITARIAN_AGGREGATED = 0
    UTILITARIAN_DISAGGREGATED = 1
    UTILITARIAN_COSTS = 2
    EGALITARIAN_AGGREGATED = 3
    EGALITARIAN_DISAGGREGATED = 4
    SUFFICIENTARIAN_AGGREGATED = 5
    SUFFICIENTARIAN_DISAGGREGATED = 6
    PRIORITARIAN_AGGREGATED = 7
    PRIORITARIAN_DISAGGREGATED = 8
