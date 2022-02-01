"""
This module contains all custom-made enumerations.
"""

from enum import Enum


class ModelSpec(Enum):
    """
    Model Specification
    """

    Validation_1 = 1
    Validation_2 = 2
    STANDARD = 3

    def __eq__(self, o: object) -> bool:
        if self.value is o.value:
            return True
        else:
            return False


class WelfareFunction(Enum):
    """
    Social Welfare Functions
    """

    UTILITARIAN = 0
    PRIORITARIAN = 1
    SUFFICIENTARIAN = 2
    EGALITARIAN = 3

    def __eq__(self, o: object) -> bool:
        if self.value is o.value:
            return True
        else:
            return False


class DamageFunction(Enum):
    """
    Damage Functions
    """

    NORDHAUS = 0
    NEWBOLD = 1
    WEITZMAN = 2

    def __eq__(self, o: object) -> bool:
        if self.value is o.value:
            return True
        else:
            return False
