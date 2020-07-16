import numpy as np
from enum import IntEnum
from src.misc_functions import create_cdf


class NameState(IntEnum):
    LOCATION = 0
    LIFE = 1
    ANTIBIOTICS = 2
    CDI = 3
    CRE = 4


class LifeState(IntEnum):
    ALIVE = 0
    DEAD = 1


class AntibioticsState(IntEnum):
    OFF = 0
    ON = 1


class CDIState(IntEnum):
    SUSCEPTIBLE = 0
    COLONIZED = 1
    CDI = 2
    DEAD = 3


class CREState(IntEnum):
    SUSCEPTIBLE = 0
    CRE = 1
    DEAD = 2


class AgeGroup(IntEnum):
    AGE0 = 0
    AGE1 = 1
    AGE2 = 2


class ConcurrentConditions(IntEnum):
    NO = 0
    YES = 1


NameState.id = "NAME"
LifeState.id = "LIFE"
AntibioticsState.id = "ANTIBIOTICS"
CDIState.id = "CDI"
CREState.id = "CRE"
AgeGroup.id = "AGE"


class EventState:
    def __init__(self, enum: IntEnum, transition_dict: dict, key_types: list):
        self.enum = enum
        self.integers = [item.value for item in enum]
        self.names = [item.name for item in enum]
        self.transition_dict = transition_dict
        self.key_types = key_types
        self.values = None

    def initiate_values(self, count: int, value: int, dtype: type = int) -> np.array:
        self.values = np.zeros(count, dtype=dtype)
        self.values.fill(value)


class SingleEvent(EventState):
    """ A state class for events that only have one option (i.e. a probability of death)
    """

    def __init__(self, enum, transition_dict, key_types):
        super().__init__(enum, transition_dict, key_types)

        self.probabilities = None

    def find_probabilities(self, keys: list):
        """ Given a set of keys, create a list of probabilities
        """
        probabilities = np.zeros(len(keys))
        for i in range(len(keys)):
            probabilities[i] = self.transition_dict[keys[i]]
        return probabilities


class MultipleEvent(EventState):
    """ A state class for States that can transition multiples states (i.e. an agent being sick)
    """

    def __init__(self, enum, transition_dict, key_types):
        super().__init__(enum, transition_dict, key_types)

        self.cdf_dict = dict()
        self.create_cdf_dict()
        self.check_inputs()

    def check_inputs(self):
        for k, v in self.transition_dict.items():
            if not np.isclose(1, sum(v), 0.000001):
                raise ValueError(
                    "Key {} for transition_dict does not sum to 1.".format(k)
                )

    def create_cdf_dict(self):
        for k, v in self.transition_dict.items():
            self.cdf_dict[k] = create_cdf(v)

    def find_cdf_probabilities(self, keys: list):
        """ Given a set of keys, create a list of cdf probabilities
        """
        probabilities = np.zeros(len(keys))
        for i in range(len(keys)):
            probabilities[i] = self.transition_dict[keys[i]]
        return probabilities


class Empty:
    """ An empty state to house extra arrays or dictionaries
    """

    def __init__(self, data_type):
        self.data_type = data_type
