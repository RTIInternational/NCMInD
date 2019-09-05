from enum import IntEnum


class CDIState(IntEnum):
    SUSCEPTIBLE = 0
    COLONIZED = 1
    CDI = 2
    DEAD = 3


class AntibioticsState(IntEnum):
    OFF = 0
    ON = 1


class NameState(IntEnum):
    LOCATION = 0
    LIFE = 1
    ANTIBIOTICS = 2
    CDI_RISK = 3


AntibioticsState.id = 'ANTIBIOTICS'
CDIState.id = 'CDI'
NameState.id = 'NAME'
