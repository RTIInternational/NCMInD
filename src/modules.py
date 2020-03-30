
from src.location import Location
from src.north_carolina import NcLocationModule

from src.disease import Disease
from src.cdi import CdiDiseaseModule
from src.cre import CreDiseaseModule


location_models = {
    "none": Location,
    "nc": NcLocationModule
}

disease_models = {
    "none": Disease,
    "cre": CreDiseaseModule,
    "cdi": CdiDiseaseModule
}
