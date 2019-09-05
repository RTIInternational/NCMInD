
from pathlib import Path
import json
import pandas as pd


class Parameters:

    def __init__(self, params_file):
        self.base = dict()
        self.tuning = dict()
        self.length_of_stay = dict()
        self.death_multipliers = dict()
        self.length_of_stay = dict()
        self.cdi = dict()
        self.update_from_file(params_file)

    def update_from_file(self, params_file):
        params_file = Path(params_file)

        params = {}
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
        else:
            print('WARNING: Parameter file does not exist.')

        for key in params:
            setattr(self, key, params[key])
