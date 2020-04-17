import pandas as pd
from src.state import NameState


class Disease:
    """ An Empty class for diseases. Use this class to run a location model with no disease.
    """

    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.cases = list()

    def step(self):
        self.collect_agents()

    def regenerate_agents(self, agent_ids):
        pass

    def save_output(self):
        pass

    def collect_agents(self, initiate: bool = False):
        columns = [NameState.LIFE.name, NameState.LOCATION.name]
        if initiate:
            daily_count_index_list = []
            for a_life in [0, 1]:
                for a_location in [
                    item.value for item in self.model.location.locations.enum
                ]:
                    daily_count_index_list.append((a_life, a_location))
            daily_counts = pd.DataFrame(daily_count_index_list, columns=columns)
            self.model.daily_counts = daily_counts.set_index(columns)

        df = pd.DataFrame(
            {
                NameState.LIFE.name: self.model.life.values,
                NameState.LOCATION.name: self.model.location.location.values,
            }
        )
        df = df.groupby(by=columns).size()
        self.model.daily_counts[self.model.time] = df
