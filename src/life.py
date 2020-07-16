import numpy as np
from src.state import NameState, LifeState, SingleEvent


class Life(SingleEvent):
    def __init__(self, model, params, enum, transition_dict, key_types):
        super().__init__(enum, transition_dict, key_types)

        self.model = model
        self.params = params
        self.agents_to_recreate = []
        self.initiate_values(
            count=len(self.model.population), value=LifeState.ALIVE.value,
        )

    def step(self):
        """ See if anyone dies of causes not related to diseases
        """
        use_agents = np.where(self.values == LifeState.ALIVE)[0]
        probabilities = self.probabilities[use_agents]
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        unique_ids = use_agents[selected_agents]
        for unique_id in unique_ids:
            self.life_update(unique_id)

    def life_update(self, unique_id: int):
        """ Perform a life update. Add agent to the list of agents to recreate.
        """
        self.model.record_state_change(
            unique_id=unique_id,
            name_state=NameState.LIFE.value,
            old=LifeState.ALIVE.value,
            new=LifeState.DEAD.value,
        )
        self.values[unique_id] = LifeState.DEAD.value
        self.agents_to_recreate.append(unique_id)
