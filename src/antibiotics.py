
import numpy as np

from enum import IntEnum
from src.misc_functions import create_cdf, random_selection
from src.state import SingleEvent, AntibioticsState, LifeState, NameState, AgeGroup


class Antibiotics(SingleEvent):
    def __init__(self, model, enum, transition_dict, key_types, antibiotic_risk):
        super().__init__(enum, transition_dict, key_types)

        self.model = model
        self.params = self.model.params.antibiotics
        self.initiate_values(
            count=len(self.model.population),
            value=AntibioticsState.OFF.value
        )
        self.initialize_probabilities()

        self.ends = np.zeros(len(self.model.population), dtype=np.int16)
        self.ends.fill(-1)
        self.risk_ratios = np.zeros(len(self.model.population), dtype=np.int16)
        self.risk_ratios.fill(antibiotic_risk['DEFAULT'])

        self.classes = IntEnum('Classes', {k: _ for _, k in enumerate(antibiotic_risk)})
        self.class_names = [item.name for item in self.classes]
        self.day = self.model.time + self.params['antibiotic_full_dose_length']

        self.type_dict = self.create_antibiotics_type_dictionary()
        self.off = AntibioticsState.OFF.value
        self.on = AntibioticsState.ON.value

    def initialize_probabilities(self):
        self.probabilities =\
            self.find_probabilities(list(zip(self.model.location.location.values, self.model.age_groups)))

    def step(self):
        """ Simulate Antibiotics
            1: Update antibiotic probabilities for any agent who has moved locations
            2: Simulate antibiotic prescription for living agents NOT ON antibiotics
            3: Living agents, ON antibiotics, completed the initial course
            4: For all events who antibiotics_ends today - take them off antibiotics
        """
        self.day = self.model.time + self.params['antibiotic_full_dose_length']
        # ----- 1:
        locations = self.model.location.location.values[self.model.location.moved_agents]
        ages = self.model.age_groups[self.model.location.moved_agents]
        ap = self.find_probabilities(list(zip(locations, ages)))
        self.probabilities[self.model.location.moved_agents] = ap

        # ----- 2: Living, Not On Antibiotics
        living = self.model.life.values == LifeState.ALIVE
        use_agents = self.model.unique_ids[(living) & (self.values == self.off)]
        probabilities = self.probabilities[use_agents]
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        if any(selected_agents):
            self.give_antibiotics(use_agents[selected_agents], previous_state=self.off)

        # ----- 3: Living, On Antibiotics, Off Initial Course
        use_agents = self.model.unique_ids[(living) & (self.values == self.on) & (self.ends < self.day)]
        probabilities = self.probabilities[use_agents]
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        if any(selected_agents):
            self.give_antibiotics(use_agents[selected_agents], previous_state=self.on)

        # ------ 4:
        for unique_id in self.model.unique_ids[self.ends == self.model.time]:
            self.model.record_state_change(
                unique_id=unique_id,
                name_state=NameState.ANTIBIOTICS,
                old=self.on,
                new=self.off
            )
            self.ends[unique_id] = self.off

    def give_antibiotics(self, agent_ids: np.array, previous_state: int = 0):
        """ Give antibiotics to an agent """
        days = np.round(
            self.model.rng.normal(
                self.params['antibiotic_administration_mean'],
                self.params['antibiotic_administration_sd'],
                len(agent_ids)
            )
        )
        end_days = self.params['antibiotic_full_dose_length'] + self.model.time + days
        self.ends[agent_ids] = end_days
        self.values[agent_ids] = self.on

        for unique_id in agent_ids:
            self.model.record_state_change(
                unique_id=unique_id,
                name_state=NameState.ANTIBIOTICS,
                old=previous_state,
                new=self.on
            )
            risk_ratio = self.assign_antibiotics_type(unique_id)
            if (risk_ratio > self.risk_ratios[unique_id]) or (previous_state == 0):
                self.risk_ratios[unique_id] = risk_ratio

    def assign_antibiotics_type(self, unique_id: int) -> float:
        """ Agents can receive different classes of antibiotics. This will randomly select a class for them. """
        dist = self.type_dict[self.model.location.location.values[unique_id]]
        a_class = random_selection(self.model.rng.rand(1), dist, self.class_names[1:])
        disease_type = self.model.params.base['disease_model']
        return self.params['relative_risk'][disease_type][a_class]

    def create_antibiotics_type_dictionary(self) -> dict:
        """ Create the antibiotics distributions lists for each location
        """
        td = dict()
        for location in self.model.location.locations.enum:
            category = self.model.location.locations.convert_int(location.value, 'category')
            dist = self.params['distributions'][category]
            td[location.value] = create_cdf(dist)
        return td


def create_antibiotics_dictionary(model) -> dict:
    d = dict()
    for location in model.location.locations.enum:
        for age in AgeGroup:
            category = model.location.locations.convert_int(location.value, 'category')
            f_id = model.location.locations.convert_int(location.value, 'ID')
            if category == 'COMMUNITY':
                d[(location.value, age.value)] = model.params.location['facilities']['COMMUNITY']['age'][str(age.value)]
            elif category == 'UNC':
                d[(location.value, age.value)] = model.params.location['facilities'][f_id]['antibiotics']
            elif category in ['LARGE', 'SMALL', 'LT', 'NH']:
                d[(location.value, age.value)] = model.params.location['facilities'][category]['antibiotics']
            else:
                raise ValueError("Category does not exist")
    return d
