import pandas as pd
import numpy as np
from enum import IntEnum
from src.state import NameState, LifeState


class Location:
    """ An Empty class for a location module.
    """

    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.moved_agents = list()
        self.current_los = dict()
        self.readmission_date = dict()
        self.readmission_location = dict()
        self.leave_facility_day = dict()
        self.last_movement_day = dict()
        self.locations = Locations()

    def step(self):
        pass

    def community_movement(self):
        pass

    def facility_movement(self):
        pass

    def location_update(self):
        pass

    def assign_los(self, unique_id: int, new_location: int):
        pass

    def regenerate_agents(self, agent_ids):
        new_locations = [self.locations.community] * len(agent_ids)

        self.model.life.probabilities = np.append(
            self.model.life.probabilities,
            self.model.life.find_probabilities(
                list(zip(self.model.age_groups[agent_ids], new_locations))
            ),
        )
        self.model.life.values = np.append(
            self.model.life.values, [LifeState.ALIVE.value] * len(agent_ids)
        ).astype(np.int16)

        keys = list(
            zip(self.model.county_codes[agent_ids], self.model.age_groups[agent_ids])
        )
        self.location.probabilities = np.append(
            self.location.probabilities, self.location.find_probabilities(keys)
        )
        self.location.values = np.append(self.location.values, new_locations).astype(
            np.int16
        )
        self.location.previous = np.append(
            self.location.previous, new_locations
        ).astype(np.int16)

    def death_of_agent(self, unique_id: int):
        """ Agents who die can no longer move locations
        """
        # ----- If agent is scheduled to move locations, stop them, as they are now dead.
        if unique_id in self.moved_agents:
            self.moved_agents.remove(unique_id)

        # ----- If agent is not at home, send them home
        if self.location.values[unique_id] != self.locations.community:
            self.current_los[unique_id] = self.model.time - self.last_movement_day.get(
                unique_id, 0
            )
            # ----- Go home
            self.model.record_state_change(
                unique_id=unique_id,
                name_state=NameState.LOCATION.value,
                old=self.location.values[unique_id],
                new=self.locations.community,
            )
            # --- Remove the agent from dictionaries
            self.location.values[unique_id] = self.locations.community
            del self.current_los[unique_id]
            del self.leave_facility_day[unique_id]

    def readmission_movement(self):
        # ----- Remove any old readmission values: We keep these until the next day so that disease states can use them
        for key in list(self.readmission_date.keys()):
            if self.readmission_date[key] < self.model.time:
                del self.readmission_date[key]
                del self.readmission_location[key]
        # ----- Find readmissions
        move_ids = [
            key
            for key, value in self.readmission_date.items()
            if value == self.model.time
        ]
        previous_location = self.locations.community
        for unique_id in move_ids:
            new_location = self.readmission_location[unique_id]
            self.model.record_state_change(
                unique_id=unique_id,
                name_state=NameState.LOCATION.value,
                old=previous_location,
                new=new_location,
            )
            self.assign_los(unique_id=unique_id, new_location=new_location)
            self.location.values[unique_id] = new_location

            # --- Finalize Movement
            self.moved_agents.append(unique_id)
            self.location.previous[unique_id] = previous_location
            self.update_death_probabilities(unique_id)

    def update_death_probabilities(self, unique_id: int):
        """ As an agent changes locations, they need their death probability update
        """
        age = self.model.age_groups[unique_id]
        location = self.location.values[unique_id]
        location = self.locations.convert_int(location, "int_category")
        self.model.life.probabilities[unique_id] = self.model.life.transition_dict[
            age, location
        ]


class Locations:
    """ An empty class for Locations.
    """

    def __init__(self):
        self.facilities = dict()
        self.categories = dict()
        self.number_of_items = 0
        self.enum = IntEnum
        self.category_enum = IntEnum
        self.categories_list = list()
        self.community = int()

    def add_value(self, row: pd.Series, int_category: int):
        if row.ID in self.facilities:
            raise ValueError("Cannot use duplicate ID {}".format(row.ID))
        d = {
            "int": self.number_of_items,
            "int_category": int_category,
            "ID": row.ID,
            "name": row.Name,
            "category": row.Category,
            "beds": row.Beds,
        }
        self.facilities[d["ID"]] = d
        self.number_of_items += 1

    def convert_int(self, an_int: int, description: str):
        return self.facilities[self.enum(an_int).name][description]

    def convert_name(self, a_name: str, description: str):
        return self.facilities[a_name][description]
