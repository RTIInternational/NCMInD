import numpy as np
import pandas as pd

from pathlib import Path
from src.state import AgeGroup, AntibioticsState, CREState, LifeState, Empty, NameState
from src.misc_functions import create_cdf, random_selection
from src.disease import Disease
from src.antibiotics import Antibiotics, create_antibiotics_dictionary


class CreDiseaseModule(Disease):
    def __init__(self, model, params):
        super().__init__(model, params)
        """ A class to house all of the CRE functions and objects
        """
        self.recent_cre_count = dict()
        self.most_recent_cre = dict()
        self.cre_count = dict()
        self.cre_probability_dict = dict()
        self.cre_carriage_dict = dict(), dict()

        # ----- Antibiotics
        self.antibiotics = Antibiotics(
            model=self.model,
            enum=AntibioticsState,
            transition_dict=create_antibiotics_dictionary(self.model),
            key_types=[self.model.location.locations.enum, AgeGroup],
            antibiotic_risk=self.model.params.antibiotics["relative_risk"]["cre"],
        )

        # ----- CRE
        self.cre = Empty("cre")
        self.cre.values = np.zeros(len(self.model.population), dtype=np.int16)
        self.cre.values.fill(CREState.SUSCEPTIBLE.value)
        self.cre.states = [item.name for item in CREState]

        self.initialize_cre()

    def initialize_cre(self):
        """ Do we start anyone with CRE?
        """
        # ----- Facility Initialization
        non_community = (
            self.model.location.location.values
            != self.model.location.locations.community
        )
        unique_ids = self.model.unique_ids[non_community]
        probabilities = np.zeros(len(unique_ids))
        probabilities.fill(self.params["cre"]["point_prevalence"]["non-ICU"])
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        for unique_id in unique_ids[selected_agents]:
            self.cre.values[unique_id] = CREState.CRE
        # ----- Community Initialization
        community = (
            self.model.location.location.values
            == self.model.location.locations.community
        )
        unique_ids = self.model.unique_ids[community]
        probabilities = np.zeros(len(unique_ids))
        probabilities.fill(self.params["cre"]["point_prevalence"]["non-ICU"])  # TODO
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        for unique_id in unique_ids[selected_agents]:
            self.cre.values[unique_id] = CREState.CRE

    def step(self):
        """ Run through a complete CRE step for all agents
            1: Simulate antibiotics use for all types of agents
            2: Simulate CRE Transition
            3: Collect the agents current states
        """
        self.antibiotics.step()
        self.simulate_cre_transitions()
        self.collect_agents()

    def simulate_cre_transitions(self):
        """ Simulate all agents CRE Transitions
            1: SUSCEPTIBLE TO CRE: Community & Facility
            2: CRE RECOVERY
        """
        living = self.model.life.values == LifeState.ALIVE
        susceptible_agents = (living) & (self.cre.values == CREState.SUSCEPTIBLE)
        cre_agents = (living) & (self.cre.values == CREState.CRE)

        # ----- 1a: Community
        community_id = self.model.location.locations.community
        in_community = self.model.unique_ids[
            (self.model.location.location.values == community_id) & susceptible_agents
        ]
        probabilities = np.zeros(len(in_community))
        probabilities.fill(self.params["cre"]["betas"]["base"])  # TODO
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        for unique_id in in_community[selected_agents]:
            self.cre_update(unique_id, CREState.CRE)

        # ----- 1b: Facility
        in_facility = self.model.unique_ids[
            (self.model.location.location.values != community_id) & susceptible_agents
        ]
        probabilities = self.find_cre_probability(in_facility)
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        for unique_id in in_facility[selected_agents]:
            self.cre_update(unique_id, CREState.CRE)

        # ----- 2: CRE Recovery
        probabilities = np.array([self.params["recovery"]] * len(cre_agents))
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        for unique_id in cre_agents[selected_agents]:
            # --- Create recovery options (Susceptible, Colonized, Death)
            to_death = self.params["death"]
            to_susceptible = 1 - to_death
            dist = [to_susceptible, to_death]
            options = [CREState.SUSCEPTIBLE, CREState.DEAD]
            end_state = random_selection(
                self.model.rng.rand(1), create_cdf(dist), options
            )
            self.cre_update(unique_id, end_state)

    def cre_update(self, unique_id: int, new_cre_state: CREState):
        """ Change the agent's cre status
            1: If the agent dies, record the state change and perform a life update
            2: Record the state change
        """
        current_cre_state = self.cre.values[unique_id]

        if new_cre_state == CREState.DEAD:
            self.model.record_state_change(
                unique_id=unique_id,
                name_state=NameState.CRE,
                old=current_cre_state,
                new=new_cre_state,
            )
            self.cre.values[unique_id] = new_cre_state
            self.model.life.life_update(unique_id)
            self.model.location.death_of_agent(unique_id)
            return

        if new_cre_state == CREState.CRE:
            self.most_recent_cre[unique_id] = self.model.time
            # --- Give Antibiotics
            self.antibiotics.give_antibiotics(
                [unique_id], previous_state=self.antibiotics.values[unique_id]
            )
            # --- Extend their LOS
            if (
                self.model.location.location.values[unique_id]
                in self.model.location.locations.all_hospitals
            ):
                self.model.location.current_los[unique_id] += 3
        # --- Record the CRE State Change
        self.model.record_state_change(
            unique_id, NameState.CRE, current_cre_state, new_cre_state
        )
        self.cre.values[unique_id] = new_cre_state

    def find_cre_probability(self, agent_ids: np.array) -> np.array:
        """ Find the probability of transition from Susceptible to CRE (non-community only)
        """
        probabilities = np.zeros(len(agent_ids))
        locations = self.model.location.location.values[agent_ids]
        for i in range(len(agent_ids)):
            if agent_ids[i] in self.model.location.icu_patients:
                probabilities[i] = self.cre_carriage["icus"][locations[i]]
            else:
                probabilities[i] = self.cre_carriage["facilites"][locations[i]]
        return probabilities

    def update_carriage_probability(self):
        """ Find the number of susceptible and cre carriage agents at each facility
        """
        base = self.params["cre"]["base_rates"]["COMMUNITY"]

        susceptible_by_facility = {
            facility_id.value: [] for facility_id in self.model.location.locations.enum
        }
        carriage_by_facility = {
            facility_id.value: [] for facility_id in self.model.location.locations.enum
        }

        non_community = (
            self.model.location.location.values
            != self.model.location.locations.community
        )
        susceptible_ids = self.model.unique_ids[
            non_community & (self.cre.values == CREState.SUSCEPTIBLE)
        ]
        carriage_ids = self.model.unique_ids[
            non_community & (self.cre.values == CREState.CRE)
        ]

        # ----- Susceptible by Hospital
        for unique_id in susceptible_ids:
            susceptible_by_facility[
                self.model.location.location.values[unique_id]
            ].append(unique_id)
        # ----- Carriage by Hospital
        for unique_id in carriage_ids:
            carriage_by_facility[self.model.location.location.values[unique_id]].append(
                unique_id
            )

        facilities = dict()
        icus = dict()
        for loc in self.model.location.locations.enum:
            if loc == self.model.location.locations.community:
                facilities[loc] = base
            else:
                m1 = len(susceptible_by_facility[loc]) * len(carriage_by_facility[loc])
                # --- Hospitals
                if loc in self.model.location.locations.all_hospitals:
                    facilities[loc] = max(
                        self.params["betas"]["non-ICU"] * m1,
                        self.params["base_rates"]["HOSPITAL"],
                    )
                    icus[loc] = max(
                        self.params["betas"]["ICU"] * m1,
                        self.params["base_rates"]["ICU"],
                    )
                # --- LTACHs
                if loc in self.model.location.locations.categories["LT"]["ints"]:
                    facilities[loc] = max(
                        self.params["betas"]["LTACH"] * m1,
                        self.params["base_rates"]["LTACH"],
                    )
                # --- NHs
                if loc in self.model.location.locations.categories["NH"]["ints"]:
                    facilities[loc] = max(
                        self.params["betas"]["NH"] * m1, self.params["base_rates"]["NH"]
                    )

        self.cre_carriage["facilites"] = facilities
        self.cre_carriage["icus"] = icus

    def regenerate_agents(self, agent_ids: np.array):
        """ When an agent dies, we regenerate them. This function will prepare a new agent with CRE values
        """
        ages = self.model.age_groups[agent_ids]
        locations = self.model.age_groups[agent_ids]

        # --- Assign no antibiotics
        antibiotics = np.zeros(len(ages))
        antibiotics.fill(AntibioticsState.OFF.value)
        self.antibiotics.values = np.append(
            self.antibiotics.values, antibiotics
        ).astype(np.int16)

        # --- Assign antibiotics end dates
        antibiotics_ends = np.zeros(len(ages))
        antibiotics_ends.fill(-1)
        self.antibiotics.ends = np.append(
            self.antibiotics.ends, antibiotics_ends
        ).astype(np.int16)

        # --- Assign susceptible CRE states
        cre_states = np.zeros(len(ages))
        cre_states.fill(CREState.SUSCEPTIBLE.value)
        self.cre.values = np.append(self.cre.values, cre_states).astype(np.int16)

        # --- Look up their Antibiotics change probability
        new_probabilities = self.antibiotics.find_probabilities(
            list(zip(locations, ages))
        )
        self.antibiotics.probabilities = np.append(
            self.antibiotics.probabilities, new_probabilities
        )

    def collect_agents(self, initiate: bool = False):
        """ Collect the daily information about CRE for all agents
        """
        if initiate:
            daily_count_index_list = []
            for an_anti in [0, 1]:
                for a_location in self.model.location.locations.enum:
                    for a_cre_state in CREState:
                        daily_count_index_list.append(
                            (an_anti, a_location.value, a_cre_state.value)
                        )
            daily_counts = pd.DataFrame(daily_count_index_list)
            daily_counts.columns = ["Antibiotics", "Location", "CRE"]
            self.model.daily_counts = daily_counts.set_index(
                ["Antibiotics", "Location", "CRE"]
            )
        # ----- Create arrays for antibiotics, life, location, and the cre_state
        antibiotics = self.antibiotics.values
        life = self.model.life.values
        locations = self.model.location.location.values
        cre_state = self.cre.values

        d_array = np.vstack((antibiotics, life, locations, cre_state)).T

        df = pd.DataFrame(d_array, columns=["Antibiotics", "Life", "Location", "CRE"])
        df = df[df.Life == LifeState.ALIVE.value]
        df = df.groupby(by=["Antibiotics", "Location", "CRE"]).size()
        self.model.daily_counts[self.model.time] = df

    def save_output(self):
        cases = pd.DataFrame(
            self.cases,
            columns=["Time", "Unique_ID", "Location", "Type", "Description", "County"],
        )
        cases.to_csv(Path(self.model.output_dir, "CRE_cases.csv"), index=False)
