import numpy as np
import pandas as pd

from math import exp
from pathlib import Path
from src.state import (
    AgeGroup,
    AntibioticsState,
    CDIState,
    LifeState,
    Empty,
    NameState,
    ConcurrentConditions,
)
from src.misc_functions import create_cdf, random_selection
from src.disease import Disease
from src.antibiotics import Antibiotics, create_antibiotics_dictionary


class CdiDiseaseModule(Disease):
    def __init__(self, model, params):
        super().__init__(model, params)
        """ A class to house all of the CDI functions and objects
        """
        self.recurrent_cdi_ends = dict()
        self.recent_cdi_count = dict()
        self.most_recent_cdi = dict()
        self.cdi_count = dict()
        self.cdi_probability_dict = dict()
        self.cdi_dict = self.create_cdi_dictionary()

        # ----- Antibiotics
        self.antibiotics = Antibiotics(
            model=self.model,
            enum=AntibioticsState,
            transition_dict=create_antibiotics_dictionary(self.model),
            key_types=[self.model.location.locations.enum, AgeGroup],
            antibiotic_risk=self.model.params.antibiotics["relative_risk"]["cdi"],
        )

        # ----- CDI
        self.cdi = Empty("cdi")
        self.cdi.values = np.zeros(len(self.model.population), dtype=np.int16)
        self.cdi.values.fill(CDIState.SUSCEPTIBLE.value)
        self.cdi.states = [item.name for item in CDIState]

        self.update_colonization_probability()
        self.initialize_colonization()

    def step(self):
        """ Run through a complete CDI step for all agents
            1: Simulate antibiotics use for all types of agents
            2: Update the colonization probability for each location
            3: Simulate CDI Transition
            4: Force a small amount of Community Onset CDI
            5: Collect the agents current states
        """
        self.antibiotics.step()
        self.update_colonization_probability()
        self.simulate_cdi_transitions()
        self.collect_agents()

    def create_cdi_dictionary(self):
        """ Create a dictionary of colonized to CDI transition rates
        """
        antibiotics_risk_ratios = list(
            self.model.params.antibiotics["relative_risk"]["cdi"].values()
        )

        d = dict()
        for k1 in self.model.location.locations.categories_list:  # Location Category
            for k2 in [
                item.value for item in ConcurrentConditions
            ]:  # Concurrent Conditions
                for k3 in AntibioticsState:  # Antibiotics State
                    for k4 in [0, 1, 2, 3]:  # Recent CDI
                        for k5 in AgeGroup:  # Age
                            for k6 in antibiotics_risk_ratios:  # Risk Ratio
                                for k7 in [
                                    "No",
                                    "Yes",
                                ]:  # First 3 days of hospital visit
                                    d[k1, k2, k3.value, k4, k5.value, k6, k7] = {}
        for key in d.keys():
            (
                category,
                concurrent_conditions,
                antibiotics_state,
                recent_cdi_count,
                age,
                risk_ratio,
                first_three,
            ) = key
            if recent_cdi_count > 0:
                d[key] = self.params["cdi"]["recurrence"]["base_rate"]
            else:
                rr = 1 * self.params["cdi"]["relative_risk"]["age"][str(age)]
                if antibiotics_state == 1:
                    rr *= risk_ratio
                rr *= self.params["cdi"]["relative_risk"]["concurrent_conditions"][
                    str(concurrent_conditions)
                ]
                if (first_three == "Yes") & (category in ["UNC", "LARGE", "SMALL"]):
                    rr *= self.params["cdi"]["co_cdi_multipliers"][category]
                d[key] = (
                    self.params["cdi"]["base_rate"][category]
                    * rr
                    * self.params["cdi"]["tuning"][category]
                )
                d[key] = min(0.5, d[key])
        return d

    def update_colonization_probability(self):
        """ Loop through the locations and find force of colonization.
        """
        ghh = 1  # ghh: overall hospital hygiene
        bs = self.params["cdi"]["base_rate"]["LARGE"]  # B_s = base CDI transition rate
        ba = self.params["colonization"]["base_rate"][
            "LARGE"
        ]  # B_a = base asymp. colonization trans/ rate
        pi = self.params["cdi"]["contact_precautions"][
            "identified"
        ]  # pi: prob agent w/ CDI is identified
        eps = self.params["cdi"]["contact_precautions"][
            "effectiveness"
        ]  # epsilon: effectiveness of cp
        d = dict()

        v = {}
        unique_ids = self.model.unique_ids[
            self.model.location.location.values
            != self.model.location.locations.community
        ]
        for unique_id in unique_ids:
            v.setdefault(self.model.location.location.values[unique_id], []).append(
                unique_id
            )
        base = (
            self.params["colonization"]["base_rate"]["COMMUNITY"]
            * self.params["colonization"]["tuning"]["COMMUNITY"]
        )

        for location in self.model.location.locations.enum:
            agent_ids = v.get(location, [])
            if location.value == 0:
                d[location] = base
            else:
                category = self.model.location.locations.int_to_category[location.value]
                if len(agent_ids) > 0:
                    cdi_status = self.cdi.values[agent_ids]
                    # CDI_st = # of CDI cases in hospital / # of STACH patients
                    cdi_st = np.count_nonzero(cdi_status == CDIState.CDI.value) / len(
                        agent_ids
                    )
                    # C_st = # of colonized cases in hospital / # of STACH patients
                    c_st = np.count_nonzero(
                        cdi_status == CDIState.COLONIZED.value
                    ) / len(agent_ids)
                    lambda_st = ghh * (
                        bs * (1 - pi) * cdi_st + ba * c_st
                    ) + pi * bs * cdi_st * (1 - eps)
                    d[location.value] = (
                        base * self.params["colonization"]["tuning"][category]
                    )
                    if lambda_st != 0:
                        d[location.value] = (
                            lambda_st * self.params["colonization"]["tuning"][category]
                        )
                else:
                    d[location.value] = (
                        base * self.params["colonization"]["tuning"][category]
                    )
        self.colonization_dict = d

    def simulate_cdi_transitions(self):
        """ Simulate all agents CDI Transitions
            Find which patients are susceptible, colonized, or currently have CDI

            1: Susecptible to Colonized (a: those in facilities, b: those in community)
            2: Colonization Recovery
            3: Colonized to CDI
            4: CDI Recovery
        """
        living = self.model.life.values == LifeState.ALIVE
        susceptible_agents = (living) & (self.cdi.values == CDIState.SUSCEPTIBLE)
        colonized_agents = self.model.unique_ids[
            (living) & (self.cdi.values == CDIState.COLONIZED)
        ]
        cdi_agents = self.model.unique_ids[(living) & (self.cdi.values == CDIState.CDI)]

        # ----- 1a:
        community = self.model.location.locations.community
        in_facility = self.model.unique_ids[
            (self.model.location.location.values != community) & susceptible_agents
        ]
        probabilities = self.find_colonization_probability(in_facility)
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        for unique_id in in_facility[selected_agents]:
            self.cdi_update(unique_id, CDIState.COLONIZED)

        # ----- 1b:
        in_community = self.model.unique_ids[
            (self.model.location.location.values == community) & susceptible_agents
        ]
        probabilities = np.zeros(len(in_community))
        probabilities.fill(self.colonization_dict[community])
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        for unique_id in in_community[selected_agents]:
            self.cdi_update(unique_id, CDIState.COLONIZED)

        # ----- 2:
        probabilities = np.array(
            [self.params["colonization"]["clearance"]] * len(colonized_agents)
        )
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        for unique_id in colonized_agents[selected_agents]:
            self.cdi_update(unique_id, CDIState.SUSCEPTIBLE)

        # ----- 3:
        probabilities = self.find_cdi_probability(colonized_agents)
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        for unique_id in colonized_agents[selected_agents]:
            self.cdi_update(unique_id, CDIState.CDI)

        # ----- 4:
        probabilities = np.array([self.params["cdi"]["recovery"]] * len(cdi_agents))
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        for unique_id in cdi_agents[selected_agents]:
            # --- Create recovery options (Susceptible, Colonized, Death)
            to_death = self.params["cdi"]["death"]["age"][
                str(int(self.model.age_groups[unique_id]))
            ]
            to_col = self.params["cdi"]["recurrence"]["probability_with_recent_CDI"][
                str(self.recent_cdi_count.get(unique_id, 0))
            ]
            dist = [1 - to_death - to_col, to_col, to_death]
            options = [CDIState.SUSCEPTIBLE, CDIState.COLONIZED, CDIState.DEAD]
            end_state = random_selection(
                self.model.rng.rand(1), create_cdf(dist), options
            )
            self.cdi_update(unique_id, end_state)

        self.update_cdi_variables()

    def update_cdi_variables(self):
        """ Any living agents who has not had CDI in X days, needs their recent cdi count reset to 0 """
        for unique_id in [
            k for k, v in self.recurrent_cdi_ends.items() if v == self.model.time
        ]:
            del self.recurrent_cdi_ends[unique_id]
            del self.recent_cdi_count[unique_id]

    def initialize_colonization(self):
        """ All agents start as SUSCEPTIBLE. Initialize a small percentage of agents to start with COLONIZED.
            Initialization based on desired prevalence of COLONIZED at each facility
        """
        rates = []
        for location in self.model.location.location.values:
            category = self.model.location.locations.int_to_category[location]
            rates.append(self.params["colonization"]["initialization"][category])
        selected_agents = np.array(rates) > self.model.rng.rand(len(rates))
        self.cdi.values[
            self.model.unique_ids[selected_agents]
        ] = CDIState.COLONIZED.value

    def cdi_update(self, unique_id: int, new_cdi_state: CDIState):
        """ Change the agent's cdi status
            1: If the agent dies, record the state change and perform a life update
            2: If the agent gets CDI we:
                2a: Check the Antibiotic risk decay equation (not everyone will actuall get CDI)
                2b: Check Recent CDI Counts
                2c: Find CDI NHSN and Association Details
            3: Record the state change
        """
        current_cdi_state = self.cdi.values[unique_id]

        if new_cdi_state == CDIState.DEAD:
            self.model.life.life_update(unique_id)

        if new_cdi_state == CDIState.CDI:
            # ----- If a recent CDI, then we must allow a new CDI to occur (so skip this)
            if unique_id not in self.recent_cdi_count:
                # ----- Use antibiotic decay equation to see if CDI should have beeen given in the first palce
                if self.antibiotics.values[unique_id] == AntibioticsState.ON:
                    if self.antibiotics.ends[unique_id] - self.model.time < 60:
                        days_into_antibiotics = (
                            self.model.time - self.antibiotics.ends[unique_id] + 90
                        )
                        new_chance = 2.4937 * exp(-0.03 * days_into_antibiotics)
                        # --- If this occurs, you should not have received CDI, and thus we stop your update
                        if self.model.rng.rand() > new_chance:
                            return
            # --- Update all of your recent CDI variables
            days_since_last_cdi = 0
            if unique_id in self.most_recent_cdi:
                days_since_last_cdi = self.model.time - self.most_recent_cdi[unique_id]
            if (days_since_last_cdi > 13) or (days_since_last_cdi == 0):
                self.cdi_count[unique_id] = self.cdi_count.get(unique_id, 0) + 1
                if self.recent_cdi_count.get(unique_id, 0) < 3:
                    self.recent_cdi_count[unique_id] = (
                        self.recent_cdi_count.get(unique_id, 0) + 1
                    )
            # --- CDI Details
            self.associate_cdi(int(unique_id), days_since_last_cdi)
            # --- Set the agents CDI variables
            self.most_recent_cdi[unique_id] = self.model.time
            self.recurrent_cdi_ends[unique_id] = (
                self.params["cdi"]["maximum_length_of_recurring_CDI"] + self.model.time
            )
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
        # --- Record the CDI State Change
        self.model.record_state_change(
            unique_id=unique_id,
            name_state=NameState.CDI,
            old=current_cdi_state,
            new=new_cdi_state,
        )
        self.cdi.values[unique_id] = new_cdi_state

    def associate_cdi(self, unique_id: int, days_since_last_cdi: int):
        """ Each CDI case must be given a specific label.
            CDI Type:
                - 0 and >56 is a new case.
                - 1-13 days is consider Duplicate
                - 14-56 days is considered Recurrent
            NHSH Definition:
                - Hospital (healthcare facility)-onset CDI (HO-CDI): A patient is considered to have hospital-onset CDI
                if the patient has CDI at least 3 days after hospital admission (specifically, on or after day 4) and
                had no prior positive in the preceding 14 days.
                - Community-onset CDI (CO-CDI):  A patient is considered to have (inpatient) community-onset CDI if the
                patient has a positive C. difficile toxin assay fewer than 3 days after hospital admission
                (specifically, days 1, 2, or 3 of admission) and had no prior positive in the preceding 14 days.
            Assiocation Definition:
                - Community-associated CDI (CA-CDI): A CDI case is classified as community-associated if the specimen
                was collected on an outpatient basis or within 3 days after admission (i.e., days 1, 2, or 3 of
                admission) to a healthcare facility in a person with no documented overnight stay in a healthcare
                facility in the preceding 12 weeks.
                - Healthcare-associated CDI (HA-CDI): All CDI cases that do not meet the aforementioned criteria are classified as healthcare-associated.
        """

        cdi_type = "Duplicate"
        if (days_since_last_cdi > 56) or (days_since_last_cdi == 0):
            cdi_type = "Incident"
        elif days_since_last_cdi > 13:
            cdi_type = "Recurrent"

        # --------------------------------------------------------------------------------------------------------------
        location = self.model.location.location.values[unique_id]
        community = self.model.location.locations.community
        nhsn = "N/A"
        association = "N/A"
        self.current = unique_id
        if cdi_type != "Duplicate":
            events = self.model.cur.execute(
                "SELECT * FROM event_tracker WHERE Unique_ID=?", (unique_id,)
            ).fetchall()
            events = [
                item
                for item in events
                if item[self.model.event_columns.index("State")] == NameState.LOCATION
            ]

            # ----- NHSN Definitions:
            if location == community:
                nhsn = "Community"
            else:
                # 2 Options:
                #   - No events and not in the community (i.e. you've also been at a facility
                #   - If the last movement event happened more than 3 days ago: HO CDI
                if len(events) == 0:
                    nhsn = "HO CDI"
                elif (
                    events[-1][self.model.event_columns.index("Time")] + 3
                    < self.model.time
                ):
                    nhsn = "HO CDI"
                else:
                    nhsn = "CO CDI"

            # ----- Assiociation: Either-or - it must be one CA or HA
            if len(events) > 0:
                end = self.model.time - 3
                start = self.model.time - 12 * 7
                movement_days = [
                    item[self.model.event_columns.index("Time")] for item in events
                ]
                # --- If currently at a facility and have been there for at least 3 days:
                test1 = location != self.model.location.locations.community
                if test1 & (
                    events[-1][self.model.event_columns.index("Time")] + 3
                    < self.model.time
                ):
                    association = "HA-CDI"
                # --- Or if any movement within the range:
                elif (
                    len(
                        [
                            item
                            for item in movement_days
                            if (item < end) and (item > start)
                        ]
                    )
                    > 0
                ):
                    association = "HA-CDI"
                else:
                    association = "CA-CDI"
            # --- If you haven't moved at all:
            elif location != self.model.location.locations.community:
                association = "HA-CDI"
            else:
                association = "CA-CDI"

        county = self.model.county_codes[unique_id]
        self.cases.append(
            (self.model.time, unique_id, location, cdi_type, nhsn, association, county)
        )

    def regenerate_agents(self, agent_ids: np.array):
        """ When an agent dies, we regenerate them. This function will prepare a new agent with CDI values """
        ages = self.model.age_groups[agent_ids]
        locations = self.model.age_groups[agent_ids]

        # --- Assign a default risk ratio
        anti_rr = np.zeros(len(ages))
        anti_rr.fill(self.model.params.antibiotics["relative_risk"]["cdi"]["DEFAULT"])
        self.antibiotics.risk_ratios = np.append(self.antibiotics.risk_ratios, anti_rr)

        # --- Assign no antibiotics
        antibiotics = np.zeros(len(ages))
        antibiotics.fill(AntibioticsState.OFF.value)
        self.antibiotics.values = np.append(self.antibiotics.values, anti_rr).astype(
            np.int16
        )

        # --- Assign antibiotics end dates
        antibiotics_ends = np.zeros(len(ages))
        antibiotics_ends.fill(-1)
        self.antibiotics.ends = np.append(
            self.antibiotics.ends, antibiotics_ends
        ).astype(np.int16)

        # --- Assign susceptible CDI states
        cdi_states = np.zeros(len(ages))
        cdi_states.fill(CDIState.SUSCEPTIBLE.value)
        self.cdi.values = np.append(self.cdi.values, cdi_states).astype(np.int16)

        # --- Look up their Antibiotics change probability
        new_probabilities = self.antibiotics.find_probabilities(
            list(zip(locations, ages))
        )
        self.antibiotics.probabilities = np.append(
            self.antibiotics.probabilities, new_probabilities
        )

    def find_cdi_probability(self, agent_ids: np.array) -> np.array:
        """ Find the probability of transition from Colonized to CDI
        """
        # ----- Only keep agent_ids that are not in the dictionary already
        use_ids = np.setdiff1d(
            agent_ids, list(self.cdi_probability_dict.keys())
        ).tolist()
        # --- Add any agents not in the community
        non_community = self.model.unique_ids[
            self.model.location.location.values
            != self.model.location.locations.community
        ]
        use_ids = use_ids + np.intersect1d(non_community, agent_ids).tolist()
        for unique_id in use_ids:
            # ------ Update Dictionary with any new colonized person or anyone who has changed locations
            k1 = self.model.location.locations.int_to_category[
                self.model.location.location.values[unique_id]
            ]
            k2 = self.model.concurrent_conditions[unique_id]
            k3 = self.antibiotics.values[unique_id]
            k4 = self.recent_cdi_count.get(unique_id, 0)
            k5 = self.model.age_groups[unique_id]
            k6 = self.antibiotics.risk_ratios[unique_id]
            k7 = (
                "Yes"
                if self.model.location.last_movement_day.get(unique_id, -3)
                - self.model.time
                > -3
                else "No"
            )
            self.cdi_probability_dict[unique_id] = self.cdi_dict[
                (k1, k2, k3, k4, k5, k6, k7)
            ]
        # ----- Remove anyone that is no longer colonized
        for unique_id in np.setdiff1d(
            list(self.cdi_probability_dict.keys()), agent_ids
        ).tolist():
            del self.cdi_probability_dict[unique_id]
        # ----- Fill probabilities
        return np.array(
            [self.cdi_probability_dict[unique_id] for unique_id in agent_ids]
        )

    def find_colonization_probability(self, agent_ids: np.array) -> np.array:
        probabilities = np.zeros(len(agent_ids))
        locations = self.model.location.location.values[agent_ids]
        for i in range(len(locations)):
            probabilities[i] = self.colonization_dict[locations[i]]
        return probabilities

    def collect_agents(self, initiate: bool = False):
        """ Collect the daily information about CDI for all agents
        """
        columns = [
            NameState.ANTIBIOTICS.name,
            NameState.LOCATION.name,
            NameState.CDI.name,
        ]
        if initiate:
            daily_count_index_list = []
            for an_anti in [0, 1]:
                for a_location in self.model.location.locations.enum:
                    for a_cdi_state in CDIState:
                        daily_count_index_list.append(
                            (an_anti, a_location.value, a_cdi_state.value)
                        )
            daily_counts = pd.DataFrame(daily_count_index_list)
            daily_counts.columns = columns
            self.model.daily_counts = daily_counts.set_index(columns)
        # ----- Create arrays for antibiotics, life, location, and the cdi_state
        living = self.model.life.values == LifeState.ALIVE
        antibiotics = self.antibiotics.values[living]
        locations = self.model.location.location.values[living]
        cdi_state = self.cdi.values[living]

        d_array = np.vstack((antibiotics, locations, cdi_state)).T

        # This is essentially a groupby. We do it this way for speed purposes.
        # lidx = np.ravel_multi_index(d_array.T, d_array.max(0) + 1)
        # unq, unqtags, counts = np.unique(lidx, return_inverse=True, return_counts=True)
        # v1 = np.zeros(self.model.daily_counts.shape[0])
        # v1[unq] = counts
        # self.model.daily_counts[self.model.time] = v1

        df = pd.DataFrame(d_array, columns=columns)
        df = df.groupby(by=columns).size()
        self.model.daily_counts[self.model.time] = df

    def save_output(self):
        cases = pd.DataFrame(
            self.cases,
            columns=[
                "Time",
                "Unique_ID",
                "Location",
                "Type",
                "NHSN",
                "Association",
                "County",
            ],
        )
        cases.to_csv(Path(self.model.output_dir, "CDI_cases.csv"), index=False)
