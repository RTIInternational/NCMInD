import numpy as np
import pandas as pd
from enum import IntEnum
from pathlib import Path

from src.state import NameState
from src.jit_functions import update_community_probability
from src.misc_functions import normalize_and_create_cdf, create_cdf, random_selection
from src.state import SingleEvent, AgeGroup, ConcurrentConditions
from src.location import Location, Locations


class NcLocationModule(Location):
    def __init__(self, model, params):
        super().__init__(model, params)
        self.icu_patients = set()
        self.locations = NcLocations(self.model.experiment_dir)

        # ----- Facility Movement
        transition_directory = self.params["transition_directory"]
        self.facility_transitions = location_dict(
            self, pd.read_csv(Path(transition_directory, "location_transitions.csv"))
        )

        # ----- Community Movement
        ct = pd.read_csv(Path(transition_directory, "community_transitions.csv"))
        ct["Probability"] = (
            ct["Probability"]
            * self.params["tuning"]["community_probability_multiplier"]
        )
        community_dict = dict()
        for _, row in ct.iterrows():
            community_dict[int(row.County_Code), int(row.Age)] = row.Probability

        # ----- Facilities by Distance
        ft = pd.read_csv("NCMIND/data/input/logrecnos.csv")
        self.logrec_movement = ft.values
        self.logrec_columns = ft.columns

        # ----- Transitions
        self.transitions = dict()
        self.transitions["UNC"] = pd.read_csv(
            Path(transition_directory, "unc_to_unc_transitions.csv")
        ).values
        self.transitions["SMALL"] = pd.read_csv(
            Path(transition_directory, "small_discharge_transitions.csv")
        ).values
        self.transitions["LARGE"] = pd.read_csv(
            Path(transition_directory, "large_discharge_transitions.csv")
        ).values

        self.nh_los = self.make_nh_los()

        self.location = SingleEvent(
            enum=self.locations.category_enum,
            transition_dict=community_dict,
            key_types=["County", AgeGroup],
        )
        self.location.probabilities = self.location.find_probabilities(
            list(zip(self.model.county_codes, self.model.age_groups))
        )
        self.location.initiate_values(
            count=len(self.model.population), value=self.locations.community
        )

        # ----- Update the community movement probability based on the concurrent conditions
        self.location.probabilites = update_community_probability(
            self.location.probabilities,
            self.model.age_groups,
            self.model.concurrent_conditions,
        )

        for unique_id, item in enumerate(
            self.model.population.Start_Location.astype(int)
        ):
            if item != 0:
                self.assign_los(unique_id=unique_id, new_location=item, initialize=True)
                self.location.values[unique_id] = item
        self.location.previous = self.location.values.copy()

    def step(self):
        self.moved_agents = []
        self.community_movement()
        self.facility_movement()
        self.readmission_movement()

    def community_movement(self):
        """ Find all agents in the community and see which ones are selected to move. Then move those agents
        """
        use_agents = np.where(self.location.values == self.locations.community)[0]
        probabilities = self.location.probabilities[use_agents]
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        unique_ids = use_agents[selected_agents]
        for unique_id in unique_ids:
            self.location_update(unique_id)

    def facility_movement(self):
        """ Move all agents not in the community whose LOS ends today """
        move_ids = [
            key
            for key, value in self.leave_facility_day.items()
            if value == self.model.time
        ]
        for unique_id in move_ids:
            self.location_update(unique_id)

    def location_update(self, unique_id: int):
        self.current = unique_id
        age = self.model.age_groups[unique_id]
        county = self.model.county_codes[unique_id]
        current_location = self.location.values[unique_id]

        # --- Remove any previous ICU Status: This is a new stay
        self.icu_patients.discard(unique_id)
        self.last_movement_day[unique_id] = self.model.time

        # ----- 80% of previous NH patients (who are leaving a hospital) must return to a NH
        previous_category = self.locations.int_to_category[
            self.location.previous[unique_id]
        ]
        if (previous_category == "NH") and (
            current_location in self.locations.all_hospitals
        ):
            new_category = "NH"
            # If more than probability, force an option other than NH to occur
            if self.model.rng.rand() > self.params["tuning"]["NH_to_ST_to_NH"]:
                p = self.find_location_transitions(county, age, current_location)
                count = 0
                while new_category == "NH":
                    new_category = random_selection(
                        self.model.rng.rand(), p, self.locations.categories_list
                    )
                    count += 1
                    if count > 20:
                        raise ValueError("While loops cause problems sir.")
        else:
            p = self.find_location_transitions(county, age, current_location)
            new_category = random_selection(
                self.model.rng.rand(), p, self.locations.categories_list
            )

        if new_category != "COMMUNITY":
            logrecno = self.model.logrecnos[unique_id]
            if new_category == "UNC":
                location_id = self.select_unc_hospital(county, current_location)
            elif new_category in [
                self.locations.category_enum.LARGE.name,
                self.locations.category_enum.SMALL.name,
            ]:
                location_id = self.select_st_hospital(
                    county, logrecno, new_category, current_location
                )
            else:
                location_id = self.select_location(logrecno, new_category)

            # --- Find the location and record a state change
            new_location = self.locations.convert_name(location_id, "int")
            self.model.record_state_change(
                unique_id=unique_id,
                name_state=NameState.LOCATION.value,
                old=current_location,
                new=new_location,
            )
            # --- Select the LOS.
            self.assign_los(unique_id=unique_id, new_location=new_location)
            self.location.values[unique_id] = new_location
        else:
            # ----- If currently in an STACH, prepare agent for possible readmission
            if current_location in self.locations.all_hospitals:
                # --- Check the readmission probability
                if self.model.rng.rand() < self.params["readmission"]["rate"]:
                    # --- If randomly selected for readmission, randomly select a day between 2 and 30 days.
                    self.readmission_date[
                        unique_id
                    ] = self.model.time + self.model.rng.randint(2, 31)
                    self.readmission_location[unique_id] = current_location
            # --- Go Home
            self.model.record_state_change(
                unique_id=unique_id,
                name_state=NameState.LOCATION.value,
                old=current_location,
                new=self.locations.community,
            )

            self.location.values[unique_id] = self.locations.community
            del self.current_los[unique_id]
            del self.leave_facility_day[unique_id]

        # --- Finalize Movement
        self.moved_agents.append(unique_id)
        self.location.previous[unique_id] = current_location
        self.update_death_probabilities(unique_id)

    def assign_los(self, unique_id: int, new_location: int, initialize: bool = False):
        """ Given a new_location (an int), select a LOS for a new patient """
        # ----- If at home, do nothing
        if new_location == self.locations.community:
            return
        location_category = self.locations.int_to_category[new_location]

        # ----- If NH: randomly select LOS from possible LOS
        if location_category == "NH":
            self.current_los[unique_id] = self.nh_los["LOS"][
                self.model.rng.randint(0, len(self.nh_los["LOS"]))
            ]
        else:
            # ----- UNC has distributions by location, all other facilities use their category
            if location_category == "UNC":
                los = self.params["facilities"][
                    self.locations.convert_int(new_location, "ID")
                ]["los"]
            else:
                los = self.params["facilities"][location_category]["los"]
            # ----- Pick a random LOS based on the distribution matching the location
            if los["distribution"] == "Gamma":
                selected_los = int(
                    round(self.model.rng.gamma(los["shape"], los["support"]), 0)
                )
            else:
                raise ValueError(
                    "LOS distribution of type {} is not supported.".format(
                        los["distribution"]
                    )
                )
            # ----- LOS cannot be 0 days. They must stay at location at least one day
            if selected_los == 0:
                selected_los += 1
            self.current_los[unique_id] = selected_los
            # ----- ICU
            if new_location in self.locations.all_hospitals:
                # --- Calculate the ICU probability
                logit = -2.4035
                if self.model.age_groups[unique_id] == AgeGroup.AGE0:
                    logit += 0.1395
                elif self.model.age_groups[unique_id] == AgeGroup.AGE1:
                    logit += 0.1326
                if location_category == "LARGE":
                    logit += 0.1867
                if (
                    self.model.concurrent_conditions[unique_id]
                    == ConcurrentConditions.YES
                ):
                    logit += 0.8169
                if self.current_los[unique_id] > 7:
                    if self.current_los[unique_id] > 30:
                        logit += 0.7337
                    else:
                        logit += 0.2571
                if self.model.rng.rand() < np.exp(logit) / (1 + np.exp(logit)):
                    self.icu_patients.add(unique_id)

        if initialize:
            # --- We assume that some of the LTACH stay is already up
            if location_category == "LT":
                self.current_los[unique_id] = int(self.current_los[unique_id] * (2 / 3))
            # --- Instead of using LOS, we use the "Time Until Leaving" distribution
            if location_category == "NH":
                a_list = self.nh_los["Time_Until_Leaving"]
                self.current_los[unique_id] = a_list[
                    self.model.rng.randint(0, len(a_list))
                ]
        # ----- Set the leave day
        self.leave_facility_day[unique_id] = (
            self.model.time + self.current_los[unique_id]
        )

    def select_location(self, logrecno: int, location_category: str):
        # ----- Find the logrec row
        ft = self.logrec_movement[self.logrec_movement[:, 0] == logrecno][0]
        if location_category == "NH":
            return ft[self.logrec_columns.get_loc("NH")]
        if location_category == "LT":
            return ft[self.logrec_columns.get_loc("LT")]
        return ValueError("Only NH and LT should use this function.")

    def select_unc_hospital(self, county: int, location_int: int):
        """ Select a random UNC hospital based on the agents county """
        ut = self.transitions["UNC"].copy()
        p = ut[ut[:, 0] == county][0][1:]
        # ---- If currently at a hospital, set that hospital probability to 0
        if location_int in self.locations.categories["UNC"]["ints"]:
            p[self.locations.categories["UNC"]["ints"].index(location_int)] = 0
        if sum(p) > 0:
            # --- Select hospital from possible UNC locations
            p = normalize_and_create_cdf(p)
            return random_selection(
                self.model.rng.rand(), p, self.locations.categories["UNC"]["ids"]
            )
        # ----- If no hospital remains, Pick another transition row that includes your hospital
        ut_probs = ut[:, 1:].copy()
        # --- If at a UNC hospital
        if location_int in self.locations.categories["UNC"]["ints"]:
            h_index = self.locations.categories["UNC"]["ints"].index(location_int)
            ut_probs = ut_probs[ut_probs[:, h_index] > 0]
            ut_probs[:, h_index] = 0
        # --- Rows with 2+ options
        ut_probs = ut_probs[np.count_nonzero(ut_probs, axis=1) > 0]
        if len(ut_probs) > 0:
            p = ut_probs[
                self.model.rng.randint(0, ut_probs.shape[0]),
            ]
            p = normalize_and_create_cdf(p)
            return random_selection(
                self.model.rng.rand(), p, self.locations.categories["UNC"]["ids"]
            )
        # ----- If no options, send to a large UNC hospital TODO:
        return random_selection(self.model.rng.rand(), [0.5, 1.0], ["UNC_7", "UNC_8"])

    def select_st_hospital(
        self, county: int, logrecno: int, selected_category: str, location_int: int
    ) -> int:
        """ Select a random nonUNC stach. """
        if selected_category == self.locations.category_enum.LARGE.name:
            large_options = self.locations.categories["LARGE"]["ids"]
            # --- Select large transitions
            ut = self.transitions["LARGE"].copy()
            p = ut[ut[:, 0] == county][0][1:]
            # --- Cannot transition to current hospital
            if location_int in self.locations.categories["LARGE"]["ints"]:
                p[self.locations.categories["LARGE"]["ints"].index(location_int)] = 0
            # If any large hospitals remain for that county:
            if sum(p) > 0:
                p = normalize_and_create_cdf(p)
                return random_selection(self.model.rng.rand(), p, large_options)
            # If not, try to use logrecno location
            else:
                fts = self.logrec_movement
                ft = fts[fts[:, self.logrec_columns.get_loc("logrecno")] == logrecno][0]
                new_location = ft[self.logrec_columns.get_loc("LARGE")]
                # --- Make sure they are not currently there
                current_location = self.locations.convert_int(location_int, "ID")
                if new_location == current_location:
                    # --- Select random location from other logrecnos in county
                    large_hospitals = fts[
                        fts[:, self.logrec_columns.get_loc("county")] == county
                    ]
                    large_hospitals = large_hospitals[
                        :, self.logrec_columns.get_loc("LARGE")
                    ]
                    if any(large_hospitals != current_location):
                        new_location = large_hospitals[
                            large_hospitals != current_location
                        ][0]
                    else:
                        # --- Select random large_hospital
                        new_location = self.model.rng.choice(
                            [
                                large
                                for large in large_options
                                if large != current_location
                            ]
                        )
                return new_location
        # --- Send to Small
        else:
            ut = self.transitions["SMALL"]
            small_options = self.locations.categories["SMALL"]["ids"]
            p = ut[ut[:, 0] == county][0][1:]
            # ---- If currently at a small hospital, set that hospital probability to 0
            if location_int in self.locations.categories["SMALL"]["ints"]:
                p[self.locations.categories["SMALL"]["ints"].index(location_int)] = 0
            if sum(p) > 0:
                p = normalize_and_create_cdf(p)
                return random_selection(self.model.rng.rand(), p, small_options)
            # --- As a last resort, select a random small hospital
            return self.model.rng.choice(
                [
                    s
                    for s in small_options
                    if s != self.locations.convert_int(location_int, "ID")
                ]
            )

    def make_nh_los(self):
        nh_los = pd.read_csv(Path(self.model.experiment_dir, "data/input/NH_LOS.csv"))
        nh_los = nh_los[(0 < nh_los.los) & (nh_los.los < 2000)].copy()
        nh_los = nh_los.astype(int)
        nh_los.cfreq = nh_los.cfreq.apply(lambda x: int(x * 0.1))
        a_list = []
        for row in nh_los.iterrows():
            a_list.extend([row[1].los] * int(row[1].cfreq))

        nh_los2 = pd.read_csv(
            Path(self.model.experiment_dir, "data/input/nh_time_until_leaving.csv")
        )

        nh_dict = dict()
        nh_dict["LOS"] = a_list
        nh_dict["Time_Until_Leaving"] = [item[0] for item in nh_los2.values]
        return nh_dict

    def find_location_transitions(self, county: int, age: int, loc_int: int) -> float:
        return list(self.facility_transitions[(county, age, loc_int)])


class NcLocations(Locations):
    """ Create the location map for facilities
    """

    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory

        for item in ["COMMUNITY", "UNC", "LARGE", "SMALL", "LT", "NH"]:
            self.categories[item] = dict()
        self.category_enum = IntEnum(
            "Facility Categories", {k: _ for _, k in enumerate(self.categories)}
        )
        self.categories_list = list(self.categories.keys())
        for item in self.categories:
            self.categories[item]["ints"] = [0] if item == "COMMUNITY" else []

        self.facilities["COMMUNITY"] = {
            "int": 0,
            "int_category": self.category_enum.COMMUNITY.value,
            "ID": "COMMUNITY",
            "category": "COMMUNITY",
        }
        self.number_of_items = 1
        # ----- Hospitals
        for _, row in pd.read_csv(
            Path(self.directory, "data/IDs/hospital_ids.csv")
        ).iterrows():
            self.categories[row.Category]["ints"].append(self.number_of_items)
            self.add_value(row, self.category_enum[row.Category].value)

        # ----- Nursing Homes
        for _, row in pd.read_csv(
            Path(self.directory, "data/IDs/nh_ids.csv")
        ).iterrows():
            self.categories[row.Category]["ints"].append(self.number_of_items)
            self.add_value(row, self.category_enum.NH.value)

        # ----- LTACHs
        for _, row in pd.read_csv(
            Path(self.directory, "data/IDs/lt_ids.csv")
        ).iterrows():
            self.categories[row.Category]["ints"].append(self.number_of_items)
            self.add_value(row, self.category_enum.LT.value)

        self.all_hospitals = (
            self.categories["UNC"]["ints"]
            + self.categories["LARGE"]["ints"]
            + self.categories["SMALL"]["ints"]
        )

        self.enum = IntEnum(
            "Locations", {k: v["int"] for k, v in self.facilities.items()}
        )

        for name in ["UNC", "LARGE", "SMALL", "LT", "NH"]:
            self.categories[name]["ids"] = [
                self.enum(item).name for item in self.categories[name]["ints"]
            ]

        self.community = self.category_enum.COMMUNITY.value
        self.int_to_category = dict()
        for location in self.enum:
            self.int_to_category[location.value] = self.facilities[location.name][
                "category"
            ]


def location_dict(model, lt):
    ld = {}
    for item in lt.values:
        key_0 = item[lt.columns.get_loc("County_Code")]
        key_1 = item[lt.columns.get_loc("Age")]
        # There is only one row for NH - apply to all NHs though
        if item[lt.columns.get_loc("Facility")] == "NH":
            cdf = create_cdf(item[lt.columns.get_loc("COMMUNITY") :])
            for key_2 in model.locations.categories["NH"]["ints"]:
                ld[(key_0, key_1, key_2)] = cdf
        # There is only one row for LT - apply to all LTs though
        elif item[lt.columns.get_loc("Facility")] == "LT":
            cdf = create_cdf(item[lt.columns.get_loc("COMMUNITY") :])
            for key_2 in model.locations.categories["LT"]["ints"]:
                ld[(key_0, key_1, key_2)] = cdf
        else:
            key_2 = model.locations.enum[item[lt.columns.get_loc("Facility")]].value
            ld[(key_0, key_1, key_2)] = create_cdf(
                item[lt.columns.get_loc("COMMUNITY") :]
            )
    return ld
