import pandas as pd
import numpy as np

from sqlite3 import connect
from pandas.io import sql
from pathlib import Path

from src.jit_functions import assign_conditions
from src.parameters import Parameters
from src.state import LifeState, AgeGroup
from src.life import Life
from src.modules import disease_models, location_models


class Ldm:
    def __init__(self, experiment: str, scenario: str, run: str):
        """ Ldm: Location & Disease model - a class built to run agent-based simulations.

        Current disease models implemented:
            - cdi: Use: "disease_model": "cdi" in the parameters.json
            - cre: Use: "disease_model": "cre"
        """

        # ----- Setup the model directory structure
        self.experiment_dir = Path(experiment)
        self.scenario_dir = Path(self.experiment_dir, scenario)
        self.run_dir = Path(self.scenario_dir, run)
        self.output_dir = Path(self.run_dir, "model_output")
        self.run_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # ----- Setup the model parameters
        self.params = Parameters(Path(self.run_dir, "parameters.json"))
        self.seed = self.params.base["seed"]
        self.rng = np.random.RandomState(self.seed)

        self.time = 0
        self.daily_counts = pd.DataFrame()

        # ----- Population
        self.population = self.read_population()
        self.unique_ids = np.array(self.population.index.values, dtype=np.int32)
        self.county_codes = self.population.County_Code.values
        self.age_groups = self.population.Age.values
        self.logrecnos = self.population.logrecno.values
        self.population = self.population.drop(
            ["County_Code", "Age", "logrecno"], axis=1
        )

        self.concurrent_conditions = assign_conditions(
            self.age_groups, self.rng.rand(len(self.population))
        )

        # ----- For Event Tracking: Create the SQL Connection
        self.event_columns = [
            "Unique_ID",
            "Time",
            "State",
            "Location",
            "LOS",
            "Old",
            "New",
            "County",
        ]
        self.conn, self.cur = None, None
        self.create_sql_connection()

        self.location = location_models[self.params.base["location_model"]](
            model=self, params=self.params.location
        )
        self.disease = disease_models[self.params.base["disease_model"]](
            model=self, params=self.params.disease
        )

        life_dict = dict()
        for k1, v1 in self.params.life["death_probabilities"].items():
            for k2, v2 in self.params.life["death_multipliers"].items():
                life_dict[
                    AgeGroup[k1].value, self.location.locations.category_enum[k2].value
                ] = (v1 * v2)

        self.life = Life(
            model=self,
            params=self.params.life,
            enum=LifeState,
            transition_dict=life_dict,
            key_types=[AgeGroup, self.location.locations.category_enum],
        )
        # --- Death probability is based on Age + Location
        locations = [
            self.location.locations.convert_int(l, "int_category")
            for l in self.location.location.values
        ]
        self.life.probabilities = self.life.find_probabilities(
            list(zip(self.age_groups, locations))
        )

        self.disease.collect_agents(initiate=True)

    # ------------------------------------------------------------------------------------------------------------------
    # ------ Read population
    # ------------------------------------------------------------------------------------------------------------------
    def read_population(self) -> pd.DataFrame:
        print("One moment: Reading population ...")
        pop = pd.read_csv(
            self.params.base["population_file"],
            dtype={
                "County_Code": np.int64,  # Must be int for quick dictionary lookups
                "Sex": np.int8,
                "Age": np.int64,  # Must be int for quick dictionary lookups
                "Age_Years": np.int8,
                "tract": np.int32,
                "blkgrp": np.int8,
                "logrecno": np.int16,
                "Start_Location": np.int16,
            },
        )
        if self.params.base["limit_pop"] < pop.shape[0]:
            pop = pop.sample(
                self.params.base["limit_pop"], random_state=self.seed
            ).reset_index(drop=True)
        return pop

    # ------------------------------------------------------------------------------------------------------------------
    # ------ Run Model
    # ------------------------------------------------------------------------------------------------------------------
    def run_model(self):
        for day in range(self.params.base["time_horizon"]):
            print("Starting day: %s" % day) if day % 15 == 0 else None
            self.time = day
            self.step()
        self.daily_counts = self.daily_counts.fillna(0)
        self.save_output()

    def step(self):
        self.life.step()
        self.location.step()
        self.disease.step()
        self.regenerate_agents()

    def regenerate_agents(self):
        """ Every 15 days, regenerate new agents for the agents who have died
        """
        agent_ids = []
        if (self.time % 15 == 0) and (len(self.life.agents_to_recreate) > 0):
            agent_ids = [
                item
                for item in self.life.agents_to_recreate
                if item < self.population.shape[0]
            ]
        if len(agent_ids) > 0:
            l1 = len(self.unique_ids)
            self.unique_ids = np.append(
                self.unique_ids, [range(l1, l1 + len(agent_ids))]
            )
            self.county_codes = np.append(
                self.county_codes, self.county_codes[agent_ids]
            )
            self.logrecnos = np.append(self.logrecnos, self.logrecnos[agent_ids])
            self.age_groups = np.append(self.age_groups, self.age_groups[agent_ids])

            # --- Assign concurrent conditions
            conditions = assign_conditions(
                self.age_groups[agent_ids], self.rng.rand(len(agent_ids))
            )
            self.concurrent_conditions = np.append(
                self.concurrent_conditions, conditions
            ).astype(np.int8)

            self.location.regenerate_agents(agent_ids)
            self.disease.regenerate_agents(agent_ids)
        self.life.agents_to_recreate = []

    # ------------------------------------------------------------------------------------------------------------------
    # ------ Output Functions
    # ------------------------------------------------------------------------------------------------------------------
    def make_events(self) -> pd.DataFrame:
        query = "SELECT * FROM event_tracker"
        return pd.DataFrame(
            self.cur.execute(query).fetchall(), columns=self.event_columns
        )

    def save_output(self):
        # ----- Save all of the model events, daily counts, and infectious cases
        self.make_events().to_csv(
            Path(self.output_dir, "model_events.csv"), compression="gzip", index=False
        )
        self.daily_counts.to_csv(Path(self.output_dir, "daily_counts.csv"), index=True)
        self.disease.save_output()

    # ------------------------------------------------------------------------------------------------------------------
    # ------ SQL Connection
    # ------------------------------------------------------------------------------------------------------------------
    def create_sql_connection(self):
        """ Create the SQL connection. SQL is used for quickly storing model events
        """
        self.conn = connect(":memory:")
        self.cur = self.conn.cursor()
        # ----- Add Event Tracker Table
        sql.to_sql(
            pd.DataFrame(columns=list(self.event_columns), dtype=int),
            name="event_tracker",
            con=self.conn,
            index=False,
        )
        self.cur.execute("CREATE INDEX id_index ON event_tracker (Unique_ID);")

    def collapse_sql_connection(self):
        """ Collapse the SQL connection. This is needed when copying models. These objects cannot be pickled. """
        self.conn.close()
        self.conn, self.cur = None, None

    # ------------------------------------------------------------------------------------------------------------------
    # ------ Miscellaneous Functions
    # ------------------------------------------------------------------------------------------------------------------
    def record_state_change(self, unique_id: int, name_state: int, old: int, new: int):
        """ Add a state change to the state_events dictionary.
        """
        los = self.location.current_los.get(unique_id, 0)
        location_status = self.location.location.values[unique_id]
        county_code = self.county_codes[unique_id]

        a_tuple = (
            unique_id,
            self.time,
            name_state,
            location_status,
            los,
            old,
            new,
            county_code,
        )
        a_tuple = tuple([int(item) for item in a_tuple])
        self.cur.execute(
            """ INSERT INTO event_tracker VALUES(?,?,?,?,?,?,?,?) """, a_tuple
        )
