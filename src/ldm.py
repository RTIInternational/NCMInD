
from sqlite3 import connect
from pandas.io import sql
from pandas import read_csv
from numpy import ones, floor, random

from src.parameters import *
from src.cdi_functions import *
from src.location_functions import *
from src.life_functions import life_movement, update_death_probabilities


class LDM:
    def __init__(self, exp_dir, scenario, run=''):
        """ LDM: Location & Disease model, is a class in which to run agent-based simulations. The base model,
        "location" will move agents through community/facility nodes. Disease models can be specified in the
        parameters json.

        Current disease models implemented:
            cdi: specify "CDI"
            cre: specify "CRE" - TODO

        Parameters:
        -----------
        exp_dir : string
            The directory of an experiment, such as 'NCMIND/demo'

        scenario : string
            Name of the specific scenario within the experiment, such as "default"

        run : string
            Name of the specific run within a scenario, such as "run_1"

        Attributes:
        -----------
        output_dir : PosixPath
            directory to place all model output

        parameters : Parameters
            The parameters as read from parameters.json in the scenario or run directory

        np_rng : numpy.random()
            A random number generator from numpy that has a random seed set by the parameters

        conn : sqlite3.Connection
            A connection object to the in-memory sqlite database

        cur : conn.cursor()
            The cursor for the sqlite connection

        locations : class Locations
            From location_functions, contains the locations and their corresponding integer values

        community_probability : array
            Probabilities of leaving the community for each demographic

        death_probability : array
            Probabilities of dying for each demographic

        time : int
            The current time (day) of the model

        agents_to_recreate : list
            IDs of all agents that have died and need to be recreated to maintain population levels

        agents_to_update : list
            IDs of all agents who have moved locations and need their death probabilities updated

        population : DataFrame
            A pandas DataFrame containing the initial population that was used for the model

        columns : dict
            Order and column names of the models data array

        data : array
            Model data, each row is an agent, each column is an attribute described in 'columns'

        """

        # ----- Setup the model directory structure
        self.exp_dir = Path(exp_dir)
        self.output_dir = Path(self.exp_dir, scenario, run, "model_output")
        Path(self.exp_dir, scenario, run).mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # ------ Setup the model parameters
        if run != "":
            self.parameters = Parameters(Path(self.exp_dir, scenario, run, 'parameters.json'))
        else:
            self.parameters = Parameters(Path(self.exp_dir, scenario, 'parameters.json'))
        # --- Set the random seed
        self.np_rng = random.RandomState(self.parameters.base['seed'])

        # ----- Prepare for Input Data
        self.locations = Locations()
        self.stachs = list()
        self.location_transitions = dict()
        self.demographic_table = array(1)
        self.community_probability = array(1)
        self.death_probability = array(1)
        # --- Read Inputs
        self.read_inputs()

        # ----- Prepare for Agents
        self.time = 0
        self.agents_to_recreate = []
        self.agents_to_update = []
        self.daily_counts = DataFrame()
        self.columns = dict()
        self.data = array(1)
        # --- Read Agents
        self.population = self.read_population()
        # --- Load Agents
        self.load_agents()

        # ----- For Event Tracking: Create the SQL Connection
        self.conn = None
        self.cur = None
        self.event_tracker_columns = dict()
        for i, item in enumerate(['Unique_ID', 'Time', 'State', 'Location', 'LOS', 'Old', 'New', 'County']):
            self.event_tracker_columns[item] = i
        self.create_sql_connection()

        # ----- For CDI Only
        if 'CDI' in self.parameters.base['model_type']:
            self.cdi = dict()
            prepare_cdi(self)

        # ----- Generate output data
        self.collect_agents(initiate=True)

    # ------------------------------------------------------------------------------------------------------------------
    # ------ Run Model
    # ------------------------------------------------------------------------------------------------------------------
    def run_model(self):
        """ Run through a location and/or disease model for all agents for all days
        """
        for day in range(self.parameters.base['time_horizon']):
            # ----- Run the location model
            if 'location' in self.parameters.base['model_type']:
                self.location_step(day)

            # ----- Run the disease Model
            if 'CDI' in self.parameters.base['model_type']:
                cdi_step(self)

            # ----- Collect the agents information if requested
            if self.time in self.parameters.base['print_days']:
                temp_df = DataFrame(self.data, columns=self.columns.keys())
                temp_df.to_csv(Path(self.output_dir, "agent_info_" + str(self.time) + ".csv"), index=False)

            # ----- Daily Cleanup
            self.collect_agents()
            self.regenerate_agents()
            self.agents_to_update = list()

        self.save_output()

    def location_step(self, day):
        self.time = day
        print('Starting day: %s' % day) if day % 25 == 0 else None

        # ----- Simulate Location & Life Movement
        community_movement(self)            # Check movement in the community
        facility_movement(self)             # Check movement in the facilities
        life_movement(self)                 # Check life movement
        update_death_probabilities(self)    # Death is only updated based on location

    # ------------------------------------------------------------------------------------------------------------------
    # ------ Output Functions
    # ------------------------------------------------------------------------------------------------------------------
    def make_events(self):
        events = DataFrame(self.cur.execute("SELECT * FROM event_tracker").fetchall(),
                           columns=list(self.event_tracker_columns.keys()))
        return events

    def save_output(self):
        """ Save model output when the model is finished running
        """
        # ----- Save all of the model events
        self.make_events().to_csv(Path(self.output_dir, 'model_events.csv'), compression='gzip', index=False)

        # ----- Save the daily counts
        self.daily_counts.to_csv(Path(self.output_dir, 'daily_counts.csv'), index=True)

        # ----- Save the CDI Cases
        if 'CDI' in self.parameters.base['model_type']:
            columns = ['Time', 'Unique_ID', 'Location', 'Type', 'NHSN_Desc', 'County']
            cdi_cases = DataFrame(self.cdi['cases'], columns=columns)
            cdi_cases.to_csv(Path(self.output_dir, 'CDI_cases.csv'), index=False)

    def collect_agents(self, initiate=False):
        # ----- CDI Version
        if 'CDI' in self.parameters.base['model_type']:
            collect_cdi_agents(self, initiate)
        # ----- Normal Version
        else:
            if initiate:
                daily_count_index_list = []
                for a_life in [0, 1]:
                    for a_location in self.locations.values.values():
                        daily_count_index_list.append((a_life, a_location))
                daily_counts = DataFrame(daily_count_index_list, columns=['Life', 'Location'])
                self.daily_counts = daily_counts.set_index(['Life', 'Location'])
            df = DataFrame(self.data[:, [self.columns['life_status'], self.columns['location_status']]],
                           columns=['Life', 'Location'])
            df = df.groupby(by=['Life', 'Location']).size()
            df.index.names = ['Life', 'Location']
            self.daily_counts[self.time] = df
            self.daily_counts = self.daily_counts.fillna(0)

    # ------------------------------------------------------------------------------------------------------------------
    # ------ Load Input Files
    # ------------------------------------------------------------------------------------------------------------------
    def read_inputs(self):
        # ----- 1: Read location transitions and create an IntEnum the model can use
        lt = read_csv(Path(self.exp_dir, '../data/input/transitions', self.parameters.base['location_file']))
        self.locations.add_values(lt['Location'].unique())
        self.stachs = [l for l in self.locations.ints if self.locations.ints[l] not in ['COMMUNITY', 'NH', 'LT']]
        # --- Update the location transitions based on the parameter values
        lt = update_location_transitions(self, lt)
        # --- Set the location to be the Enum value
        lt['Location'] = [self.locations.values[item] for item in lt['Location']]
        # --- Convert to a dictionary for quick look-up
        for item in lt.values:
            self.location_transitions[(item[0], item[-1])] = item[5:-1]

        # ----- 2: Read in the Community Transition Document
        ct = read_csv(Path(self.exp_dir, '../data/input/transitions/community_transitions.csv'))
        ct['probability'] = ct['probability'] * self.parameters.tuning["community_probability_multiplier"]
        ct['Index'] = ct.index
        self.community_probability = ct.probability.values
        self.demographic_table = ct[['County_Code', 'Sex', 'Age', 'Race']].reset_index().values

        # ----- 3: Death Probabilities
        dp = read_csv(Path(self.exp_dir, '../data/input/transitions/death_probabilities.csv'))
        # --- Death probability df does not have the appropriate indices, we will create them here.
        dp = ct[['County_Code', 'Sex', 'Age', 'Race']].merge(dp, on=['Sex', 'Age', 'Race'])
        dp = dp.sort_values(by=['County_Code', 'Sex', 'Age', 'Race']).reset_index(drop=True)
        self.death_probability = dp.Probability.values

        # ----- 4: Correct Parameters
        self.correct_parameters()

    # ------------------------------------------------------------------------------------------------------------------
    # ------ SQL Connection
    # ------------------------------------------------------------------------------------------------------------------
    def create_sql_connection(self):
        self.conn = connect(":memory:")
        self.cur = self.conn.cursor()
        # ----- Add Event Tracker Table
        sql.to_sql(DataFrame(
            columns=list(self.event_tracker_columns.keys()), dtype=int),
            name='event_tracker', con=self.conn, index=False)
        sql_command = "CREATE INDEX id_index ON event_tracker (Unique_ID);"
        self.cur.execute(sql_command)

    def collapse_sql_connection(self):
        """
        Collapse the SQL connection. This is needed when copying models. These objects cannot be pickled.
        """
        self.conn.close()
        self.conn = None
        self.cur = None

    # ------------------------------------------------------------------------------------------------------------------
    # ------ Initialization Functions
    # ------------------------------------------------------------------------------------------------------------------
    def read_population(self):
        print('1/2: Reading population...')
        p = self.parameters
        population =\
            read_csv(Path(self.exp_dir, "../data/synthetic_population", p.base['population_file']))
        if p.base['limit_pop'] < population.shape[0]:
            population = population.sample(p.base['limit_pop'],
                                           random_state=self.parameters.base['seed']).reset_index(drop=True)
        # ----- SQL does not look up int64s. Convert to Float
        for column in ['County_Code', 'Sex', 'Age', 'Race']:
            population[column] = population[column].astype(float)

        return population

    def load_agents(self):
        print('2/2: Loading population...')

        # --- 1st column
        self.columns['unique_id'] = 0
        unique_id = array(self.population.index.values)
        # --- 2nd Column
        self.columns['age_group'] = 1
        age_group = array(self.population.Age.values)
        # --- 3rd Column
        self.columns['demo_id'] = 2
        demo_id = array(find_demo_id(self.demographic_table, self.population.County_Code.values,
                                     self.population.Sex.values, self.population.Age.values,
                                     self.population.Race.values), dtype=int)
        # --- 4th & 5th
        self.columns['community_probability'] = 3
        self.columns['location_status'] = 4
        community_probability = find_probability(self.community_probability, demo_id)
        location_status = array([self.locations.values[item] for item in self.population.Start_Location])
        # --- 6th Column
        self.columns['life_status'] = 5
        life_status = ones((len(unique_id),))
        # --- 7th Column
        self.columns['current_los'] = 6
        current_los = array([select_los(self, item) for item in location_status])
        # --- 8th & 9th Column
        self.columns['nh_patient'] = 7
        self.columns['leave_facility_day'] = 8
        nh_patient = []
        for i, item in enumerate(location_status):
            if item == self.locations.values['NH']:
                nh_patient.append(1)
                # Assume the NH patients are already 1/2 way done with their stay
                current_los[i] = int(floor(current_los[i] / 2))
            else:
                nh_patient.append(0)
        leave_facility_day = self.time + current_los
        # --- 10th Column
        self.columns['death_probability'] = 9
        death_probability = find_probability(self.death_probability, demo_id)
        # --- Find the right death multiplier and save the values
        death_multiplier = [self.parameters.death_multipliers[item] for item in location_status]
        death_probability = death_probability * death_multiplier
        death_probability = death_probability * death_multiplier

        self.data = vstack([unique_id, age_group, demo_id, community_probability, location_status, life_status,
                            current_los, nh_patient, leave_facility_day, death_probability]).T

    # ------------------------------------------------------------------------------------------------------------------
    # ------ Miscellaneous Functions
    # ------------------------------------------------------------------------------------------------------------------
    def regenerate_agents(self):
        # ---- Regenerate the dead agents every 15 days
        if (self.time % 15 == 0) and (len(self.agents_to_recreate) > 0):
            agent_ids = [int(item) for item in self.agents_to_recreate if item < self.population.shape[0]]
            if len(agent_ids) > 0:
                # Fix Location
                new_data = self.data[agent_ids, :]
                start = self.data.shape[0]
                new_data[:, self.columns['unique_id']] = range(start, start + len(agent_ids))
                new_data[:, self.columns['current_los']] = -1
                new_data[:, self.columns['leave_facility_day']] = -1
                new_data[:, self.columns['life_status']] = 1
                new_data[:, self.columns['location_status']] = self.locations.values['COMMUNITY']

                self.data = vstack((self.data, new_data))

                # Fix Disease State
                if 'CDI' in self.parameters.base['model_type']:
                    regenerate_cdi_variables(self, agent_ids, new_data)

            self.agents_to_recreate = []

    def record_state_change(self, row, name_state, old, new):
        """ Add a state change to the state_events dictionary.

        Parameters
        ----------
        row : numpy array
            The row of data for the agent

        name_state : IntEnum
            The IntEnum of the state being updated

        old : the old enum value for the agent

        new : the new enum value for the agent
        """
        county = self.demographic_table[int(row[self.columns['demo_id']])][1]

        a_tuple = (row[self.columns['unique_id']], self.time, name_state.value,
                   row[self.columns['location_status']], row[self.columns['current_los']], old, new, int(county))
        sql1 = ''' INSERT INTO event_tracker VALUES(?,?,?,?,?,?,?,?) '''
        self.cur.execute(sql1, a_tuple)

    def correct_parameters(self):
        """ Parameters are stored with strings (i.e. "COMMUNITY") instead of integers (i.e. 0). """
        self.parameters.length_of_stay = self.switch_keys(self.parameters.length_of_stay)
        self.parameters.death_multipliers = self.switch_keys(self.parameters.death_multipliers)
        if 'CDI' in self.parameters.base['model_type']:
            p = self.parameters.cdi
            p['colonization']['tuning'] = self.switch_keys(p['colonization']['tuning'])
            p['colonization']['initialization'] =\
                self.switch_keys(p['colonization']['initialization'])
            p['antibiotics']['distributions'] = self.switch_keys(p['antibiotics']['distributions'])
            p['antibiotics']['facility'] = self.switch_keys(p['antibiotics']['facility'])
            p['forced_co_cdi'] = self.switch_keys(p['forced_co_cdi'])

    def switch_keys(self, d):
        """ Switch strings for integers - for location (facility and community) names """
        new_d = dict()
        for key in d.keys():
            new_d[self.locations.values[key]] = d[key]
        return new_d
