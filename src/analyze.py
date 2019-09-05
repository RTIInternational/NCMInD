
from src.state import *
from src.parameters import Parameters
from src.location_functions import Locations
from pathlib import Path
from pandas import DataFrame, Series, read_csv

import plotly
import plotly.graph_objs as go


class Analyze:
    """
    Provides common analysis routines for the NCMIND model output data.
    """

    def __init__(self, exp_dir="NCMIND/demo", scenario="CDI_example", run="", a_object=None):
        """
        """
        # ----- Create the location MAP. Helps with cleaner visualizations
        self.map = {
            'COMMUNITY': 'Community',
            'CALDWELL': 'Caldwell',
            'CHATHAM': 'Chatham',
            'HIGHPOINT': 'High Point',
            'JOHNSTON': 'Johnston',
            'LENOIR': 'Lenoir',
            'MARGARET': 'Margaret',
            'NASH': 'Nash',
            'REX': 'Rex',
            'UNC_CH': 'UNC Chapel Hill',
            'WAYNE': 'Wayne',
            'ST': 'S-TACH',
            'LT': 'L-TACH',
            'NH': 'Nursing Home',
            'DEAD': 'Deceased'
        }

        self.cdiff_map = {
            'SUSCEPTIBLE': 'Susceptible',
            'COLONIZED': 'Asymptomatically Colonized',
            'CDI': 'CDI',
            'DEAD': 'Dead'
        }

        if a_object:
            # ----- Setup the class based on an object
            self.exp_dir = a_object.exp_dir
            self.output_dir = a_object.output_dir
            self.parameters = a_object.parameters
            self.daily_counts = a_object.daily_counts.reset_index()
            self.events = a_object.make_events()
            self.locations = a_object.locations
        else:
            # ----- Setup the class based on a directory
            self.exp_dir = Path(exp_dir)
            self.output_dir = Path(exp_dir, scenario, run, 'model_output')
            self.parameters = Parameters(Path(exp_dir, scenario, run, "parameters.json"))
            self.daily_counts = read_csv(Path(self.output_dir, 'daily_counts.csv'))
            location_transitions = read_csv(Path(self.exp_dir, '../data/input/transitions/',
                                                 self.parameters.base['location_file']))
            # noinspection PyArgumentList
            self.locations = Locations()
            self.locations.add_values(location_transitions['Location'].unique())

            if 'CDI' in self.parameters.base['model_type']:
                self.cdi_cases = read_csv(Path(self.output_dir, 'CDI_cases.csv'))

            # ---- Some files may be compressed now
            try:
                self.events = read_csv(Path(self.output_dir, 'model_events.csv'))
            except UnicodeDecodeError:
                self.events = read_csv(Path(self.output_dir, 'model_events.csv'), compression='gzip')

        self.UNC =\
            [i for i in list(self.locations.ints) if self.locations.ints[i] not in ['COMMUNITY', 'LT', 'NH', 'ST']]
        self.ST = [self.locations.values['ST']]
        self.LT = [self.locations.values['LT']]
        self.NH = [self.locations.values['NH']]
        self.COMMUNITY = [self.locations.values[item] for item in list(self.locations.values) if item in ['COMMUNITY']]
        self.catchment_counties =\
            [1, 21, 23, 27, 35, 37, 49, 51, 57, 61, 63, 65, 67, 69, 79, 81, 83, 85, 89, 101, 103, 105, 107, 125, 127,
             129, 133, 135, 145, 147, 149, 151, 155, 161, 163, 175, 183, 189, 191, 193, 195]

    def available_population(self, locations=None, risks=None, antibiotics=None, risk_column='CDI'):
        """ Count the alive population at each day, filtered by the parameters

        Parameters
        ----------
        locations : list
            location integers

        risks : list
            risk integers

        antibiotics : list
            antibiotic integers

        risk_column : str
            specify the name of the risk column

        Returns
        -------
        pandas DataFrame

        """
        # ----- Find Population Total
        dc = self.daily_counts
        # ----- Only consider alive people
        dc = dc[dc['Life'] == 1]
        # ----- Filter to Specific Locations, risks, or antibiotic use
        if locations:
            dc = dc[dc['Location'].isin(locations)]
        if risks:
            dc = dc[dc[risk_column].isin(risks)]
        if antibiotics:
            dc = dc[dc['Antibiotics'].isin(antibiotics)]
        # ----- Filter to integer columns
        dc = dc.loc[:, [item for item in dc.columns if item not in ['Antibiotics', 'Life', 'Location', risk_column]]]

        return DataFrame(dc.sum().values)

    def filter_events(self, state, new=None, old=None, locations=None):
        """ Filter model events to those that match the parameters

        Parameters
        ----------
        state : NameState
            specify the state of interest

        new : list
            list of values the New column can equal

        old : list
            list of values the Old state can equal

        locations : list
            list of locations the agent can be when an event happens

        Returns
        -------
        events : Pandas DataFrame

        """
        events = self.events
        events = events[events['State'] == state]
        if locations:
            events = events[events['Location'].isin(locations)]
        if old:
            events = events[events['Old'].isin(old)]
        if new:
            events = events[events['New'].isin(new)]

        return events

    def sum_events(self, state, new=None, old=None, locations=None):
        """ Sum the total number of events that occurred, by day

        Parameters
        ----------
        See filter_events()

        Returns
        -------
        pandas DataFrame of summed events by day

        """

        events = self.filter_events(state, new, old, locations)
        return DataFrame(events.groupby(by='Time').size())

    def calculate_patient_days(self, unc_only=False):
        """ Calculate the total number of patient days for each facility

        Parameters
        ----------
        unc_only : boolean (default : False)
            Filter to only patients from UNC catchment area

        Return
        ------
        names : list
            names of the facilities, in order, to match patient_days

        patient_days : list
            list of facility names and their patient days

        """
        # ----- For patients who have already left
        location_events = self.events[self.events['State'] == NameState.LOCATION]

        if unc_only:
            location_events = location_events[location_events.County.isin(self.catchment_counties)]

        # ----- For patients who have not yet left
        last_events = location_events.groupby(by=['Unique_ID']).last()

        patient_days = []
        names = []
        for name in list(self.locations.values):
            names.append(name)
            # --- All the events where the patient left a location
            facility_events = location_events[location_events['Old'] == self.locations.values[name]]
            # --- All the places where a patient is still at the location
            last_events = last_events[last_events['New'] == self.locations.values[name]]
            last_events['Total_Days'] = self.parameters.base['time_horizon'] - last_events.Time
            patient_days.append(facility_events.LOS.sum() + last_events.Total_Days.sum())

        return names, patient_days

    # ------------------------------------------------------------------------------------------------------------------
    # ----- National Healthcare Safety Network (NHSN Definitions)
    # ------------------------------------------------------------------------------------------------------------------
    def onset_cdi(self, skip_days=90, unc_only=False):
        """ Calculate onset CDI for both the hospital and the inpatient community level

        Parameters
        ----------
        skip_days : int
            Filter to only CDI cases that happened after X days

        unc_only : boolean (default : False)
            Filter to only patients from UNC catchment area
        """

        # ----- Incident CDI Cases after X days (skip_days)
        cases = self.cdi_cases[self.cdi_cases.Time > skip_days]
        cases = cases[cases.Type != 'Duplicate']
        cases = cases[cases.NHSN_Desc != 'Community']

        if unc_only:
            cases = cases[cases.County.isin(self.catchment_counties)]

        # ----- Create the 13 HO-CDI and the 13 CO-CDI Numbers
        multiplier = 10_000 * self.parameters.base['time_horizon'] / (self.parameters.base['time_horizon'] - skip_days)
        patient_days = self.calculate_patient_days(unc_only)
        names, ho_rates, co_rates = list(), list(), list()
        unc_ho, unc_co = 0, 0
        unc_patient_days = 0
        ltcf_count = 0
        ltcf_patient_days = patient_days[1][patient_days[0].index('LT')] + patient_days[1][patient_days[0].index('NH')]

        for name in list(self.locations.values):
            if name != 'COMMUNITY':
                i = self.locations.values[name]
                names.append(name)
                ho_count = cases[(cases.Location == i) & (cases.NHSN_Desc == 'HO CDI')].shape[0]
                co_count = cases[(cases.Location == i) & (cases.NHSN_Desc != 'HO CDI')].shape[0]
                if self.locations.values[name] in self.UNC:
                    unc_ho += ho_count
                    unc_co += co_count
                    unc_patient_days += patient_days[1][i]
                if self.locations.values[name] in self.NH + self.LT:
                    ltcf_count += ho_count
                if patient_days[1][i] > 0:
                    ho_rates.append(ho_count / patient_days[1][i] * multiplier)
                    co_rates.append(co_count / patient_days[1][i] * multiplier)
                else:
                    ho_rates.append(0)
                    co_rates.append(0)

        # ----- Add UNC Overall
        names.append('UNC Overall')
        co_rates.append(unc_co / unc_patient_days * multiplier)
        ho_rates.append(unc_ho / unc_patient_days * multiplier)

        ho_onset = dict()
        co_onset = dict()
        for i, item in enumerate(names):
            ho_onset[item] = ho_rates[i]
            co_onset[item] = co_rates[i]
        # --- Add LTCF
        ho_onset['LTCF'] = ltcf_count / ltcf_patient_days * multiplier
        if unc_only:
            ho_onset['UNC_Regional'] = (ltcf_count + unc_ho) / (unc_patient_days + ltcf_patient_days) * multiplier
        return co_onset, ho_onset

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Emerging Infections Program (EIP) Definitions
    # ------------------------------------------------------------------------------------------------------------------
    def associated_cdi(self, skip_days=90):
        """ Calculate community associated (CA-CDI) and healthcare associated (HA-CDI)
        """

        total_population = self.available_population().mean()[0]
        cases = self.cdi_cases

        # ----- Skip first 70 days
        cases = cases[cases.Time > 90]
        cases = cases[cases.Type != 'Duplicate']
        multiplier = 100_000 * self.parameters.base['time_horizon'] / (self.parameters.base['time_horizon'] - skip_days)

        # Determine which are CA and which are HA
        ha_cdi, ca_cdi = 0, 0
        cdi_happened_at = []
        for case in cases.iterrows():
            # Find all events for the agent that were location related
            person_events = self.events[self.events['Unique_ID'] == case[1].Unique_ID]
            person_events = person_events[person_events['State'] == NameState.LOCATION]
            # If there are any location movements:
            if person_events.shape[0] > 0:
                # The only days we care about are 12 weeks in the past up until 3 days before event
                t = case[1].Time
                start = max((t - 12 * 7), 0)
                end = max((t - 3), 0)
                events = self.create_timeline_from_events(person_events)
                # Filter to only days of interest
                events = events[start:end]
                # If agent is not in the community (i.e. greater than 0) for any of these days, this is HA
                if events[events > 0].shape[0] > 0:
                    cdi_happened_at.append(events[events != 0].iloc[-1])
                    ha_cdi += 1
                else:
                    ca_cdi += 1
            else:
                ca_cdi += 1

        return (['Community Associated', 'Healthcare Associated'],
                [ca_cdi / total_population * multiplier, ha_cdi / total_population * multiplier])

    def cdi_deaths(self):
        """ Find the number of CDI deaths per 100,000 individuals

        Return
        ------
        Total deaths attributed to cdi
        """
        total_population = self.available_population().mean()[0]
        deaths = self.sum_events(state=NameState.CDIFF_RISK, new=[CDIState.DEAD]).sum()[0]

        return round(deaths / total_population * 100000, 4)

    def incidence(self, state, new, old=None, locations=None, risks=None, antibiotics=None, per=100_000,
                  by='Total', last_9=True):
        """ Calculate the incidence of an event

        Parameters
        ----------
        state : NameState
            The state of the event

        new, old : see filter_events()

        locations, risks, antibiotics : see available_population()

        per : integer
            The denominator in the calculation

        by : str
            'Total', 'Month', or 'Year'

        last_9 : boolean
            Only calculate incidence for the last 9 months and inflate the number to reflect a full year

        Return
        ------
        Incidence "by" total, month, or year

        """
        totals_df = self.available_population(locations, risks, antibiotics)
        events_df = self.sum_events(state=state, new=new, old=old, locations=locations)

        df = totals_df
        df.columns = ['totals_df']
        df['events_df'] = events_df
        df['events_df'] = df['events_df'].fillna(0)
        if last_9:
            df = df[91:]

        if by == 'Month':
            month_list = []
            events = df['events_df'].values
            for i in range(0, int(df.shape[0] / 30)):
                month_list.append(events[i * 30:(i + 1) * 30].sum())
            return month_list

        if by == 'Year':
            return df['events_df'].sum() / df['totals_df'].mean() * per * 365 / df.shape[0]

        if by == 'Total':
            return events_df.sum().values[0]

    def prevalence(self, locations=None, risks=None, antibiotics=None, by='Year', last_9=True, per=100_000):
        """ Calculate the prevalence of an event

        Parameters
        ----------
        locations, risks, antibiotics : see available_population()

        per : integer
            The denominator in the calculation

        by : str
            'Month' or 'Year'

        last_9 : boolean
            Only calculate incidence for the last 9 months and inflate the number to reflect a full year

        Return
        ------
        Prevalence "by" month, or year

        """

        pop_with_state = self.available_population(locations=locations, risks=risks, antibiotics=antibiotics)
        total_pop = self.available_population(locations=locations)

        prev = pop_with_state / total_pop
        if last_9:
            prev = prev[91:] * per
        else:
            prev = prev * per
            return prev.iloc[-1][0]

        if by == 'Month':
            month_list = []
            for i in range(int(prev.shape[0] / 30)):
                month_list.append(prev[i * 30:(i + 1) * 30].mean()[0])
            return month_list
        if by == 'Year':
            # Yearly Prevalence
            return prev.mean()[0]

    @staticmethod
    def create_timeline_from_events(events):
        """ Given a set of events, find where a patient was at each time period
        """
        df = Series(index=range(366))
        for event in events.reset_index().iterrows():
            if event[0] == 0:
                df.loc[0] = event[1].Old
                df.loc[event[1].Time] = event[1].New
            else:
                df.loc[event[1].Time] = event[1].New

        return df.fillna(method='ffill')

    # ------------------------------------------------------------------------------------------------------------------
    # ----- The following are all used for graphical purposes
    # ------------------------------------------------------------------------------------------------------------------
    def count_by_x(self, state, locations=None, risks=None, antibiotics=None, only_alive=True, risk_column='CDI'):
        dc = self.daily_counts

        if only_alive:
            dc = dc[dc['Life'] == 1]
        if isinstance(locations, list):
            dc = dc[dc['Location'].isin(locations)]
        if isinstance(risks, list):
            dc = dc[dc[risk_column].isin(risks)]
        if isinstance(antibiotics, list):
            dc = dc[dc['Antibiotics'].isin(antibiotics)]

        counts = DataFrame()
        for i in [c for c in dc.columns if c not in ['Antibiotics', 'Life', 'Location', risk_column]]:
            counts[i] = dc.groupby(by=[state])[i].sum()
        return counts

    def make_cdi_graph(self, filename='risk_graph.html', locations='', risks='', antibiotics=''):
        df = self.count_by_x('CDI', locations, risks, antibiotics)
        trace0 = go.Scatter(
            x=df.shape[1], y=df.sum(),
            fill='tonexty',
            name='Total Individuals'
        )
        data = [trace0]
        for item in [CDIState.SUSCEPTIBLE, CDIState.COLONIZED, CDIState.CDI]:
            trace = go.Scatter(
                x=df.shape[1], y=df.loc[item.value],
                fill='tozeroy',
                name=self.cdiff_map[item.name]
            )
            data.append(trace)

        layout = go.Layout(
            title="CDI Risk State by Day",
            xaxis=dict(title='Day'),
            yaxis=dict(title='Count')
        )

        if filename == "":
            return {'data': data,
                    'layout': layout}
        else:
            fig = go.Figure(data=data, layout=layout)
            plotly.offline.plot(fig, filename=str(self.output_dir) + '/' + filename, show_link=False)

    def make_antibiotic_graph(self, filename='antibiotic_graph.html',
                              location_ids='', risk_ids='', antibiotic_ids=''):
        df = self.count_by_x('Antibiotics', location_ids, risk_ids, antibiotic_ids)
        trace0 = go.Scatter(
            x=df.shape[1], y=df.sum(),
            fill='tonexty',
            name='Total Individuals'
        )
        trace1 = go.Scatter(
            x=df.shape[1], y=df.loc[0],
            fill='tozeroy',
            name='Off Antibiotics'
        )
        trace2 = go.Scatter(
            x=df.shape[1], y=df.loc[1],
            fill='tozeroy',
            name='On Antibiotics'
        )
        layout = go.Layout(
            title='Antibiotic State by Day',
            xaxis=dict(title='Day'),
            yaxis=dict(title='Count')
        )
        data = [trace0, trace1, trace2]

        if filename == "":
            return {'data': data,
                    'layout': layout}
        else:
            fig = go.Figure(data=data, layout=layout)
            plotly.offline.plot(fig, filename=str(self.output_dir) + '/' + filename, show_link=False)

    def make_location_graph(self, locations='', risks='', antibiotics='', filename='location_graph.html'):
        df = self.count_by_x('Location', locations, risks, antibiotics)
        data = list()
        # Remove Community
        df = df[df.index != self.locations.values['COMMUNITY']]
        df.index = [self.map[self.locations.ints[i]] for i in df.index]

        for i in df.index:
            data.append(
                go.Scatter(
                    x=df.shape[1], y=df.loc[i],
                    fill='tozeroy',
                    name=i
                )
            )
        layout = go.Layout(
            title="Location and Count by Day",
            xaxis=dict(title='Day'),
            yaxis=dict(title='Count')
        )
        if filename == "":
            return {'data': data,
                    'layout': layout}
        else:
            fig = go.Figure(data=data, layout=layout)
            plotly.offline.plot(fig, filename=str(self.output_dir) + '/' + filename, show_link=False)

    def make_location_graph_combined(self, filename='location_graph_combined.html'):
        df = self.count_by_x('Location')
        data = list()
        df.index = [self.map[self.locations.values(i)] for i in df.index]
        data.append(
            go.Scatter(
                x=df.shape[1], y=df.loc[self.map['ST']],
                fill='tozeroy',
                name='STACH-non UNC'
            )
        )
        data.append(
            go.Scatter(
                x=df.shape[1], y=df.loc[self.map['LT']],
                fill='tozeroy',
                name='LTACH'
            )
        )
        data.append(
            go.Scatter(
                x=df.shape[1], y=df.loc[self.map['NH']],
                fill='tozeroy',
                name='Nursing Home'
            )
        )
        data.append(
            go.Scatter(
                x=df.shape[1],
                y=df.loc[[self.map[self.locations.ints[v]] for v in self.locations.values if v.value not in
                          [0, 11, 12, 13]]].sum(),
                fill='tozeroy',
                name='STACH-UNC'
            )
        )
        layout = go.Layout(
            title="Location and Count by Day",
            xaxis=dict(title='Day'),
            yaxis=dict(title='Count')
        )
        if filename == "":
            return {'data': data,
                    'layout': layout}
        else:
            fig = go.Figure(data=data, layout=layout)
            plotly.offline.plot(fig, filename=str(self.output_dir) + '/' + filename, show_link=False)


def monthly_average(daily_totals):
    daily_totals = daily_totals.T
    month_list = []
    for i in range(0, 12):
        month_list.append(daily_totals[i * 30:(i + 1) * 30].mean().values[0])
    return month_list
