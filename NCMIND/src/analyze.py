

import pandas as pd
import plotly
import plotly.graph_objs as go

from src.state import NameState, CDIState, LifeState
from src.parameters import Parameters
from src.north_carolina import NcLocations
from src.misc_functions import int_to_category
from pathlib import Path


class Analyze:
    """ Provides common analysis routines for the NCMIND model output data.
    """

    def __init__(self, experiment: str, scenario: str, run: str):

        self.cdi_map = {
            'SUSCEPTIBLE': 'Susceptible',
            'COLONIZED': 'Asymptomatically Colonized',
            'CDI': 'CDI',
            'DEAD': 'Dead'
        }

        self.experiment_dir = Path(experiment)
        self.output_dir = Path(experiment, scenario, run, 'model_output')
        self.params = Parameters(Path(experiment, scenario, run, "parameters.json"))
        self.daily_counts = pd.read_csv(Path(self.output_dir, 'daily_counts.csv'))
        self.locations = NcLocations(self.experiment_dir)

        if 'cdi' == self.params.base['disease_model']:
            self.cdi_cases = pd.read_csv(Path(self.output_dir, 'CDI_cases.csv'))
        if 'cre' == self.params.base['disease_model']:
            self.cre_cases = pd.read_csv(Path(self.output_dir, 'CRE_cases.csv'))

        # ---- Some files may be compressed
        try:
            self.events = pd.read_csv(Path(self.output_dir, 'model_events.csv'))
        except UnicodeDecodeError:
            self.events = pd.read_csv(Path(self.output_dir, 'model_events.csv'), compression='gzip')

        self.catchment_counties =\
            [1, 21, 23, 27, 35, 37, 49, 51, 57, 61, 63, 65, 67, 69, 79, 81, 83, 85, 89, 101, 103, 105, 107, 125, 127,
             129, 133, 135, 145, 147, 149, 151, 155, 161, 163, 175, 183, 189, 191, 193, 195]

    def available_population(
            self,
            locations: list = None,
            disease_states: list = None,
            antibiotics: list = None,
            disease_state_column: str = 'CDI') -> pd.DataFrame:
        """ Count the alive population at each day, filtered by the parameters
        """
        # ----- Find Population Total
        dc = self.daily_counts
        # ----- Filter to Specific Locations, disease_states, or antibiotic use
        if locations:
            dc = dc[dc['Location'].isin(locations)]
        if disease_states:
            dc = dc[dc[disease_state_column].isin(disease_states)]
        if antibiotics:
            dc = dc[dc['Antibiotics'].isin(antibiotics)]
        # ----- Filter to integer columns
        dc = dc.loc[:, [c for c in dc.columns if c not in ['Antibiotics', 'Life', 'Location', disease_state_column]]]

        return pd.DataFrame(dc.sum().values)

    def filter_events(
            self,
            state: NameState,
            new: list = None,
            old: list = None,
            locations: list = None) -> pd.DataFrame:
        """ Filter model events to those that match the parameters

        Parameters
        ----------
        state : specify the state of interest
        new : list of values the New column can equal
        old : list of values the Old state can equal
        locations : list of locations the agent can be when an event happens
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

    def sum_events(
            self,
            state: NameState,
            new: list = None,
            old: list = None,
            locations: list = None) -> pd.DataFrame:
        """ Sum the total number of events that occurred, by day
        """

        events = self.filter_events(
            state=state,
            new=new,
            old=old,
            locations=locations
        )
        return pd.DataFrame(events.groupby(by='Time').size())

    def calculate_patient_days(
            self,
            unc_only: bool = False,
            grouped: bool = True) -> pd.DataFrame:
        """ Calculate the total number of patient days for each facility

        Parameters
        ----------
        unc_only : Filter to only patients from UNC catchment area
        """
        # ----- For patients who have already left
        location_events = self.events[self.events['State'] == NameState.LOCATION].copy()
        location_events['Old'] = int_to_category(self.locations, location_events['Old'])
        location_events['New'] = int_to_category(self.locations, location_events['New'])

        if unc_only:
            location_events = location_events[location_events.County.isin(self.catchment_counties)]

        # ----- For patients who have not yet left
        last_events = location_events.groupby(by=['Unique_ID']).last()

        patient_days_list = []
        names = []
        for name in location_events.Old.unique():
            names.append(name)
            # --- All the events where the patient left a location
            facility_events = location_events[location_events['Old'] == name]
            # --- All the places where a patient is still at the location
            last_events = last_events[last_events['New'] == name]
            last_events['Total_Days'] = self.params.base['time_horizon'] - last_events.Time
            patient_days_list.append(facility_events.LOS.sum() + last_events.Total_Days.sum())

        df = pd.DataFrame([names, patient_days_list]).T
        df.columns = ['Location', 'Count']
        for item in self.locations.categories_list:
            if item not in df.Location.values:
                df.loc[df.shape[0] + 1] = [item, 1]

        return df

    # ------------------------------------------------------------------------------------------------------------------
    # ----- National Healthcare Safety Network (NHSN Definitions)
    # ------------------------------------------------------------------------------------------------------------------
    def onset_cdi(
            self,
            skip_days: int = 90,
            unc_only: bool = False) -> (list, list):
        """ Calculate onset CDI for both the hospital and the inpatient community level

        Parameters
        ----------
        skip_days : Filter to only CDI cases that happened after X days
        unc_only : Filter to only patients from UNC catchment area
        """

        # ----- Incident CDI Cases after X days (skip_days)
        cases = self.cdi_cases[self.cdi_cases.Time > skip_days]
        cases = cases[cases.Type != 'Duplicate']
        cases = cases[cases['NHSN'] != 'Community']
        cases['Category'] = int_to_category(self.locations, cases.Location)

        if unc_only:
            cases = cases[cases.County.isin(self.catchment_counties)]

        # ----- Create the 13 HO-CDI and the 13 CO-CDI Numbers
        multiplier = 10_000 * self.params.base['time_horizon'] / (self.params.base['time_horizon'] - skip_days)
        patient_days = self.calculate_patient_days(
            unc_only=unc_only,
            grouped=True
        )
        patient_days = patient_days.set_index('Location')
        names, ho_rates, co_rates = list(), list(), list()

        for name in list(patient_days.index.values):
            if name != 'COMMUNITY':
                names.append(name)
                ho_count = cases[(cases.Category == name) & (cases['NHSN'] == 'HO CDI')].shape[0]
                co_count = cases[(cases.Category == name) & (cases['NHSN'] != 'HO CDI')].shape[0]
                days = patient_days.loc[name].Count
                if days > 0:
                    ho_rates.append(ho_count / days * multiplier)
                    co_rates.append(co_count / days * multiplier)
                else:
                    ho_rates.append(0)
                    co_rates.append(0)

        # --- Calculate LTCF as well
        ltcf_ho_count = cases[(cases.Category.isin(['LT', 'NH'])) & (cases['NHSN'] == 'HO CDI')].shape[0]
        ltcf_co_count = cases[(cases.Category.isin(['LT', 'NH'])) & (cases['NHSN'] != 'HO CDI')].shape[0]

        ltcf_patient_days = patient_days.loc["NH"].Count + patient_days.loc["LT"].Count
        names.append("LTCF")
        ho_rates.append(ltcf_ho_count / ltcf_patient_days * multiplier)
        co_rates.append(ltcf_co_count / ltcf_patient_days * multiplier)

        # --- Create the output
        ho_onset = dict()
        co_onset = dict()
        for i, item in enumerate(names):
            ho_onset[item] = ho_rates[i]
            co_onset[item] = co_rates[i]
        if unc_only:
            unc_ho = cases[(cases.Category == 'UNC') & (cases['NHSN'] == 'HO CDI')].shape[0]
            unc_patient_days = patient_days.loc["UNC"].Count
            ho_onset['Catchment HO'] = (ltcf_ho_count + unc_ho) / (unc_patient_days + ltcf_patient_days) * multiplier

        return co_onset, ho_onset

    # ------------------------------------------------------------------------------------------------------------------
    # ----- Emerging Infections Program (EIP) Definitions
    # ------------------------------------------------------------------------------------------------------------------
    def associated_cdi(
            self,
            skip_days: int = 90,
            unc_only: bool = False) -> (list, list):
        """ Calculate community associated (CA-CDI) and healthcare associated (HA-CDI)
        """

        population = self.available_population().mean()[0]
        cases = self.cdi_cases

        # ----- Skip first X days
        cases = cases[cases.Time > skip_days]
        cases = cases[cases.Type != 'Duplicate']
        multiplier = 100_000 * self.params.base['time_horizon'] / (self.params.base['time_horizon'] - skip_days)

        if unc_only:
            cases = cases[cases.County.isin(self.catchment_counties)]
            population = population * .6209  # Number obtained from Six by Six file with county populations

        ca_cdi = cases[cases.Association == 'CA-CDI'].shape[0]
        ha_cdi = cases[cases.Association == 'HA-CDI'].shape[0]

        names = ['Community Associated', 'Healthcare Associated']
        return (names, [ca_cdi / population * multiplier, ha_cdi / population * multiplier])

    def cdi_deaths(self) -> float:
        """ Find the number of CDI deaths per 100,000 individuals

        Return
        ------
        Total deaths attributed to cdi
        """
        total_population = self.available_population().mean()[0]
        deaths = self.sum_events(
            state=NameState.CDIFF_RISK,
            new=[CDIState.DEAD]
        ).sum()[0]

        return round(deaths / total_population * 100_000, 4)

    def incidence(
            self,
            state: NameState,
            new: list,
            old: list,
            locations: list = None,
            disease_states: list = None,
            antibiotics: list = None,
            per: int = 100_000,
            by: str = 'Total',
            skip_days: int = 90) -> float:
        """ Calculate the incidence of an event

        Parameters
        ----------
        state : The state of the event
        new, old : see filter_events()
        locations, disease_states, antibiotics : see available_population()
        per : The denominator in the calculation
        by : 'Total', 'Month', or 'Year'
        skip_days : number of initial days to skip

        Return
        ------
        Incidence "by" total, month, or year
        """
        totals_df = self.available_population(
            locations=locations,
            disease_states=disease_states,
            antibiotics=antibiotics
        )
        events_df = self.sum_events(
            state=state,
            new=new,
            old=old,
            locations=locations
        )

        df = totals_df
        df.columns = ['totals_df']
        df['events_df'] = events_df
        df['events_df'] = df['events_df'].fillna(0)
        df = df[skip_days:]

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

    def prevalence(
        self,
        locations: list,
        disease_states: list,
        antibiotics: list,
        by: str,
        skip_days: int = 90,
        per: int = 100_000) -> float:
        """ Calculate the prevalence of an event

        Parameters
        ----------
        locations, disease_states, antibiotics : see available_population()
        per : The denominator in the calculation
        by : 'Month' or 'Year'

        Return
        ------
        Prevalence "by" month, or year
        """

        pop_with_state = self.available_population(
            locations=locations,
            disease_states=disease_states,
            antibiotics=antibiotics
        )
        total_pop = self.available_population(locations=locations)

        prev = pop_with_state / total_pop
        prev = prev[skip_days:] * per

        if by == 'Month':
            month_list = []
            for i in range(int(prev.shape[0] / 30)):
                month_list.append(prev[i * 30:(i + 1) * 30].mean()[0])
            return month_list
        if by == 'Year':
            # Yearly Prevalence
            return prev.mean()[0]

    @staticmethod
    def create_timeline_from_events(events) -> pd.DataFrame:
        """ Given a set of events, find where a patient was at each time period
        """
        df = pd.Series(index=range(366))
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
    def count_by_x(
        self,
        state: str,
        locations: list = None,
        disease_states: list = None,
        antibiotics: list = None,
        risk_column: str = 'CDI') -> list:

        dc = self.daily_counts
        if NameState.LIFE.name in dc.columns:
            dc = dc[dc[NameState.LIFE.name] == LifeState.ALIVE.value]
        if locations:
            dc = dc[dc[NameState.LOCATION.name].isin(locations)]
        if disease_states:
            dc = dc[dc[risk_column].isin(disease_states)]
        if antibiotics:
            dc = dc[dc[NameState.ANTIBIOTICS.name].isin(antibiotics)]

        counts = pd.DataFrame()
        for i in [c for c in dc.columns if c not in [item.name for item in NameState]]:
            counts[i] = dc.groupby(by=[state])[i].sum()
        return counts

    def make_cdi_graph(
        self,
        filename: str,
        locations: list = None,
        disease_states: list = None,
        antibiotics: list = None) -> plotly.offline.plot:

        df = self.count_by_x('CDI', locations, disease_states, antibiotics)
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
                name=self.cdi_map[item.name]
            )
            data.append(trace)

        layout = go.Layout(
            title="CDI Risk State by Day",
            xaxis=dict(title='Day'),
            yaxis=dict(title='Count')
        )

        if filename == "":
            return {'data': data, 'layout': layout}
        else:
            fig = go.Figure(data=data, layout=layout)
            plotly.offline.plot(fig, filename=str(self.output_dir) + '/' + filename, show_link=False)

    def make_antibiotic_graph(
            self,
            filename: str,
            locations: list = None,
            disease_states: list = None,
            antibiotics: list = None):
        df = self.count_by_x(
            state=NameState.ANTIBIOTICS.name,
            locations=locations,
            disease_states=disease_states,
            antibiotics=antibiotics
        )
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
            return {'data': data, 'layout': layout}
        else:
            fig = go.Figure(data=data, layout=layout)
            plotly.offline.plot(fig, filename=str(self.output_dir) + '/' + filename, show_link=False)

    def make_location_graph(
            self,
            filename: str,
            locations: list = None,
            disease_states: list = None,
            antibiotics: list = None):
        df = self.count_by_x(
            state=NameState.LOCATION.name,
            locations=locations,
            disease_states=disease_states,
            antibiotics=antibiotics
        )
        data = list()
        # Remove Community
        df = df[df.index != self.locations.facilities['COMMUNITY']]
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
            return {'data': data, 'layout': layout}
        else:
            fig = go.Figure(data=data, layout=layout)
            plotly.offline.plot(fig, filename=str(self.output_dir) + '/' + filename, show_link=False)

    def make_location_graph_combined(self, filename='location_graph_combined.html'):
        df = self.count_by_x(
            state=NameState.LOCATION.name
        )
        data = list()
        df.index = [self.map[self.locations.facilities(i)] for i in df.index]
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
                y=df.loc[[self.map[self.locations.ints[v]] for v in self.locations.facilities if v.value not in
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
