
import pandas as pd
import argparse as arg
import numpy as np
import re

from src.location import Locations
from src.misc_functions import create_cdf, random_selection


def rule_of_3(df: pd.DataFrame) -> pd.DataFrame:
    """
    The rule of 3 states that if a demographic is missing from our population, adding "3" individuals of that
    demographic will have a minimal impact on our results
    """
    x = 3
    gb = df.groupby(['County_Code', 'Age'])
    gb = pd.DataFrame(gb.size())
    gb = gb.reset_index()

    # ----- Go through each combination of county and age. Create a list containing the demographic
    # combination and the number of agents that are needed.
    agents_needed = []
    for county in df.County_Code.unique():
        for age_group in df.Age.unique():
            temp_gb =\
                gb[(gb.County_Code == county) & (gb.Age == age_group)]
            if temp_gb.shape[0] == 0:
                agents_needed.append([county, age_group, ])
            elif temp_gb[0].values[0] < x:
                agents_needed.append([county, age_group, x - temp_gb[0].values[0]])

    # ----- Add agents to the synthetic population based on which demographics were missing
    new_df = pd.DataFrame()
    print("The rule of 3 will now alter: {} demographic combinations".format(len(agents_needed)))
    for row in agents_needed:
        # ----- row[4] contains the number of agents to append.
        for i in range(row[4]):
            # ----- Assign a random age in years
            if row[2] == 0:
                age_years = 25
            elif row[2] == 1:
                age_years = 55
            else:
                age_years = 70
            # --- put the row together
            values = row[0:4] + [age_years] + row[5:]
            values = pd.DataFrame(values).T
            values.columns = df.columns
            new_df = new_df.append(values, ignore_index=True)

    return df.append(new_df, ignore_index=True)


def select_facility(hospital: dict, discharge_data: pd.DataFrame, county_age_dict: dict, fill_percent=.7):
    """ Append the number of hospital patients required for a given county/age dictionary.
        This will help fill the beds with agents
    """
    hospital_id = hospital['ID']
    rows = discharge_data[discharge_data[hospital_id] > 0].copy()
    rows.loc[:, 'Percentage'] = rows[hospital_id] / rows[hospital_id].sum()

    for bed in range(0, int(hospital['beds'] * fill_percent)):
        p = create_cdf(rows.Percentage.values)
        # --- Randomly select a county based on the percentage
        county = random_selection(np.random.rand(1, 1)[0], p, rows.County_Code.values)
        # --- Randomly select an age based on 40% <50, 20% 50-65, and 40% 65+
        age = random_selection(np.random.rand(1, 1)[0], [.4, .6, 1.0], np.array([0, 1, 2]))
        county_age_dict[(county, age)].append(hospital['int'])


def initialize_syn_pop(df: pd.DataFrame) -> pd.DataFrame:
    """ Initialize the synthetic population with starting locations
    """
    print("Step 1/5: Setting Indices")
    df['p_id'] = range(df.shape[0])
    start_locations = np.zeros(len(df))
    population = df.set_index(['County_Code', 'Age']).sort_index()
    locations = Locations("NCMIND")

    # ----- Assigning STACH Agents -------------------------------------------------------------------------------------
    print("Step 2/5: Reading Discharge data")
    large_dis = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name="Large-D")
    small_dis = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name="Small-D")
    unc_dis = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name="unc-D")

    print("Step 3/5: Assigning Initial Agents for STACHs")
    county_age = dict()
    for county in df.County_Code.unique():
        for age in df.Age.unique():
            county_age[(county, age)] = []

    # ----- Large Hospitals
    for large_int in locations.large_ints:
        h = locations.ints[large_int]
        select_facility(h, large_dis, county_age)
    # ----- Small Hospitals
    for small_int in locations.small_ints:
        h = locations.ints[small_int]
        select_facility(h, small_dis, county_age)
    # ----- UNC Hospitals
    for unc_int in locations.unc_ints:
        h = locations.ints[unc_int]
        select_facility(h, unc_dis, county_age)

    # ----- Assign starting locations for all STACH beds, based on county and age.
    for key, value in county_age.items():
        pop = population.loc[key]
        ids = pop.sample(int(min(pop.shape[0], len(value)))).p_id
        for _, x_id in enumerate(ids):
            start_locations[x_id] = value[_]

    # ----- Switch the index to logrec
    population = (
        population
        .reset_index()
        .set_index('logrecno')
        .sort_index()
    )
    # ----- Assigning Nursing Home and LTACH Agents --------------------------------------------------------------------
    # ----- Logrec to NH/LT Pairings
    logrecs = pd.read_csv("NCMIND/data/input/logrecnos.csv")
    logrecs.index = logrecs.logrecno

    # ----- Loop through each facility and assign logrecs to NH. Assume 70% capacity for all
    print("Step 4/5: Assigning bed counts to logrecnos for NH/LTs")
    for _, item in locations.ints.items():
        # ----- LTACHs
        if item['category'] == 'LT':
            use_logrecs = list(logrecs[logrecs.LT == item['ID']].logrecno.values)
            new_column = pd.Series(np.random.choice(use_logrecs, round(item['beds'] * .7))).value_counts()
            logrecs[item['ID']] = new_column
        # ----- NHs
        if item['category'] == 'NH':
            use_logrecs = list(logrecs[logrecs.NH == item['ID']].logrecno.values)
            if len(use_logrecs) > 0:
                new_column = pd.Series(np.random.choice(use_logrecs, round(item['beds'] * .7))).value_counts()
                logrecs[item['ID']] = new_column
            else:
                print("There are no geographys for NH: {}".format(item['ID']))

    logrecs = logrecs[[item for item in logrecs.columns if len(re.findall(r"_\d", item)) > 0]]
    logrecs = logrecs.fillna(0)

    # ----- Loop through each tract and randomly select people for the NH or LTACH
    count = 0
    print("Step 5/5: Randomly assigning agents from each logrec to NHs/LTACHs.")
    for _, logrec in logrecs.iterrows():
        count = count + 1
        if count % 500 == 0:
            print("{:.0%} done with logrec assignments.".format(count / logrecs.shape[0]))
        logrec = logrec[logrec > 0]

        if len(logrec) > 0:
            pop = population.loc[logrec.name]
            for item in logrec.index:
                if 'NH' in item:
                    temp_pop = pop[(pop.Age == 2) & (start_locations[pop.p_id] == 0)]
                else:
                    temp_pop = pop[start_locations[pop.p_id] == 0]
                if len(temp_pop) > 0:
                    ids = temp_pop.sample(int(min(temp_pop.shape[0], logrec[item]))).p_id
                    for x_id in ids:
                        start_locations[x_id] = locations.values[item]['int']
                else:
                    print('Tried to select patients for logrec {} & facility {}, but none were found.'.format(
                        logrec.name, item))

    df['Start_Location'] = start_locations
    return df.drop(['p_id'], axis='columns')


def extract_syn_pop():
    """ Extract the necessary columns from the 2017 synthetic population to use for NCMiND
    """

    # ----- Read the 2017 Synthetic Persons and Households files
    df = pd.read_csv('NCMIND/data/synthetic_population/37/NC2017_Persons.csv',
                    usecols=['hh_id', 'agep', 'sex', 'rac1p'])
    df_household = pd.read_csv('NCMIND/data/synthetic_population/37/NC2017_Households.csv',
                               usecols=['hh_id', 'logrecno', 'county', 'tract', 'blkgrp'])
    df = df.merge(df_household)

    df = df.rename(columns={'agep': 'Age', 'sex': 'Sex', 'rac1p': 'Race', 'county': 'County_Code'})

    # ----- Correct the Age
    df['Age_Years'] = df['Age']
    df['Age'] = -1
    df.loc[df['Age_Years'] < 50, 'Age'] = 0
    df.loc[df['Age_Years'] > 64, 'Age'] = 2
    df.loc[df['Age'] == -1, 'Age'] = 1
    # --- Correct the Race
    df.loc[df['Race'] > 2, 'Race'] = 3

    df = df[['County_Code', 'Sex', 'Age', 'Race', 'Age_Years', 'tract', 'blkgrp', 'logrecno']]

    # ----- Make sure every demographic has at least 3 agents.
    df = rule_of_3(df)

    # ----- Pad the population to 10m people. The approximate population of NC
    number_to_add = 10_042_802 - df.shape[0]
    new_people = df.sample(number_to_add)
    df = df.append(new_people)
    df = df.reset_index(drop=True)

    # ----- Initialize the population with a starting location
    df = initialize_syn_pop(df)
    # --- Save the full population
    df.to_csv("NCMIND/data/synthetic_population/synthetic_population.csv", index=False)
    # --- Now limit to orange county, and create another file
    df_orange = df[df['County_Code'] == 135]
    df_orange.to_csv("NCMIND/data/synthetic_population/synthetic_population_orange.csv", index=False)

    # ----- Aggregate by County and Age
    (df
        .groupby(by=['County_Code', 'Age'])
        .size()
        .reset_index()
        .to_csv("NCMIND/data/temp/population_by_county_age.csv", index=False))
    # ----- Aggregate by County
    (df
        .groupby(by=['County_Code'])
        .size()
        .reset_index()
        .to_csv("NCMIND/data/temp/population_by_county.csv", index=False))


if __name__ == '__main__':
    parser = arg.ArgumentParser(description='None')

    # Set the random seed
    parser.add_argument(
        '--seed',
        type=str,
        default=1111,
        help='Seed to use for the model'
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    extract_syn_pop()
