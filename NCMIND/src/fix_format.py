
import pandas as pd
import argparse
from pathlib import Path
import copy
import numpy as np
pd.set_option('mode.chained_assignment', None)


def clean_df(ddf, set_index=True):
    """ Correct demographic variables to be consistent. Fix column names, and if needed, rename hospitals accordingly

    Parameters
    ----------
    ddf : pandas DataFrame
        A raw DataFrame that needs cleaning
    set_index : bool
        Create a multiindex on the DataFrame, or not

    Returns
    -------
    daily_counts : DataFrame
        A DataFrame of the daily counts (by state) for all agents
    """

    sex_dictionary = {"M": 1, "F": 2}
    race_dictionary = {"White": 1, "Black": 2, "Other": 3}
    age_dictionary = {"L50": 0, "5064": 1, "G65": 2}

    ddf = ddf.replace({"Sex": sex_dictionary, "Race": race_dictionary, "Age Group": age_dictionary})

    # --- Rename columns, set index, drop columns
    ddf = ddf.rename(columns={'Age Group': 'Age'})
    # --- Rename hospitals
    ddf = ddf.rename(columns=names_dict)

    # ----- Set index and drop columns
    if set_index:
        index_ddf = ddf.set_index(['County_Code', 'Sex', 'Age', 'Race'])
        index_ddf = index_ddf.drop(['From', 'County'], axis=1)
        return index_ddf.sort_index()
    else:
        return ddf


def clean_community(data_dir='NCMIND/data'):
    """ Clean the community transition file

    Parameters
    ----------
    data_dir : str
        Directory containing raw data from the statistics team

    """

    # ----- Generate Directory
    Path.cwd().joinpath(data_dir, 'raw/location_transitions/').mkdir(exist_ok=True)
    scenario_dir = Path.cwd().joinpath(data_dir, 'raw/location_transitions/clean_location_transitions')
    scenario_dir.mkdir(exist_ok=True)

    # ----- Load data and remove duplicates
    df = pd.read_csv(Path(data_dir, 'raw/location_transitions/Community.csv'))
    df = df.drop_duplicates()
    index_df = clean_df(df)

    # ----- Update Probabilities
    final_data = []
    for co in index_df.index.get_level_values('County_Code').unique():
        for sex in index_df.index.get_level_values('Sex').unique():
            for age in index_df.index.get_level_values('Age').unique():
                for race in index_df.index.get_level_values('Race').unique():
                    d = index_df.loc[co, sex, age, race]

                    # ----- Convert probabilities
                    row = [co, sex, age, race]
                    # Probability of leaving home
                    prob_of_movement = 1 - d[d['To'] == 'Home']['Transition Probability'].values[0]
                    row.append(prob_of_movement)

                    if prob_of_movement == 0:
                        hp = [float(0)] * 13
                    else:
                        # Probability of UNC hospitals
                        hp = d[d['To'] == 'UNC'].values[0][1:]
                        if hp[0] != 0:
                            hp = hp[1:] * hp[0] / prob_of_movement
                            hp = hp.tolist()
                        else:
                            hp = [float(0)] * 10

                        # Add probability of Other Hospital
                        s_tach =\
                            d[d['To'] == 'Non-UNC']['Transition Probability'].values[0] / prob_of_movement
                        l_tach =\
                            d[d['To'] == 'LT']['Transition Probability'].values[0] / prob_of_movement
                        nh =\
                            d[d['To'] == 'NH']['Transition Probability'].values[0] / prob_of_movement

                        hp = hp + [s_tach] + [l_tach] + [nh]

                        if sum(hp) != 0:
                            hp = [float(i) / sum(hp) for i in list(hp)]

                    row = row + hp
                    final_data.append(row)

    fd = pd.DataFrame(final_data)
    fd.columns = ['County_Code', 'Sex', 'Age', 'Race', 'probability'] + names

    d = Path("raw/location_transitions/clean_location_transitions/")
    fd.to_csv(Path(data_dir, d, 'community_transitions.csv'), index=False)
    fd.to_csv(Path(data_dir, 'input/transitions/community_transitions.csv'), index=False)


# ----------------------------------------------------------------------------------------------------------------------
# ----- x to ... -------------------------------------------------------------------------------------------------------
def clean_x(data_dir='NCMIND/data', x="UNC"):
    """ Clean the UNC or Non-UNC Transition files

    Parameters
    ----------
    data_dir : str
        Directory containing raw data from the statistics team
    x : str
        One of 'UNC', 'Non-UNC
    """

    # ----- Load data and remove duplicates
    df = pd.read_csv(data_dir + '/raw/location_transitions/' + x + '.csv')
    df = df.drop_duplicates()

    index_df = clean_df(df)

    # ----- Update Probabilities
    final_data = []
    for co in index_df.index.get_level_values('County_Code').unique():
        for sex in index_df.index.get_level_values('Sex').unique():
            for age in index_df.index.get_level_values('Age').unique():
                for race in index_df.index.get_level_values('Race').unique():
                    d = index_df.loc[co, sex, age, race]

                    # ----- Convert Probabilities
                    row = [co, sex, age, race]
                    home = d[d['To'] == 'Home']['Transition Probability'].values[0]
                    if x == "UNC":
                        hospital_number = d.shape[1] - 2
                        unc = [d[d['To'] == 'UNC']['Transition Probability'][0] * 1/hospital_number] * hospital_number
                    else:
                        unc = d[d['To'] == 'UNC']['Transition Probability'][0] * d[d['To'] == 'UNC'].values[0][2:]
                    nh = d[d['To'] == 'NH']['Transition Probability'].values[0]
                    l_tach = d[d['To'] == 'LT']['Transition Probability'].values[0]
                    s_tach = d[d['To'] == 'Non-UNC']['Transition Probability'].values[0]

                    all_prob = [home] + list(unc) + [s_tach] + [l_tach] + [nh]

                    # --- Make sure this sums to 1
                    if round(sum(all_prob), 8) != 1:
                        if sum(all_prob) != 0:
                            all_prob = [float(i) / sum(all_prob) for i in list(all_prob)]

                    row = row + all_prob
                    final_data.append(row)

    df = pd.DataFrame(final_data)
    if x == 'NH':
        df = df.fillna(0)
    df.columns = ['County_Code', 'Sex', 'Age', 'Race'] + [community] + names

    df.to_csv(data_dir + '/raw/location_transitions/clean_location_transitions/' + x + '_transitions.csv', index=False)


def clean_unc_to_unc(data_dir='NCMIND/data'):
    """ Clean the UNC to UNC Hospital File

    Parameters
    ----------
    data_dir : str
        Directory containing raw data from the statistics team
    """

    # ----- Load data and remove duplicates
    df = pd.read_csv(Path(data_dir, 'raw/location_transitions/UNC-to-UNC.csv'))

    # --- Rename hospitals
    df = df.rename(columns=names_dict)

    df['Hospital'] = df['Hospital'].str.upper()
    df.loc[df['Hospital'] == 'HIGH POINT', 'Hospital'] = 'HIGHPOINT'
    df.loc[df['Hospital'] == 'UNC-CH', 'Hospital'] = 'UNC_CH'

    df = df.drop(['County'], axis=1)

    df.to_csv(Path(data_dir, "raw/location_transitions/clean_location_transitions/UNC-to-UNC.csv"), index=False)


def clean_initial_population(data_dir='NCMIND/data'):
    """ Clean the initial population files

    Parameters
    ----------
    data_dir : str
        Directory containing raw data from the statistics team
    """
    cc = pd.read_csv(Path(data_dir, '../../base_files/county_codes.csv'))

    # ----- UNC Hospitals  -----
    df = pd.read_csv(Path(data_dir, 'raw/initial_population/Initial_UNC_Population.csv'))
    df = df.drop_duplicates()
    df = clean_df(df, set_index=False)
    # --- Add County Code
    # co = df['County'].values
    # df.insert(loc=1, column='County_Code',
    #           value=[cc[cc['County'].isin([i])]['County_Code'].values[0] for i in co])
    df.to_csv(Path(data_dir, 'input/initial_population/UNC_initial.csv'), index=False)

    # ----- Non-UNC Hospitals -----
    df = pd.read_csv(Path(data_dir, 'raw/initial_population/Initial_Non-UNC_Population.csv'))
    df = df.drop_duplicates()
    df = clean_df(df, set_index=False)
    # --- Add County Code
    co = df['County'].values
    df.insert(loc=1, column='County_Code',
              value=[cc[cc['County'].isin([i])]['County_Code'].values[0] for i in co])
    df.to_csv(Path(data_dir, 'input/initial_population/Non-UNC_initial.csv'), index=False)

    # ----- NH ------
    df = pd.read_csv(Path(data_dir, 'raw/initial_population/Initial_NH_Population.csv'))
    df = df.drop_duplicates()
    df = clean_df(df, set_index=False)
    # ----- Save as CSV
    df.to_csv(Path(data_dir, 'input/initial_population/NH_initial.csv'), index=False)

    # ----- LT ------
    df = pd.read_csv(Path(data_dir, 'raw/initial_population/Initial_LT_Population.csv'))
    df = df.drop_duplicates()
    df = clean_df(df, set_index=False)
    # ----- Save as CSV
    df.to_csv(Path(data_dir, 'input/initial_population/LT_initial.csv'), index=False)

    # ----- Read in the initial population counts and filter it to only counties in the population
    # UNC
    unc_initial = pd.read_csv(Path(data_dir, 'input/initial_population/UNC_initial.csv'))
    # Non-UNC
    non_unc_initial = pd.read_csv(Path(data_dir, 'input/initial_population/Non-UNC_initial.csv'))
    # NH
    nh_initial = pd.read_csv(Path(data_dir, 'input/initial_population/NH_initial.csv'))
    # nh_initial.loc[nh_initial['Age'] < 2, 'Patients'] = 0
    # LT
    lt_initial = pd.read_csv(Path(data_dir, 'input/initial_population/LT_initial.csv'))

    # Merge Non-UNC and UNC
    initial = pd.merge(unc_initial, non_unc_initial[['County_Code', 'Age', 'Sex', 'Race', 'Patients']],
                       on=['County_Code', 'Age', 'Sex', 'Race'], how='outer')
    initial = initial.rename(columns={"Patients": "ST"})
    # Merge UNC and NH
    initial = pd.merge(initial, nh_initial[['County_Code', 'Age', 'Sex', 'Race', 'Patients']],
                       on=['County_Code', 'Age', 'Sex', 'Race'])
    initial = initial.rename(columns={"Patients": "NH"})
    # Merge UNC+NH with LT
    initial = pd.merge(initial, lt_initial[['County_Code', 'Age', 'Sex', 'Race', 'Patients']],
                       on=['County_Code', 'Age', 'Sex', 'Race'], how='outer')
    initial = initial.rename(columns={"Patients": "LT"})
    initial = initial.fillna(0)

    # ----- Save as CSV
    initial.to_csv(Path(data_dir, 'input/initial_population/all_initial.csv'), index=False)


def make_one_file(data_dir='NCMIND/data'):
    """ Combine all transition files into one single file

    Parameters
    ----------
    data_dir : str
        Directory containing raw data from the statistics team
    """

    d = Path("raw/location_transitions/clean_location_transitions/")

    # ----- COMMUNITY
    c = pd.read_csv(Path(data_dir, d, "community_transitions.csv"))
    one_file_community = c.drop('probability', axis=1)
    one_file_community.insert(4, 'COMMUNITY', 0)
    one_file_community.insert(0, 'From', 'COMMUNITY')

    # ----- Non-UNC
    one_file_non_unc = pd.read_csv(Path(data_dir, d, "Non-UNC_transitions.csv"))
    one_file_non_unc.insert(0, 'From', 'ST')

    # ----- LT
    one_file_lt = pd.read_csv(Path(data_dir, d, "LT_transitions.csv"))
    one_file_lt.insert(0, 'From', 'LT')

    # ----- NH
    one_file_nh = pd.read_csv(Path(data_dir, d, "NH_transitions.csv"))
    one_file_nh.insert(0, 'From', 'NH')

    # ----- UNC to UNC.
    unc = pd.read_csv(Path(data_dir, d, "UNC_transitions.csv"))
    one_file_unc = pd.DataFrame()
    hospital_names = [item for item in names_dict.values() if item != 'County_Code']
    for hospital in hospital_names:
        a = copy.copy(unc)
        a.insert(0, 'From', hospital)

        one_file_unc = one_file_unc.append(a)
    one_file_unc = one_file_unc.reset_index(drop=True)

    # ----- UNC to UNC File #2
    unc_10 = pd.read_csv(Path(data_dir, d, "UNC-to-UNC.csv"))

    for index, row in unc_10.iterrows():
        rows = one_file_unc[(one_file_unc['County_Code'] == row['County_Code']) & (
            one_file_unc['From'] == row['Hospital'])][hospital_names]
        total_p = rows.sum(axis=1).values[0]
        # If the row has values, update the UNC file
        if row[hospital_names].values.sum() > 0:
            one_file_unc.loc[rows.index, hospital_names] = row[hospital_names].values * total_p
        # If the row has no value (UNC to UNC has no data), update file to send people to the community.
        else:
            one_file_unc.loc[rows.index, hospital_names] = 0
            one_file_unc.loc[rows.index, 'COMMUNITY'] += total_p

    one_file = pd.DataFrame()
    one_file = one_file.append(one_file_community).reset_index(drop=True)
    one_file = one_file.append(one_file_unc).reset_index(drop=True)
    one_file = one_file.append(one_file_non_unc).reset_index(drop=True)
    one_file = one_file.append(one_file_lt).reset_index(drop=True)
    one_file = one_file.append(one_file_nh).reset_index(drop=True)
    one_file = one_file.rename(columns={"From": "Location"})

    # ----- Add the demographic index
    ct = one_file_community
    ct['Index'] = ct.index
    ct = ct.set_index(['County_Code', 'Sex', 'Age', 'Race'])
    id_list = [ct.loc[item.County_Code, item.Sex, item.Age, item.Race]['Index'] for index, item in one_file.iterrows()]
    one_file['Demographic_ID'] = id_list

    one_file.to_csv(Path(data_dir, "input/transitions/location_transitions.csv"), index=False)


def remove_hospitals(hospital_list, df):
    """ Removes list of hospitals from the location transition file. It will update probabilities accordingly

    Parameters
    ----------
    hospital_list : list
        List of hospital names to remove
    data_dir : str
        Directory containing raw data from the statistics team

    Return
    ------
    The location transition file with specific hospitals removed.
    """
    df = df.copy()
    hospital_names = [item for item in df.columns if item not in
                      ['Location', 'County_Code', 'Sex', 'Age', 'Race', 'Demographic_ID', 'COMMUNITY', 'LT', 'NH']]

    df['Sum_Old'] = df[[item for item in df.columns if item in hospital_names]].sum(axis=1)
    df = df[[item for item in df.columns if item not in hospital_list]]
    df['Sum_New'] = df[[item for item in df.columns if item in hospital_names]].sum(axis=1)
    df['Sum_Other'] = df[['COMMUNITY', 'LT', 'NH']].sum(axis=1)

    # If Other and New are both 0, ST needs to be 1
    df['ST'] = np.where((df['Sum_Other'] == 0) & (df['Sum_New'] == 0), 1, df['ST'])

    df['Sum_New'] = np.where((df['Sum_Other'] == 0) & (df['Sum_New'] == 0), 1, df['Sum_New'])
    df['Sum_Ratio'] = df['Sum_Old'] / df['Sum_New']

    for column in [item for item in df.columns if item in hospital_names]:
        df[column] = df[column] * df['Sum_Ratio']

    df = df.drop(['Sum_Old', 'Sum_New', 'Sum_Other', 'Sum_Ratio'], axis=1)

    # Remove the rows mentioning the hospital
    # df = df.drop(df[df.Location.isin(hospital_list)].index)

    return df


# ----- What to run when the script is called ----- #
if __name__ == '__main__':

    # These values are hardcoded for our project.
    names_dict = {
        'County Code': 'County_Code',
        'Caldwell': 'CALDWELL',
        'Chatham': 'CHATHAM',
        'High Point': 'HIGHPOINT',
        'Johnston': 'JOHNSTON',
        'Lenoir': 'LENOIR',
        'Margaret': 'MARGARET',
        'Nash': 'NASH',
        'Rex': 'REX',
        'UNC-CH': 'UNC_CH',
        'Wayne': 'WAYNE'
    }

    names = [item for item in names_dict.values() if item != 'County_Code'] + ['ST', 'LT', 'NH']
    community = 'COMMUNITY'

    parser = argparse.ArgumentParser(description='None')

    parser.add_argument(
        'experiment_dir',
        help='directory containing the experiment'
    )
    args = parser.parse_args()

    clean_community(args.experiment_dir)
    clean_x(args.experiment_dir, x='UNC')
    clean_x(args.experiment_dir, x='Non-UNC')
    clean_x(args.experiment_dir, x='LT')
    clean_x(args.experiment_dir, x='NH')
    clean_unc_to_unc(args.experiment_dir)
    clean_initial_population(args.experiment_dir)

    make_one_file(args.experiment_dir)
