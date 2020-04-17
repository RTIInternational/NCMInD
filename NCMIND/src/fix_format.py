
import pandas as pd

from src.misc_functions import age_dictionary


def clean_df(temp_df: pd.DataFrame) -> pd.DataFrame:
    """ Correct demographic variables to be consistent. Fix column names, and if needed, rename hospitals accordingly
    Parameters
    ----------
    temp_df : pandas DataFrame
        A raw DataFrame that needs cleaning
    Returns
    -------
    daily_counts : DataFrame
        A DataFrame of the daily counts (by state) for all agents
    """

    temp_df = temp_df.replace({"Age_Group": age_dictionary})

    return temp_df.rename(columns={'Age_Group': 'Age'})


def make_community_transitions():
    """ Clean the community transition file
    """
    temp_df = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name='Community')
    index_temp_df = clean_df(temp_df)
    community = index_temp_df[['County_Code', 'Age', 'Probability', 'COMMUNITY', 'UNC', 'LARGE', 'SMALL', 'LT', 'NH']]
    # Normalize
    cols = ['COMMUNITY', 'UNC', 'LARGE', 'SMALL', 'LT', 'NH']
    temp = community[cols].sum(axis=1)
    for column in cols:
        community[column] = community[column] / temp
 
    community.to_csv("NCMIND/data/input/community_transitions.csv", index=False)


def make_location_transitions():
    """ Clean and combine all of the location transitions
    """
    proportions = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name="Proportions")
    to_unc = proportions[proportions.Parameter == 'non-UNC to UNC inflated'].Proportion.values[0]
    unc_to_unc = proportions[proportions.Parameter == '% UNC that move to UNC'].Proportion.values[0]
    large_to_large = proportions[proportions.Parameter == 'Large to Large'].Proportion.values[0]
    small_to_large = proportions[proportions.Parameter == 'Small to Large'].Proportion.values[0]
    nh_st_nh = proportions[proportions.Parameter == 'NH to ST to NH'].Proportion.values[0]

    # ----- Large
    large = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name="Large")
    large = large.drop(['Total'], axis='columns')
    large = large.set_index('Category').T
    l_nh = proportions[proportions.Parameter == 'NH to Large ST'].Proportion.values[0]
    large = calculate_hospital_movement(large, to_unc=to_unc, to_large=large_to_large, from_nh=l_nh, nh_st_nh=nh_st_nh)
    large = clean_df(large)

    # ----- Small
    s = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name="Small")
    s = s.drop(['Total'], axis='columns')
    s = s.set_index('Category').T
    s_nh = proportions[proportions.Parameter == 'NH to Small ST'].Proportion.values[0]
    small = calculate_hospital_movement(s, to_unc=to_unc, to_large=small_to_large, from_nh=s_nh, nh_st_nh=nh_st_nh)
    small = clean_df(small)

    # ----- UNC
    unc_df = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name="unc-Data")
    unc_df = unc_df.drop(['Total'], axis='columns')
    unc_df = unc_df.set_index('Category').T
    unc_nh = proportions[proportions.Parameter == 'NH to UNC'].Proportion.values[0]
    unc = calculate_hospital_movement(unc_df, to_unc=unc_to_unc, to_large=small_to_large,\
        from_nh=unc_nh, nh_st_nh=nh_st_nh)
    unc = clean_df(unc)

    temp_df = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name='NH')
    index_temp_df = clean_df(temp_df)
    nh = index_temp_df[unc.columns]

    temp_df = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name='LT')
    index_temp_df = clean_df(temp_df)
    lt = index_temp_df[unc.columns]

    temp_df = pd.read_csv("NCMIND/data/input/community_transitions.csv")
    temp_df['Facility'] = "COMMUNITY"
    community = temp_df[unc.columns]

    location_transitions = pd.concat([unc, large, small, nh, lt, community])

    location_transitions.to_csv("NCMIND/data/input/location_transitions.csv", index=False)


def make_unc_to_unc_discharges():
    """ Clean the UNC to UNC Hospital File
    """
    # ----- Load data and remove extra columns
    temp_df = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name='unc-D')
    temp_df = temp_df.drop(['Total', 'County'], axis=1)
    columns = [item for item in temp_df.columns if 'UNC' in item]
    column_sum = temp_df[columns].sum(axis=1)
    for column in columns:
        temp_df[column] = temp_df[column] / column_sum
    temp_df = temp_df.fillna(0)
    temp_df.to_csv("NCMIND/data/input/unc_to_unc_transitions.csv", index=False)


def make_nonUNC_discharges():
    """ Clean both the large, and small nonUNC files
    """
    def clean_df2(temp_df):
        temp_df = temp_df.drop(['County', 'Total'], axis=1)
        columns = [item for item in temp_df.columns if 'nonUNC' in item]
        column_sum = temp_df[columns].sum(axis=1)
        for column in columns:
            temp_df[column] = temp_df[column] / column_sum
        return temp_df.fillna(0)

    temp_df = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name='Large-D')
    clean_df2(temp_df).to_csv("NCMIND/data/input/large_discharge_transitions.csv", index=False)
    temp_df = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name='Small-D')
    clean_df2(temp_df).to_csv("NCMIND/data/input/small_discharge_transitions.csv", index=False)


def calculate_hospital_movement(df: pd.DataFrame, to_unc: float, to_large: float, from_nh: float,
    nh_st_nh: float = .8) -> pd.DataFrame:
    """ Using individual hospital level data, calculate the transition probabilities

    Parameters
    ----------
    to_unc : proportion of patients who go to a UNC hospital from this facility
    to_large : proportion of STACH discharges that go to a Large hospital, as opposed to a small one
    from_nh : the number of patients who go to this type of facility each year from a NH
    nh_st_nh : Proportion of patients who go to an STACH from a NH who return to the NH
    """
    discharges = pd.read_excel("NCMIND/data/six_by_six.xlsx", sheet_name="unc-D")

    community_columns = [
        'Patient Disposition Home, self, or outpatient care',
        'Patient Disposition Discharged, transferred to psychiatric facility', 'Patient Disposition Hospice',
        'Patient Disposition Left against medical advice', 'Patient Disposition Court/Law Enforcement',
        'Patient Disposition Expired', 'Patient Disposition Other/Unknown']
    locations = ['Community', 'ST', 'LT', 'NH']
    ages = ['L50', '50-64', '65+']

    # ----- Calculate age groups
    df['L50'] = df['Age Group Less than 1 Year'] + df['Age Group 1 - 17 years'] + df['Age Group 18 - 44 years'] +\
        df['Age Group 45 - 64 years'] * .25
    df['50-64'] = df['Age Group 45 - 64 years'] * .75
    df['65+'] = df['Age Group 65 - 84 years'] + df['Age Group 85 or more years']
    df['L50%'] = df['L50'] / (df['L50'] + df['50-64'] + df['65+'])
    df['50-64%'] = df['50-64'] / (df['L50'] + df['50-64'] + df['65+'])
    df['65+%'] = df['65+'] / (df['L50'] + df['50-64'] + df['65+'])

    # ----- Group Dispositions
    df['Community'] = df[community_columns].sum(axis=1)
    df['ST'] = df['Patient Disposition Discharged, transferred to acute facility'] +\
        df['Patient Disposition Discharged, transferred']
    df['LT'] = df['Patient Disposition Discharged, transferred to long term acute care facility (LTAC)']
    df['NH'] = df['Patient Disposition Discharged, transferred to facility that provides nursing, custodial, or supportive care']

    # ----- In the model, 80% of NH to ST patients are forced to return to the NH. We correct this here:
    # TODO: .7 is used to reduce this further. We still don't have enough NH to hospital movement
    df['nh_to_hospital'] = [item / df.NH.sum() * from_nh * .7 for item in df.NH]
    # --- Remove people going to the NH (because the model already forces people)
    df['NH'] = df.apply(lambda x: max(0, x.NH - x.nh_to_hospital * nh_st_nh), axis=1)

    # ----- Calculate % Movement by Age (note NH cannot be <65 years old)
    for location in locations:
        for age in ages:
            # We need more older folks to go to LTs (so we can have more NH movement)
            if location == 'LT':
                if age in ['L50', '50-64']:
                    df[location + "_" + age] = df[age + "%"] * .8 * df[location]
                else:
                    df[location + "_" + age] = df[age + "%"] * 1.35 * df[location]
            else:
                df[location + "_" + age] = df[age + "%"] * df[location]
            # ----- Add NH to Community for Young agents
            if (location == 'Community') & (age != '65+'):
                df[location + "_" + age] += df[age + "%"] * df['NH']
            # ----- Remove NH for Young Patients
            if (location == 'NH'):
                if age != '65+':
                    df[location + "_" + age] = 0
                else:
                    df[location + "_" + age] = df[location]

    # ----- Calculate Percentages
    pairs = dict()
    for hospital, row in df.iterrows():
        for age in ages:
            values = row.loc[[item + "_" + age for item in locations]]
            pairs[hospital + "_" + age] = [item / values.sum() for item in values]
    pairs_df = pd.DataFrame(pairs).T

    # ----- Find the row for each combination
    all_rows = []
    for hospital in df.index:
        for county_code in discharges.County_Code.values:
            for age in ages:
                values = pairs_df.loc[hospital + "_" + age]
                row = [county_code, age, hospital]
                # Find Community
                row.append(values[0])
                # Find Hospitals
                if discharges[discharges.County_Code == county_code].Total.values[0] > 0:
                    row.append(values[1] * to_unc)
                    row.append(values[1] * (1 - to_unc) * to_large)
                    row.append(values[1] * (1 - to_unc) * (1 - to_large))
                else:
                    row.append(0)
                    row.append(values[1] * to_large)
                    row.append(values[1] * (1 - to_large))
                # LT and NH
                row.append(values[2])
                row.append(values[3])
                all_rows.append(row)

    df2 = pd.DataFrame(all_rows)
    df2.columns = ['County_Code', 'Age_Group', 'Facility', 'COMMUNITY', 'UNC', 'LARGE', 'SMALL', 'LT', 'NH']
    return df2


# ----- What to run when the script is called ----- #
if __name__ == '__main__':

    make_community_transitions()
    make_location_transitions()
    make_unc_to_unc_discharges()
    make_nonUNC_discharges()
