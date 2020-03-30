
import pandas as pd

if __name__ == '__main__':
    # ----- Read the hospital names and categories
    c = pd.read_csv("NCMIND/data/IDs/hospital_ids.csv")
    # ----- For the lowercased names
    lower_ids = c[['ID']]
    lower_ids.index = c['Column Name']
    lower_ids_dict = lower_ids.to_dict()['ID']
    # ----- For the proper names
    proper_ids = c[['ID']]
    proper_ids.index = c['PDF Name']
    proper_ids_dict = proper_ids.to_dict()['ID']

    county_codes = pd.read_csv("NCMIND/data/county_codes.csv")[['County', 'County_Code']]

    # ----- Discharges by Hospital/County
    a = pd.read_csv("NCMIND/data/input/from_pdf/2017_master_ptorg_final.csv")
    a = a.fillna(0)
    a = a.rename(columns = {'RESIDENCE': 'County'})
    drops =\
        a[a.County.isin(['Actual', 'Calculated', 'Unreported', 'TENNESSEE', 'SOUTH CAROLINA',
        'VIRGINIA', 'Other/Missing', 'GEORGIA'])]
    a = a.drop(drops.index).reset_index(drop=True)
    a['johnston_health'] = a['johnston_health_clayton'] + a['johnston_health_smithfield']
    a['unc_chapel_hill'] = a['unc_hillsborough'] + a['university_of_north_carolina_hospitals']

    unc_hospitals = list(c[c['Category'] == 'UNC']['Column Name'])
    cat_1 = list(c[c['Category'] == 'Large']['Column Name'])
    cat_2 = list(c[c['Category'] == 'Small']['Column Name'])


    # Split the PDF Data into the three Categories
    unc_discharges = a[['County'] + list(a.columns[a.columns.isin(unc_hospitals)])]
    unc_discharges.columns = [
        lower_ids_dict[item] if item in lower_ids_dict else item for item in unc_discharges.columns]
    columns = list(unc_discharges.columns)
    columns.sort()
    unc_discharges = county_codes.merge(unc_discharges[columns])
    unc_discharges.insert(2, 'Total', unc_discharges.drop('County_Code', axis=1).sum(axis='columns'))
    unc_discharges.to_csv("NCMIND/data/temp/unc_discharges.csv", index=False)

    c1_discharges = a[['County'] + list(a.columns[a.columns.isin(cat_1)])]
    c1_discharges.columns = [lower_ids_dict[item] if item in lower_ids_dict else item for item in c1_discharges.columns]
    c1_discharges = county_codes.merge(c1_discharges)
    c1_discharges.insert(2, 'Total', c1_discharges.drop('County_Code', axis=1).sum(axis='columns'))
    c1_discharges.to_csv("NCMIND/data/temp/cat_1_discharges.csv", index=False)

    c2_discharges = a[['County'] + list(a.columns[a.columns.isin(cat_2)])]
    c2_discharges.columns = [lower_ids_dict[item] if item in lower_ids_dict else item for item in c2_discharges.columns]
    c2_discharges = county_codes.merge(c2_discharges)
    c2_discharges.insert(2, 'Total', c2_discharges.drop('County_Code', axis=1).sum(axis='columns'))
    c2_discharges.to_csv("NCMIND/data/temp/cat_2_discharges.csv", index=False)

    # ----- Age/Disposition Breakdown
    b = pd.read_csv("NCMIND/data/input/from_pdf/2017_subset_ptchar_for_analysis.csv").fillna(0)
    b = b.rename(columns = {'Unnamed: 0': 'Category'})
    b['Johnston Health'] = b['Johnston Health Clayton'] + b['Johnston Health Smithfield']
    b = b.drop(['Actual', 'Calculated', 'Difference'], axis='columns')

    order = [
        'Patient Residence State NC',
        'Patient Residence State Not NC',
        'Age Group Less than 1 Year',
        'Age Group 1 - 17 years',
        'Age Group 18 - 44 years',
        'Age Group 45 - 64 years',
        'Age Group 65 - 84 years',
        'Age Group 85 or more years',
        'Patient Disposition Home, self, or outpatient care',
        'Patient Disposition Discharged, transferred to acute facility',
        'Patient Disposition Discharged, transferred to facility that provides nursing, custodial, or supportive care',
        'Patient Disposition Discharged, transferred',
        'Patient Disposition Discharged, transferred to long term acute care facility (LTAC)',
        'Patient Disposition Discharged, transferred to psychiatric facility',
        'Patient Disposition Hospice',
        'Patient Disposition Left against medical advice',
        'Patient Disposition Court/Law Enforcement',
        'Patient Disposition Expired',
        'Patient Disposition Other/Unknown'
    ]
    b = b.set_index("Category").loc[order] 
    b = b.reset_index()

    unc_hospitals_b = list(c[c['Category'] == 'UNC']['PDF Name'])
    cat_1_b = list(c[c['Category'] == 'Large']['PDF Name'])
    cat_2_b = list(c[c['Category'] == 'Small']['PDF Name'])

    # Split the PDF Data into the three Categories
    unc_breakdown = b[['Category'] + list(b.columns[b.columns.isin(unc_hospitals_b)])]
    unc_breakdown.columns = [
        proper_ids_dict[item] if item in proper_ids_dict else item for item in unc_breakdown.columns]
    columns = list(unc_breakdown.columns)
    columns.sort()
    unc_breakdown = unc_breakdown[columns]
    unc_breakdown.insert(1, 'Total', unc_breakdown.sum(axis='columns'))
    unc_breakdown = unc_breakdown.append(unc_breakdown.loc[0:1].sum(numeric_only=True), ignore_index=True)
    unc_breakdown.to_csv("NCMIND/data/temp/unc_breakdown.csv", index=False)

    cat_1_breakdown = b[['Category'] + list(b.columns[b.columns.isin(cat_1_b)])]
    cat_1_breakdown.columns = [
        proper_ids_dict[item] if item in proper_ids_dict else item for item in cat_1_breakdown.columns]
    columns = list(cat_1_breakdown.columns)
    columns = ['Category'] + sorted(columns[1:], key = lambda x: int(x.split("_")[1]))
    cat_1_breakdown = cat_1_breakdown[columns]
    cat_1_breakdown.insert(1, 'Total', cat_1_breakdown.sum(axis='columns'))
    cat_1_breakdown = cat_1_breakdown.append(cat_1_breakdown.loc[0:1].sum(numeric_only=True), ignore_index=True)
    cat_1_breakdown.to_csv("NCMIND/data/temp/cat_1_breakdown.csv", index=False)

    cat_2_breakdown = b[['Category'] + list(b.columns[b.columns.isin(cat_2_b)])]
    cat_2_breakdown.columns = [
        proper_ids_dict[item] if item in proper_ids_dict else item for item in cat_2_breakdown.columns]    
    columns = list(cat_2_breakdown.columns)
    columns = ['Category'] + sorted(columns[1:], key = lambda x: int(x.split("_")[1]))
    cat_2_breakdown = cat_2_breakdown[columns]
    cat_2_breakdown.insert(1, 'Total', cat_2_breakdown.sum(axis='columns'))
    cat_2_breakdown = cat_2_breakdown.append(cat_2_breakdown.loc[0:1].sum(numeric_only=True), ignore_index=True)
    cat_2_breakdown.to_csv("NCMIND/data/temp/cat_2_breakdown.csv", index=False)
