
import sys

from src.analyze import *
from src.ldm import *

model = LDM(exp_dir='NCMIND/cdi_calibration', scenario='default')
model.run_model()

# ----- Run analysis on the model results
a_object = Analyze(exp_dir='NCMIND/cdi_calibration', scenario='default')


# ----- What percent of antibiotics are prescribed where?
e = model.make_events()
print(e[(e['State'] == NameState.ANTIBIOTICS) & (e['New'] == 1)].shape[0])
print(e[(e['State'] == NameState.ANTIBIOTICS) & (e['New'] == 1)].Location.value_counts())
print(e[(e['State'] == NameState.ANTIBIOTICS) & (e['New'] == 1) &
        (e['Location'].isin(a_object.UNC + a_object.ST))].shape[0])


# ----------------------------------------------------------------------------------------------------------------------
# ----- Check the average colonization by location
df = pd.read_csv("NCMIND/cdi_calibration/default/model_output/daily_counts.csv")
# Check only living
df = df[df['Life'] == 1]
# LTACH: should match initialization percentage
df2 = df[df.Location == model.locations.ints['LT']]
df2.loc[df2.CDI == 2, df2.columns[4:]].sum().plot()
print(df2.loc[df2.CDI == 2, df2.columns[4:]].sum().sum() / df2.loc[:, df2.columns[4:]].sum().sum())
# NH: should match initialization percentage
df2 = df[df.Location == model.locations.ints['NH']]
df2.loc[df2.CDI == 2, df2.columns[4:]].sum().plot()
print(df2.loc[df2.CDI == 2, df2.columns[4:]].sum().sum() / df2.loc[:, df2.columns[4:]].sum().sum())
# STACHs: should match initialization percentage
df2 = df[[item in model.stachs for item in df.Location]]
df2.loc[df2.CDI == 2, df2.columns[4:]].sum().plot()
print(df2.loc[df2.CDI == 2, df2.columns[4:]].sum().sum() / df2.loc[:, df2.columns[4:]].sum().sum())
# COMMUNITY: should match initialization percentage
df2 = df[df.Location == 0]
df2.loc[df2.CDI == 2, df2.columns[4:]].sum().plot()
print(df2.loc[df2.CDI == 2, df2.columns[4:]].sum().sum() / df2.loc[:, df2.columns[4:]].sum().sum())

# Patient Days in Nursing Homes
p_days = a_object.calculate_patient_days(unc_only=True)
pd_nh = p_days[1][-1] * 10_000_000 / model.population.shape[0]
pd_nh / 10_000

# ----------------------------------------------------------------------------------------------------------------------
# ----- Colonized Patients in the community
print(pd.Series(model.cdi['data'][:, model.cdi['columns']['cdi_status']]).value_counts())
# --- What percent of colonized patients are on antibiotics?
colonized = model.cdi['data'][:, model.cdi['columns']['cdi_status']] == CDIState.COLONIZED
print(pd.Series(model.cdi['data'][:, model.cdi['columns']['antibiotics_status']][colonized]).value_counts())
print(pd.Series(model.cdi['data'][:, model.cdi['columns']['antibiotics_status']]).value_counts())
# Colonized Patients in STACHs
