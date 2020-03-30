
from src.analyze import Analyze
from src.state import NameState, CDIState
import pandas as pd

# Get the locations
a_object = Analyze(exp_dir='NCMIND/demo', scenario='cdi_example', run='run_1')

normal = pd.read_csv("NCMIND/stewardship_paper/from_baldur/normal.csv", compression='gzip')
unc10 = pd.read_csv("NCMIND/stewardship_paper/from_baldur/unc_10.csv", compression='gzip')
unc20 = pd.read_csv("NCMIND/stewardship_paper/from_baldur/unc_20.csv", compression='gzip')
unc30 = pd.read_csv("NCMIND/stewardship_paper/from_baldur/unc_30.csv", compression='gzip')

use_ints = [item for item in a_object.locations.values if item not in [
    a_object.locations.ints['COMMUNITY'], a_object.locations.ints['LT'],
    a_object.locations.ints['NH'], a_object.locations.ints['ST']]]


# How many antibiotics at UNC STACHS
print(normal[(normal.State == NameState.ANTIBIOTICS) & (normal.New == 1) & (normal.Location.isin(use_ints))].shape[0])
print(unc10[(unc10.State == NameState.ANTIBIOTICS) & (unc10.New == 1) & (unc10.Location.isin(use_ints))].shape[0])
print(unc20[(unc20.State == NameState.ANTIBIOTICS) & (unc20.New == 1) & (unc20.Location.isin(use_ints))].shape[0])
print(unc30[(unc30.State == NameState.ANTIBIOTICS) & (unc30.New == 1) & (unc30.Location.isin(use_ints))].shape[0])

# How many CDI cases at UNC STACHS
print(normal[(normal.State == NameState.CDI_RISK) & (normal.New == CDIState.CDI) &
             (normal.Location.isin(use_ints))].shape[0])
print(unc10[(unc10.State == NameState.CDI_RISK) & (unc10.New == CDIState.CDI) &
            (unc10.Location.isin(use_ints))].shape[0])
print(unc20[(unc20.State == NameState.CDI_RISK) & (unc20.New == CDIState.CDI) &
            (unc20.Location.isin(use_ints))].shape[0])
print(unc30[(unc30.State == NameState.CDI_RISK) & (unc30.New == CDIState.CDI) &
            (unc30.Location.isin(use_ints))].shape[0])

# How many colonized cases at UNC STACHS
print(normal[(normal.State == NameState.CDI_RISK) & (normal.New == CDIState.COLONIZED) &
             (normal.Location.isin(use_ints))].shape[0])
print(unc10[(unc10.State == NameState.CDI_RISK) & (unc10.New == CDIState.COLONIZED) &
            (unc10.Location.isin(use_ints))].shape[0])
print(unc20[(unc20.State == NameState.CDI_RISK) & (unc20.New == CDIState.COLONIZED) &
            (unc20.Location.isin(use_ints))].shape[0])
print(unc30[(unc30.State == NameState.CDI_RISK) & (unc30.New == CDIState.COLONIZED) &
            (unc30.Location.isin(use_ints))].shape[0])


# ----- Scenario: Round #2
unc10_2 = pd.read_csv("NCMIND/stewardship_paper/from_baldur/2_unc_10.csv", compression='gzip')
unc20_2 = pd.read_csv("NCMIND/stewardship_paper/from_baldur/2_unc_20.csv", compression='gzip')

# How many antibiotics at UNC STACHS - Should all be fairly similar
print(normal[(normal.State == NameState.ANTIBIOTICS) & (normal.Old == 0) & (normal.New == 1) &
             (normal.Location.isin(use_ints))].shape[0])
print(unc10_2[(unc10_2.State == NameState.ANTIBIOTICS) & (unc10_2.Old == 0) & (unc10_2.New == 1) &
              (unc10_2.Location.isin(use_ints))].shape[0])
print(unc20_2[(unc20_2.State == NameState.ANTIBIOTICS) & (unc20_2.Old == 0) & (unc20_2.New == 1) &
              (unc20_2.Location.isin(use_ints))].shape[0])

# How much should the assigned risk ratio change?
avg1 = (4800 * .4 * 2 + 4800 * .3 * 5 + 4800 * .3 * 12)/4800
print(avg1)
avg2 = (4800 * .4 * 2 + 4800 * .4 * 5 + 4800 * .2 * 12)/4800
print(avg2)
avg3 = (4800 * .4 * 2 + 4800 * .5 * 5 + 4800 * .1 * 12)/4800
print(avg3)


# How many CDI cases at UNC STACHS
print(normal[(normal.State == NameState.CDI_RISK) & (normal.New == CDIState.CDI) &
             (normal.Location.isin(use_ints))].shape[0])
print(unc10_2[(unc10_2.State == NameState.CDI_RISK) & (unc10_2.New == CDIState.CDI) &
              (unc10_2.Location.isin(use_ints))].shape[0])
print(unc20_2[(unc20_2.State == NameState.CDI_RISK) & (unc20_2.New == CDIState.CDI) &
              (unc20_2.Location.isin(use_ints))].shape[0])
