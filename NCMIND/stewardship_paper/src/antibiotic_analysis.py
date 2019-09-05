
import os
from numpy import mean
import sys
sys.path.append('')

try:
    from src.analyze import *
    from src.ldm import *
except Exception as E:
    raise


if __name__ == "__main__":
    folder = "NCMIND/stewardship_paper/"
    scenarios = os.listdir(folder)

    results = []
    for scenario in scenarios:
        print(scenario)
        if ("1." in scenario) or ("normal" in scenario):
            runs = os.listdir(folder + scenario)

            community_antibiotics = []
            nh_antibiotics = []
            lt_antibiotics = []
            unc_antibiotics = []

            for run in runs:
                if "run_" in run:
                    a_object = Analyze(exp_dir=folder, scenario=scenario, run=run)

                    # Find all antibiotic cases:
                    e = a_object.events
                    anti = e[(e.State == NameState.ANTIBIOTICS) & (e.New == 1) &
                             (e.County.isin(a_object.catchment_counties))]

                    community_antibiotics.append(anti[anti.Location.isin(a_object.COMMUNITY)].shape[0])
                    nh_antibiotics.append(anti[anti.Location.isin(a_object.NH)].shape[0])
                    lt_antibiotics.append(anti[anti.Location.isin(a_object.LT)].shape[0])
                    unc_antibiotics.append(anti[anti.Location.isin(a_object.UNC)].shape[0])

            results.append([scenario,
                            mean(community_antibiotics),
                            mean(nh_antibiotics),
                            mean(lt_antibiotics),
                            mean(unc_antibiotics)])

    df = pd.DataFrame(results)
    df.columns = ['Scenario', 'Community', 'NH', 'LT', 'UNC']
    df = df.sort_values(by='Scenario')
    df.to_csv(folder + "antibiotics_results.csv", index=False)
