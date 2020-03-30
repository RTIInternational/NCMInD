
import pandas as pd
import numpy as np
import os

from NCMIND.src.analyze import Analyze
from src.state import NameState
from src.misc_functions import int_to_category


if __name__ == "__main__":
    experiment = "NCMIND"
    scenarios = os.listdir("NCMIND/stewardship_paper/")

    results = []
    for scenario in scenarios:
        print(scenario)
        if ("1." in scenario) or ("normal" in scenario):
            scenario = "stewardship_paper/" + scenario
            runs = os.listdir(experiment + "/" + scenario)

            community_antibiotics = []
            nh_antibiotics = []
            lt_antibiotics = []
            unc_antibiotics = []

            for run in runs:
                if "run_" in run:
                    analysis = Analyze(experiment=experiment, scenario=scenario, run=run)

                    # Find all antibiotic cases:
                    events = analysis.events
                    events['Category'] = int_to_category(analysis.locations, events.Location)
                    catchment = events.County.isin(analysis.catchment_counties)
                    anti = events[(events.State == NameState.ANTIBIOTICS) & (events.New == 1) & (catchment)]

                    community_antibiotics.append(len(anti[anti.Category == 'COMMUNITY']))
                    nh_antibiotics.append(len(anti[anti.Category == 'NH']))
                    lt_antibiotics.append(len(anti[anti.Category == 'LT']))
                    unc_antibiotics.append(len(anti[anti.Category == 'UNC']))

            results.append([scenario,
                            np.mean(community_antibiotics),
                            np.mean(nh_antibiotics),
                            np.mean(lt_antibiotics),
                            np.mean(unc_antibiotics)])

    df = pd.DataFrame(results)
    df.columns = ['Scenario', 'Community', 'NH', 'LT', 'UNC']
    df = df.sort_values(by='Scenario')
    df.to_csv("NCMIND/stewardship_paper/antibiotics_results.csv", index=False)
