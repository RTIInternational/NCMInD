<<<<<<< HEAD
## Transition Data
=======
## COVID


### Transition Data
>>>>>>> 5648d68836179261fc5b458a48a08ec66efd8b8b

The data behind the location transitions is not all publicly available. We have provided the aggregated transition data, but remove the source of the data. 

#### PDF Data
Some of the data is provided by extracting tables from PDFs. We have extracted this data and provided two CSVs here `modules/data/input/from_pdf/`

These files need to be cleaned and pasted into the 6x6. 

```
docker-compose run --rm hai bash -c "python3.6 modules/src/clean_pdf_csvs.py"
```

This results in 6 files in `data/temp/` that need to be added to the 6x6. These must be manually added to the 6x6 because excel does some calculations to find our target values. Paste the six files into the 6x6 using the following map:

- `unc_discharges.csv`: `unc-D`
- `cat_1_discharges.csv`: `Large-D`
- `cat_2_discharges.csv`: `Small-D`
- `unc_breakdown.csv`: `unc-Data`
- `cat_1_breakdown.csv`: `Large`
- `cat_2_breakdown.csv`: `Small`



We then read these files, clean them, and prepare them for the model. To do so, run the following:

```
docker-compose run --rm hai bash -c "python3.6 modules/src/fix_format.py"
```

#### Death Probabilities:
In `modules/data/` there is an excel file called `mortality.xlsx`. These mortality rates were taken from CDC
Wonder. We looked at NC deaths by age, sex, and race. [link](https://wonder.cdc.gov/controller/datarequest/D140;jsessionid=5A767ED9BA64A7E66597304A920BC503)

We ended up only using death rates by age, with multipliers for risk based on location. Death probabilities are in the `parameters.json` file. 


<<<<<<< HEAD
## Synthetic Population
=======
### 2: Extract and Clean Synthetic Population
>>>>>>> 5648d68836179261fc5b458a48a08ec66efd8b8b

There is an intensive process to prepare RTIs synthetic population for use in this project. We are not providing those details here. 

The final version of the synthetic population used in these models has been provided. 


#### A note on initial populations: 
Initial populations for facilities are based on two factors.

- Bed size
- The Census tracts that are aligned with that hospital

To initialize populations for facilities, we loop through each facility, find the tracts that are aligned with that facility, and randomly select individuals (based on age) to start in that facility. 

## Demo for Orange County 
To demo the model, we use the population of Orange County.

Run the following:

```
docker-compose run --rm hai bash -c "python3.6 run_model.py NCMInD cdi_demo run_1"
```

And check the `NCMInD/cdi_demo/run_1/` for the model output files. 


## Stewardship Runs:

### Antibiotic Reduction

Before you can run antibiotic reduction scenarios, you need to figure out how much to reduce antibiotics by 10% and 20% for NH and STACHs

Run the following:
```docker-compose run -d hai bash -c "python3.6 NCMIND/stewardship_paper/src/lower_antibiotics.py"```

This will create enough output for you to create a trendline to determine how much to reduce antibiotics daily rates by. Review "NCMIND/stewardship_paper/output/percent_drop.csv". The trendline can be calculated in excel by making a scatterplot plot and then viewing the trendline. 

This has already been completed and incorporated into the sceanarios. 


### Run the models:

We ran each scenario with 5m agents and 20 runs per model using the following:

```
docker-compose run -d hai bash -c "python3.6 NCMIND/stewardship_paper/src/scenario_runs.py --num_agents=5000000 --runs_per=20"
```





