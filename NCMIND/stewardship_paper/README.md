## Run Simulation Scenarios

### Antibiotic Reduction

Before you can run antibiotic reduction scenarios, you need to figure out how much to reduce antibiotics by 10% and 20% for NH and STACHs

Run the following:
```docker-compose run -d hai bash -c "python3.6 NCMIND/stewardship_paper/src/lower_antibiotics.py"```

This will create enough output for you to create a trendline to determine how much to reduce antibiotics daily rates by. Review "NCMIND/stewardship_paper/output/percent_drop.csv". The trendline can be calculated in excel by making a scatterplot plot and then viewing the trendline. 

This has already been completed and incorporated into the scenarios.


### Complete the runs
`docker-compose run -d hai bash -c "python3.6 NCMIND/stewardship_paper/src/scenario_runs.py --num_agents=5000000 --runs_per=20"`
