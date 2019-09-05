## Run Simulation Scenarios

`docker-compose run -d hai bash -c "python3.6 NCMIND/stewardship_paper/src/scenario_runs.py"`

#### Commands

Assess is we did drop antibiotics by 10, 20, and 30%
`docker-compose run  hai bash -c "python3.6 NCMIND/stewardship_paper/src/antibiotic_analysis.py"`

Grab the files:
`scp krjones@baldur.rtp.rti.org:/home/krjones/cdc-hai/NCMIND/stewardship_paper/scenario_output_aggregated.csv .`

`scp krjones@baldur.rtp.rti.org:/home/krjones/cdc-hai/NCMIND/stewardship_paper/scenario_output.csv .`

`scp krjones@baldur.rtp.rti.org:/home/krjones/cdc-hai/NCMIND/stewardship_paper/scenario_means.csv .`

`scp krjones@baldur.rtp.rti.org:/home/krjones/cdc-hai/NCMIND/stewardship_paper/antibiotics_results.csv .`

