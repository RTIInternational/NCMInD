
from time import time
import argparse as arg
from pathlib import Path

from src.ldm import LDM


if __name__ == '__main__':
    description = """ Run a model. If the name of input_dir starts with
    "scenario", then treat it as a scenario directory and run that scenario.
    Otherwise, treat it as an experiment directory by finding all the scenario
    subdirectories and running those scenarios."""

    parser = arg.ArgumentParser(description=description)

    parser.add_argument(
        'exp_dir',
        default='NCMIND/demo',
        help='The directory of the experiment'
    )
    parser.add_argument(
        'scenario',
        default='default',
        help='The name of the scenario.'
    )
    parser.add_argument(
        '--run',
        default='',
        help='The name of the run for the scenario'
    )

    args = parser.parse_args()
    print(args)

    scenario_dir = Path(args.exp_dir, args.scenario)
    run_dir = Path(args.exp_dir, args.scenario, args.run)
    output_dir = Path(run_dir, "model_output")

    # ----- initialize model
    ts = time()
    model = LDM(args.exp_dir, args.scenario, args.run)

    t = round((time() - ts) / 60 / 60, 5)
    with open(Path(output_dir, "model_creation_details.txt"), "w") as text_file:
        text_file.write("It took %s hours to initialize this model." % str(t))

    # ----- Run the Model
    model.run_model()

    t = round((time() - ts) / 60 / 60, 5)
    with open(Path(output_dir, "model_completion_details.txt"), "w") as text_file:
        text_file.write("It took %s hours to initialize and complete this model run." % str(t))

    # ----- Close the SQL Connection
    model.collapse_sql_connection()
