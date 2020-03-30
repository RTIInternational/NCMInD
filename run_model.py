
import argparse as arg
from pathlib import Path

from src.ldm import Ldm


if __name__ == '__main__':
    description = """ Run a model. Must provide at least an experiemtn and a scenario directory."""

    parser = arg.ArgumentParser(description=description)

    parser.add_argument(
        'exp_dir',
        default='NCMIND/location_demo',
        help='The directory of the experiment'
    )
    parser.add_argument(
        'scenario',
        default='default',
        help='The name of the scenario.'
    )
    parser.add_argument(
        'run',
        default='',
        help='The name of the run for the scenario'
    )

    args = parser.parse_args()

    scenario_dir = Path(args.exp_dir, args.scenario)
    # ----- initialize model
    m = Ldm(args.exp_dir, args.scenario, args.run)
    # --- Run the Model
    m.run_model()
    # --- Close the SQL Connection
    m.collapse_sql_connection()
