import os
import sys
from pathlib import Path
import argparse

# Append project root for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stadion.utils.launch import generate_run_commands
from definitions import DEFAULT_DATA_GEN_TYPES

def make_cmds(exp, args):
    methods_str = " ".join(args.only_methods or [])
    cmds = []

    if args.run_data:
        cmds.append(
            f"python manager.py {exp} "
            f"--data --submit --n_datasets={args.n_datasets} "
            f"--compute {args.compute}"
        )

    if args.run_methods:
        cmds.append(
            f"python manager.py {exp} "
            f"--methods_train_validation --n_datasets {args.n_datasets} "
            f"--submit --only_methods {methods_str} "
            f"--compute {args.compute}"
        )

    if args.run_summary:
        cmds.append(
            f"python manager.py {exp} "
            f"--summary_train_validation --n_datasets {args.n_datasets} "
            f"--submit --only_methods {methods_str} "
            f"--compute {args.compute}"
        )

    return cmds


def launch_all(data_gen_types, args):
    all_commands = []
    for exp in data_gen_types:
        cmds = make_cmds(exp, args)
        all_commands.extend(cmds)

    generate_run_commands(
        command_list=all_commands,
        dry=not args.submit,
        mode=args.compute,  # or "cluster" if cluster submission is preferred
        hours=5,
        mins=0,
        n_cpus=4,
        n_gpus=1,
        mem=4000,
        prompt=False,
        output_path_prefix=None,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch hyperparameter runs.")
    parser.add_argument("--submit", action="store_true", help="Actually submit jobs (default: dry-run)")
    parser.add_argument("--n_datasets", type=int, default=50, help="Number of datasets to generate")
    parser.add_argument("--only_methods", nargs="+", help="Restrict to specified methods")
    parser.add_argument("--compute", type=str, default="local", choices=["local", "cluster"])
    parser.add_argument("--only_gen_types", nargs="+", choices=DEFAULT_DATA_GEN_TYPES,
                        help="Restrict to specified data generation types")
    
    parser.add_argument("--run_data", action="store_true", help="Run the --data step")
    parser.add_argument("--run_methods", action="store_true", help="Run the --methods_train_validation step")
    parser.add_argument("--run_summary", action="store_true", help="Run the --summary_train_validation step")
    
    args = parser.parse_args()
    
    # If none of the run flags are explicitly passed, default to all True
    if not (args.run_data or args.run_methods or args.run_summary):
        args.run_data = True
        args.run_methods = True
        args.run_summary = True

    selected_gen_types = args.only_gen_types or DEFAULT_DATA_GEN_TYPES
    launch_all(selected_gen_types, args)

# sample usage
# python hyperparam.py --submit --n_datasets 50

# python hyperparam.py --run_data --n_datasets 50         # Only data
# python hyperparam.py --run_summary --n_datasets 50      # Only summary
# python hyperparam.py --run_data --run_methods            # Data + methods, skip summar
