import os
import sys
from pathlib import Path
import argparse

# Append project root for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stadion.utils.launch import generate_run_commands

DEFAULT_DATA_GEN_TYPES = [
    "linear-er",
    "linear-sf",
    "scm-er",
    "scm-sf",
    "sergio-er",
    "sergio-sf",
]

def make_cmds(exp, n_datasets, methods):
    methods_str = " ".join(methods)
    return [
        (
            f"python manager.py {exp} "
            f"--data --submit --n_datasets={n_datasets}"
        ),
        (
            f"python manager.py {exp} "
            f"--methods_train_validation --n_datasets {n_datasets} "
            f"--submit --only_methods {methods_str}"
        ),
        (
            f"python manager.py {exp} "
            f"--summary_train_validation --n_datasets {n_datasets} "
            f"--submit --only_methods {methods_str}"
        ),
    ]

def launch_all(data_gen_types, args):
    all_commands = []
    for exp in data_gen_types:
        cmds = make_cmds(exp, args.n_datasets, args.only_methods)
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
    parser.add_argument("--only_methods", nargs="+", default=["ours-lnl_u_diag", "ours-linear_u_diag"],
                        help="Restrict to specified methods")
    parser.add_argument("--compute", type=str, default="local", choices=["local", "cluster"])
    parser.add_argument("--only_gen_types", nargs="+", choices=DEFAULT_DATA_GEN_TYPES,
                        help="Restrict to specified data generation types")

    args = parser.parse_args()

    selected_gen_types = args.only_gen_types or DEFAULT_DATA_GEN_TYPES
    launch_all(selected_gen_types, args)

# sample usage
# python hyperparam.py --submit --n_datasets 50 --only_methods ours-linear_u_diag ours-lnl_u_diag
