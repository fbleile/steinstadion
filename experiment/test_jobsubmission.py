import os
import argparse
import tempfile

# -------------------------------
# Parse command-line argument
# -------------------------------
parser = argparse.ArgumentParser(description="Submit a Slurm array job with N test tasks")
parser.add_argument(
    "--num-jobs",
    type=int,
    default=3,
    help="Number of test sbatch jobs to deploy (default: 3)"
)
parser.add_argument(
    "--submit",
    action="store_true",
    help="If set, actually submit the array job; otherwise dry-run"
)
args = parser.parse_args()
N_TEST_JOBS = args.num_jobs
dry = not args.submit  # True if --submit is not provided

# -------------------------------
# Constants
# -------------------------------
MAX_CONCURRENT = 200
josubmission_sh = os.path.abspath("experiment/jobsubmission.sh")

# -------------------------------
# Generate sbatch commands
# -------------------------------
commands = []
for i in range(1, N_TEST_JOBS + 1):
    cmd = (
        "sbatch "
        "--get-user-env "
        "--export=ALL "
        "--clusters=serial "
        "--partition=serial_std "
        f"--cpus-per-task=1 "
        f"--mem=4000M "
        f"--time=2:00:00 "
        f"--job-name=test_child_{i} "
        "--wrap=\""
        "module load slurm_setup; "
        "source $HOME/miniconda3/etc/profile.d/conda.sh; "
        "conda activate steinstadion-env; "
        f"python $HOME/steinstadion/experiment/test_script.py\""
    )
    commands.append(cmd)

commands_file = "experiment/commands_list.txt"  # fixed path
with open(commands_file, "w") as f:
    for cmd in commands:
        f.write(cmd + "\n")

# Number of commands
NUM_COMMANDS = len(commands)

submit_cmd = (
    f"sbatch "
    "--get-user-env "
    "--export=ALL "
    "--clusters=serial "
    "--partition=serial_std "
    f"--cpus-per-task=1 "
    f"--mem=1000M "
    f"--time=2:00:00 "
    f"--array=1-{NUM_COMMANDS}%{MAX_CONCURRENT} "
    f"--wrap=\"export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0C/ge86xim2/steinstadion && "
    f"bash experiment/jobsubmission.sh {commands_file}\""
)


print("Run this command on the login node to submit the array job:")
if dry:
    print(submit_cmd)
else:
    print(submit_cmd)
    os.system(submit_cmd)


# python experiment/test_jobsubmission.py --num-jobs 2 --submit