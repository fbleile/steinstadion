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
        "sbatch --wrap='"
        "module load slurm_setup; "
        "source $HOME/miniconda3/etc/profile.d/conda.sh; "
        "conda activate steinstadion-env; "
        "python $HOME/steinstadion/experiment/test_script.py'"
        f" --job-name=test_child_{i}"
    )
    commands.append(cmd)

commands_file = "experiment/commands_list.txt"  # fixed path
with open(commands_file, "w") as f:
    for cmd in commands:
        f.write(cmd + "\n")

# Convert list of commands into a single string using a delimiter
commands_str = ":::".join(commands)

# Number of commands
NUM_COMMANDS = len(commands)

# Generate the array job submission command
submit_cmd = f"sbatch --array=1-{NUM_COMMANDS}%200 jobsubmission.sh {commands_file}"

print("Run this command on the login node to submit the array job:")
if dry:
    print(submit_cmd)
else:
    os.system(submit_cmd)


# python experiment/test_jobsubmission.py --num-jobs 2 --submit