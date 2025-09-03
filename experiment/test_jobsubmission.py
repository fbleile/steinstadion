import os
import argparse

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
myarray_sh = os.path.abspath("experiment/jobsubmission.sh")

# -------------------------------
# Generate sbatch commands
# -------------------------------
commands = []
for i in range(1, N_TEST_JOBS + 1):
    cmd = (
        "sbatch --wrap=\""
        "module load slurm_setup; "
        "source $HOME/miniconda3/etc/profile.d/conda.sh; "
        "conda activate steinstadion-env; "
        "export TMPDIR=/tmp; export MPLCONFIGDIR=/tmp/matplotlib; mkdir -p /tmp/matplotlib; "
        "export PYTHONPATH=$PYTHONPATH:$HOME/steinstadion; "
        f"python -c 'import os, time; "
        "task_id=os.environ.get(\"SLURM_ARRAY_TASK_ID\", \"0\"); "
        "print(f\"Hello from SLURM_ARRAY_TASK_ID={task_id}\"); "
        "print(f\"User: {os.environ.get('USER')}, Home: {os.environ.get('HOME')}\"); "
        "time.sleep(2); "
        "print(f\"Task {task_id} finished!\")'"
        f"\" --job-name=test_child_{i}"
    )
    commands.append(cmd)

# Convert list of commands into a single string using a delimiter
commands_str = ":::".join(commands)

# Number of commands
NUM_COMMANDS = len(commands)

# Generate the array job submission command
submit_cmd = (
    f"sbatch --array=1-{NUM_COMMANDS}%{MAX_CONCURRENT} "
    f"--export=ALL,COMMANDS=\"{commands_str}\" "
    f"{myarray_sh}"
)

print("Run this command on the login node to submit the array job:")
if dry:
    print(submit_cmd)
else:
    os.system(submit_cmd)


# python experiment/test_jobsubmission.py --num-jobs 2 --submit