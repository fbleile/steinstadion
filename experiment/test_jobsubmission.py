import os

# Number of test sbatch jobs
N_TEST_JOBS = 10  # adjust for testing
MAX_CONCURRENT = 200

# Path to your array job script
myarray_sh = os.path.abspath("jobsubmission.sh")

# Generate sbatch commands with environment setup and inline Python
commands = []
for i in range(1, N_TEST_JOBS + 1):
    cmd = (
        "sbatch --wrap=\""
        "module load slurm_setup; "
        f"source $HOME/miniconda3/etc/profile.d/conda.sh; "
        "conda activate steinstadion-env; "
        "export TMPDIR=/tmp; export MPLCONFIGDIR=/tmp/matplotlib; mkdir -p /tmp/matplotlib; "
        "export PYTHONPATH=$PYTHONPATH:$HOME/steinstadion; "
        f"python -c 'import os, time; "
        "task_id=os.environ.get(\"SLURM_ARRAY_TASK_ID\", \"0\"); "
        "print(f\"Hello from SLURM_ARRAY_TASK_ID={task_id}\"); "
        "print(f\"User: {os.environ.get(\"USER\")}, Home: {os.environ.get(\"HOME\")} \"); "
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
print(submit_cmd)
