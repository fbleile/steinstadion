#!/bin/bash
#SBATCH --job-name=dynamic_submitter
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH -D ./
# --- Config ---
MAX_JOBS=200
# --- Paths ---
SCRIPT_DIR=$(dirname "$(realpath "$0")")
COMMANDS_FILE="$SCRIPT_DIR/command_list.txt"
# --- Safety checks ---
if [[ ! -f "$COMMANDS_FILE" ]]; then
    echo "Error: commands file '$COMMANDS_FILE' does not exist."
    exit 1
fi
if [[ ! -s "$COMMANDS_FILE" ]]; then
    echo "Error: commands file '$COMMANDS_FILE' is empty."
    exit 1
fi
# --- Functions ---
get_current_jobs() {
    # Count all pending or running tasks (including array tasks) of this user on the serial cluster
    squeue -u $USER --clusters=serial -h -t PD,R,CG -o "%i" | awk '
    {
        gsub(/\[|\]/,"");          # remove brackets
        split($1, parts, "_");     # parts[1]=jobID, parts[2]=task range
        if (parts[2] == "") {
            print 1
        } else if (parts[2] ~ /-/) {
            split(parts[2], range, "-")
            print range[2]-range[1]+1
        } else {
            print 1
        }
    }' | paste -sd+ - | bc
}
count_jobs_in_command() {
    local cmd="$1"
    if [[ "$cmd" =~ --array[=[:space:]]+([0-9,-:]+) ]]; then
        local arr="${BASH_REMATCH[1]}"
        python - <<EOF
expr = "$arr"
total = 0
for part in expr.split(","):
    if "-" in part:
        rng = part.split(":")[0]
        step = 1
        if ":" in part:
            rng, step = part.split(":")
            step = int(step)
        start, end = map(int, rng.split("-"))
        total += (end - start)//step + 1
    else:
        total += 1
if total > $MAX_JOBS:
    import sys
    print(f"Error: array sbatch command would submit {total} jobs, exceeding MAX_JOBS={MAX_JOBS}.", file=sys.stderr)
    sys.exit(1)
print(total)
EOF
    else
        echo 1
    fi
}
# --- Main loop ---
while :; do
    # Read the first line (FIFO)
    cmd=$(head -n 1 "$COMMANDS_FILE")
    
    if [[ -z "$cmd" ]]; then
        echo "No more commands left in '$COMMANDS_FILE'. Exiting."
        exit 0
    fi
    # Count jobs this command would submit
    needed=$(count_jobs_in_command "$cmd")
    # Wait until there's enough room
    while true; do
        current=$(get_current_jobs)
        if (( current + needed <= MAX_JOBS )); then
            echo "$(date '+%Y-%m-%d %H:%M:%S') Submitting: $cmd"
            # Run the command and capture both stderr and exit code
            output=$(bash -c "$cmd" 2>&1)
            exit_code=$?
            if [[ $exit_code -ne 0 ]]; then
                if echo "$output" | grep -q "AssocMaxSubmitJobLimit"; then
                    echo "Hit submission limit (AssocMaxSubmitJobLimit). Waiting..."
                    sleep 60
                    # retry the same command without removing it from file
                    continue
                else
                    echo "Error running command: $output"
                    # decide: either exit or skip this command
                    exit 1
                fi
            fi
            # If submission succeeded, remove the executed line from file (pop)
            tail -n +2 "$COMMANDS_FILE" > "$COMMANDS_FILE.tmp" && mv "$COMMANDS_FILE.tmp" "$COMMANDS_FILE"
            break
        else
            echo "Currently $current jobs on serial cluster, need $needed more. Waiting..."
            sleep 30
        fi
    done
done