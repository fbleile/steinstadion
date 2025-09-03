#!/usr/bin/env python3
import os
import time

task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
user = os.environ.get("USER", "unknown")
home = os.environ.get("HOME", "unknown")

print(f"Hello from SLURM_ARRAY_TASK_ID={task_id}")
print(f"User: {user}, Home: {home}")
time.sleep(2)
print(f"Task {task_id} finished!")

# --- Write to file ---
output_path = f"experiment/test_{task_id}.txt"
with open(output_path, "w") as f:
    f.write(f"Task {task_id} report\n")
    f.write(f"User: {user}\n")
    f.write(f"Home: {home}\n")
    f.write("This is a test output file from the SLURM array job.\n")