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
