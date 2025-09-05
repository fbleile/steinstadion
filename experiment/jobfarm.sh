#!/bin/bash
#SBATCH -D /dss/dsshome1/0C/ge86xim2/steinstadion
#SBATCH -o slurm_logs/jobfarm.%N.%j.out
#SBATCH -J JobFarm
#SBATCH --get-user-env
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=5:59:00
#SBATCH --mail-type=NONE
#SBATCH --export=ALL
module load slurm_setup
module load jobfarm
source /dss/dsshome1/0C/ge86xim2/miniconda3/etc/profile.d/conda.sh
conda activate steinstadion-env
export TMPDIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
mkdir -p /tmp/matplotlib
export PYTHONPATH=$PYTHONPATH:/dss/dsshome1/0C/ge86xim2/steinstadion
# Use the existing command list
taskdb="experiment/command_list.txt"
# Start JobFarm
jobfarm start $taskdb
