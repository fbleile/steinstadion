#!/bin/bash
#SBATCH -D /dss/dsshome1/0C/ge86xim2/steinstadion
#SBATCH -o slurm_logs/jobfarm.%N.%j.out
#SBATCH -J JobFarm
#SBATCH --get-user-env
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --mem=1000M
#SBATCH --time=2:00:00
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
taskdb="experiment/command_list"
txt_file="{$taskdb}.txt"
# delete prev jobfarm db
rm -f "{$taskdb}.db"
rm -rf "{$taskdb}.txt_res"
# Start JobFarm
jobfarm start $txtfile
