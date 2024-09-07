#!/bin/bash
#SBATCH --mem=90G
#SBATCH --time=96:00:00
#SBATCH --output=/dev/null
#SBATCH --mail-type=ALL --mail-user=mlobillich1@sheffield.ac.uk

out_path=$1
script_log="out/${out_path}/${SLURM_JOB_ID}_log.out"

echo "$@"
echo "JOB ID: $SLURM_JOB_ID"

shift
script=$1
shift

module load Anaconda3/2022.05

module list

source activate env_dissertation_project
which python
python -V

python $script "OUT_PATH=${out_path}" JOB_ID=$SLURM_JOB_ID "$@" >> $script_log 2>&1

echo "Done."
