#!/bin/bash
#SBATCH --mem=90G
#SBATCH --time=96:00:00
#SBATCH --output=out/log/%j.out
#SBATCH --mail-type=ALL --mail-user=mlobillich1@sheffield.ac.uk

module load Anaconda3/2022.05

module list

echo "JOB ID: $SLURM_JOB_ID"

source activate env_dissertation_project
which python
python -V

script=$1
shift
python $script JOB_ID=$SLURM_JOB_ID "$@"
