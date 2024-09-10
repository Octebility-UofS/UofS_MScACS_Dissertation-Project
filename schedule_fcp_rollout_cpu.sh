#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null

module load Anaconda3/2022.05

module list

source activate env_dissertation_project
which python
python -V

python test_fcp_rollout.py "$@" cpu