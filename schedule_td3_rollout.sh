#!/bin/bash
#SBATCH --partition=gpu --qos=gpu --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/dev/null

module load Anaconda3/2022.05
module load CUDA/12.1.1
nvcc --version

module list

source activate env_dissertation_project
which python
python -V

python test_td3_rollout.py "$@"