#!/bin/bash
#SBATCH --partition=gpu --qos=gpu --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/log/%j.out
#SBATCH --mail-type=ALL --mail-user=mlobillich1@sheffield.ac.uk

module load Anaconda3/2022.05
module load CUDA/12.1.1
nvcc --version

module list

echo "JOB ID: $SLURM_JOB_ID"

source activate env_dissertation_project
python $1 $SLURM_JOB_ID