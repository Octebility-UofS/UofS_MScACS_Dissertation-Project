#!/bin/bash
#SBATCH --partition=gpu --qos=gpu --gres=gpu:a100:1
#SBATCH --mem=8G
#SBATCH --output=out/log/%j.out
#SBATCH --mail-type=ALL --mail-user=mlobillich1@sheffield.ac.uk

module load Anaconda3/2022.05
module load CUDA/11.8.0
nvcc --version

source activate env_dissertation_project
python $1 %j