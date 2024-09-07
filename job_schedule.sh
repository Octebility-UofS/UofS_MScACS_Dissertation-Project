#!/bin/bash
#SBATCH --partition=gpu --qos=gpu --gres=gpu:a100:1
#SBATCH --mem=90G
#SBATCH --time=96:00:00
#SBATCH --output=/dev/null
#SBATCH --mail-type=ALL --mail-user=mlobillich1@sheffield.ac.uk

out_path=$1
slurm_log="out/${out_path}/${SLURM_JOB_ID}_slurm_log.out"
script_log="out/${out_path}/${SLURM_JOB_ID}_log.out"

echo "$@" >> $slurm_log 2>&1
echo "JOB ID: $SLURM_JOB_ID" >> $slurm_log 2>&1

shift
script=$1
shift

module load Anaconda3/2022.05
module load CUDA/12.1.1
nvcc --version >> $slurm_log 2>&1

module list >> $slurm_log 2>&1

source activate env_dissertation_project
which python >> $slurm_log 2>&1
python -V >> $slurm_log 2>&1

python $script "OUT_PATH=${out_path}" JOB_ID=$SLURM_JOB_ID "$@" >> $script_log 2>&1
