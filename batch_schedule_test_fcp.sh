sbatch --time=02:00:00 job_schedule.sh test_fcp.py ENV_STEPS=1e2 NUM_UPDATES=1e2 NUM_CHECKPOINTS=10
sbatch --time=04:00:00 job_schedule.sh test_fcp.py ENV_STEPS=1e3 NUM_UPDATES=1e2 NUM_CHECKPOINTS=10
sbatch --time=12:00:00 job_schedule.sh test_fcp.py ENV_STEPS=1e3 NUM_UPDATES=1e3 NUM_CHECKPOINTS=10
sbatch --time=24:00:00 job_schedule.sh test_fcp.py ENV_STEPS=1e4 NUM_UPDATES=1e3 NUM_CHECKPOINTS=10
sbatch --time=48:00:00 job_schedule.sh test_fcp.py ENV_STEPS=1e4 NUM_UPDATES=1e4 NUM_CHECKPOINTS=10
sbatch --time=96:00:00 job_schedule.sh test_fcp.py ENV_STEPS=1e5 NUM_UPDATES=1e4 NUM_CHECKPOINTS=10
sbatch --time=96:00:00 job_schedule.sh test_fcp.py ENV_STEPS=1e5 NUM_UPDATES=1e3 NUM_CHECKPOINTS=10
sbatch --time=96:00:00 job_schedule.sh test_fcp.py ENV_STEPS=1e3 NUM_UPDATES=1e5 NUM_CHECKPOINTS=10