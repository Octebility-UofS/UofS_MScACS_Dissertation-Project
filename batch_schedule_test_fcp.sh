#!/bin/bash

schedule_test_fcp () {
    local time=$1
    local env_steps=$2
    local update_steps=$3
    sbatch --time=$time --output="out/fcp/mixed/${env_steps}-${update_steps}/%j_slurm_log.out" job_schedule.sh "fcp/mixed/${env_steps}-${update_steps}" test_fcp.py ENV_STEPS=$env_steps NUM_UPDATES=$update_steps NUM_CHECKPOINTS=10
}


schedule_test_fcp 10:00:00 1e1 1e1
schedule_test_fcp 20:00:00 1e2 1e1
schedule_test_fcp 48:00:00 1e2 1e2
schedule_test_fcp 48:00:00 1e3 1e2
schedule_test_fcp 48:00:00 1e3 1e3
schedule_test_fcp 96:00:00 1e4 1e3
schedule_test_fcp 96:00:00 1e4 1e4
schedule_test_fcp 96:00:00 1e5 1e4
schedule_test_fcp 96:00:00 1e5 1e3
schedule_test_fcp 96:00:00 1e3 1e5