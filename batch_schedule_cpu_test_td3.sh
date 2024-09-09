schedule_test_td3 () {
    local time=$1
    local env_steps=$2
    local update_steps=$3
    local p_freq=$4
    sbatch --time=$time --output="out/td3/cpu/${env_steps}-${update_steps}/%j_slurm_log.out" job_schedule.sh "td3/cpu/${env_steps}-${update_steps}" test_td3.py REPLAY_ENV_STEPS=$env_steps NUM_UPDATES=$update_steps POLICY_FREQ=$p_freq BACKEND=cpu
}

schedule_test_td3 12:00:00 1e1 1e1 2
schedule_test_td3 12:00:00 1e2 1e1 2
schedule_test_td3 48:00:00 1e2 1e2 2
schedule_test_td3 48:00:00 1e2 1e2 4
schedule_test_td3 48:00:00 1e2 1e2 6
schedule_test_td3 96:00:00 1e3 1e2 2
schedule_test_td3 96:00:00 1e3 1e2 4
schedule_test_td3 96:00:00 1e3 1e2 6
schedule_test_td3 96:00:00 1e3 1e3 4
schedule_test_td3 96:00:00 1e4 1e3 4
schedule_test_td3 96:00:00 1e3 1e5 4
schedule_test_td3 96:00:00 1e3 1e7 4
