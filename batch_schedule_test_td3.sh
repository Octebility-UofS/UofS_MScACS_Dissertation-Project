schedule_test_td3 () {
    local time=$1
    local env_steps=$2
    local update_steps=$3
    local p_freq=$4
    sbatch --time=$time --output="out/td3/mixed/${env_steps}-${update_steps}-pf${p_freq}/%j_slurm_log.out" job_schedule.sh "td3/mixed/${env_steps}-${update_steps}-pf${p_freq}" test_td3.py REPLAY_ENV_STEPS=$env_steps NUM_UPDATES=$update_steps POLICY_FREQ=$p_freq
}

schedule_test_td3 04:00:00 1e1 1e1 2
schedule_test_td3 04:00:00 1e2 1e1 2
schedule_test_td3 08:00:00 1e2 1e2 2
schedule_test_td3 08:00:00 1e2 1e2 4
schedule_test_td3 08:00:00 1e2 1e2 6
schedule_test_td3 16:00:00 1e3 1e2 2
schedule_test_td3 16:00:00 1e3 1e2 4
schedule_test_td3 16:00:00 1e3 1e2 6
schedule_test_td3 96:00:00 1e3 1e3 4
schedule_test_td3 96:00:00 1e4 1e3 4
schedule_test_td3 96:00:00 1e3 1e5 4
schedule_test_td3 96:00:00 1e3 1e7 4
