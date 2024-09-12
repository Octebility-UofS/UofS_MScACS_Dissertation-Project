schedule_test_td3 () {
    local time=$1
    local env_steps=$2
    local update_steps=$3
    local p_freq=$4
    local environments=$5
    local batch_size=$6
    sbatch --time=$time --output="out/td3/mixed/${env_steps}-${update_steps}-pf${p_freq}/%j_slurm_log.out" \
    job_schedule.sh "td3/mixed/${env_steps}-${update_steps}-pf${p_freq}" test_td3.py \
    REPLAY_ENV_STEPS=$env_steps NUM_UPDATES=$update_steps POLICY_FREQ=$p_freq NUM_ENVS=$environments BATCH_SIZE=$batch_size
}

schedule_test_td3 12:00:00 1e1 1e1 2 256 256
schedule_test_td3 12:00:00 1e2 1e1 2 256 1024

schedule_test_td3 24:00:00 1e2 1e2 2 256 2048
schedule_test_td3 24:00:00 1e2 1e2 4 256 2048
schedule_test_td3 24:00:00 1e2 1e2 6 256 2048

schedule_test_td3 96:00:00 1e3 1e2 2 256 8192
schedule_test_td3 96:00:00 1e3 1e2 4 256 8192
schedule_test_td3 96:00:00 1e3 1e2 6 256 8192

schedule_test_td3 96:00:00 1e3 1e3 2 256 8192
schedule_test_td3 96:00:00 1e3 1e4 2 256 8192
schedule_test_td3 96:00:00 1e3 1e5 2 256 8192

schedule_test_td3 96:00:00 1e4 1e3 2 256 16384
schedule_test_td3 96:00:00 1e4 1e4 2 256 16384
