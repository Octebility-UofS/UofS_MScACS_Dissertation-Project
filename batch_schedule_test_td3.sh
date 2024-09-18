schedule_test_td3 () {
    local time=$1
    local explore_steps=$2
    local env_steps=$3
    local environments=$4
    local batch_size=$5
    local p_freq=$6
    sbatch --time=$time --output="out/td3/gpu/${env_steps}-${explore_steps}-pf${p_freq}/%j_slurm_log.out" \
    job_schedule.sh "td3/gpu/${env_steps}-${explore_steps}-pf${p_freq}" test_td3.py \
    TOTAL_STEPS=$env_steps EXPLORATION_STEPS=$explore_steps NUM_ENVS=$environments BATCH_SIZE=$batch_size POLICY_FREQ=$p_freq
}

schedule_test_td3 01:00:00 4e1 8e2 8 64 2
schedule_test_td3 01:00:00 8e1 16e2 8 64 2
schedule_test_td3 01:00:00 16e1 32e2 8 64 2
schedule_test_td3 01:00:00 8e1 8e4 16 128 2
schedule_test_td3 01:00:00 16e1 16e4 32 256 2
schedule_test_td3 01:00:00 16e3 4e6 128 512 2

schedule_test_td3 06:00:00 16e2 16e6 16 4096 2
schedule_test_td3 12:00:00 16e2 16e7 16 4096 2
schedule_test_td3 48:00:00 16e2 32e7 16 4096 2
schedule_test_td3 96:00:00 16e2 64e7 16 4096 2
