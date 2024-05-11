#!/bin/bash

# Define directories and labels for DDQN experiments
log_folder_ddqn1="hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_11-05-2024_10-19-35"
name_ddqn1="DDQN, Seed 1"

log_folder_ddqn2="hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_11-05-2024_10-20-06"
name_ddqn2="DDQN, Seed 2"

log_folder_ddqn3="hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_11-05-2024_10-20-42"
name_ddqn3="DDQN, Seed 3"

# Define directories and labels for DQN experiments
log_folder_dqn1="hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_11-05-2024_09-54-54"
name_dqn1="DQN, Seed 1"

log_folder_dqn2="hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_11-05-2024_09-55-22"
name_dqn2="DQN, Seed 2"

log_folder_dqn3="hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_11-05-2024_09-55-38"
name_dqn3="DQN, Seed 3"

# Configuration for plot
data_key="Eval_AverageReturn"
title="$data_key comparison between DQN and DDQN on LunarLander-v2"

# Run the Python script to generate the plot
python aai4160/scripts/parse_tensorboard.py \
    --input_log_files data/$log_folder_dqn1 data/$log_folder_dqn2 data/$log_folder_dqn3 data/$log_folder_ddqn1 data/$log_folder_ddqn2 data/$log_folder_ddqn3 \
    --human_readable_names "$name_dqn1" "$name_dqn2" "$name_dqn3" "$name_ddqn1" "$name_ddqn2" "$name_ddqn3" \
    --colors "blue" "blue" "blue" "red" "red" "red" \
    --data_key "$data_key" \
    --title "$title" \
    --x_label_name "Train Environment Steps" \
    --y_label_name "$data_key" \
    --output_file "${title// /_}.png" \
    #--plot_mean_std
