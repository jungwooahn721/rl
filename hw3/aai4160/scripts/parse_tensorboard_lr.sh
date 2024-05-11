#!/bin/bash
log_folder_1="hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_11-05-2024_15-14-28"
name_1="lr: 5e-4"

log_folder_2="hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_11-05-2024_15-48-22"
name_2="lr: 1e-3"

log_folder_3="hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_11-05-2024_15-14-37"
name_3="lr: 5e-3"

log_folder_4="hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_11-05-2024_15-14-43"
name_4="lr: e-2"

log_folder_5="hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_11-05-2024_15-48-36"
name_5="lr: 5e-2"

# title="average return with different learning rates"

# data_key="Eval_AverageReturn"
# data_key="critic_loss"
# data_key="q_values"
data_key="Eval_AverageReturn"
# title="$data_key with different learning rates"
title="Return value with different learning rates"


python aai4160/scripts/parse_tensorboard.py \
    --input_log_files data/$log_folder_1 data/$log_folder_2 data/$log_folder_3 data/$log_folder_4 data/$log_folder_5 \
    --human_readable_names "$name_1" "$name_2" "$name_3" "$name_4" "$name_5" \
    --data_key "$data_key" \
    --title "$title" \
    --x_label_name "Train Environment Steps" \
    --y_label_name "$data_key" \
    --output_file "$title.png" \
    # --plot_mean_std