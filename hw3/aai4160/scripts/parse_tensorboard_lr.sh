#!/bin/bash
log_folder_1="hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_11-05-2024_09-31-00"
naeme_1="lr = 1e-3 (default)"

log_folder_2="hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_11-05-2024_11-15-34"
name_2="lr = 5e-2"

# title="average return with different learning rates"

# data_key="Eval_AverageReturn"
# data_key="critic_loss"
# data_key="q_values"
data_key="Eval_AverageReturn"
title="$data_key with different learning rates"


python aai4160/scripts/parse_tensorboard.py \
    --input_log_files data/$log_folder_1 data/$log_folder_2\
    --human_readable_names "$naeme_1" "$name_2" \
    --data_key "$data_key" \
    --title "$title" \
    --x_label_name "Train Environment Steps" \
    --y_label_name "$data_key" \
    --output_file "$title.png" \
    # --plot_mean_std