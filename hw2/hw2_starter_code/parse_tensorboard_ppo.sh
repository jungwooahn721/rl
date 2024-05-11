#!/bin/bash
log_folder_1="q2_pg_reacher_Reacher-v4_30-04-2024_11-01-20"
log_folder_2="q2_pg_reacher_ppo_Reacher-v4_30-04-2024_11-18-55"

title="with, without PPO"

python aai4160/scripts/parse_tensorboard.py \
    --input_log_files data/$log_folder_1 data/$log_folder_2 \
    --human_readable_names "w/o PPO" "w/ PPO" \
    --data_key "Eval_AverageReturn" \
    --title "$title" \
    --x_label_name "Train Environment Steps" \
    --y_label_name "Eval Return" \
    --output_file "$title.png" \
    # --plot_mean_std
