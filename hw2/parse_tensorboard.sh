#!/bin/bash
log_folder_1="/root/rl/hw2/hw2_starter_code/data/q2_pg_cartpole_CartPole-v0_29-04-2024_15-23-57"
log_folder_2="/root/rl/hw2/hw2_starter_code/data/q2_pg_cartpole_lb_CartPole-v0_29-04-2024_15-41-08"

python /root/rl/hw2/hw2_starter_code/aai4160/scripts/parse_tensorboard.py \
    --input_log_files $log_folder_1 $log_folder_2 \
    --human_readable_names "Vanilla" "Reward to go" \
    --data_key "Eval_AverageReturn" \
    --title "Your Title Here" \
    --x_label_name "Train Environment Steps" \
    --y_label_name "Eval Return" \
    --output_file "output_plot.png" \
    --plot_mean_std
