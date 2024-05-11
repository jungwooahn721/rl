#!/bin/bash
log_folder_1="hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_11-05-2024_09-31-00"
# log_folder_2="q2_pg_cheetah_baseline_HalfCheetah-v4_30-04-2024_10-52-56"


title="cartpole: eval return"

python aai4160/scripts/parse_tensorboard.py \
    --input_log_files data/$log_folder_1 \
    --human_readable_names "Eval_AverageReturn" \
    --data_key "Eval_AverageReturn" \
    --title "$title" \
    --x_label_name "Train Environment Steps" \
    --y_label_name "Eval Return" \
    --output_file "$title.png" \
    # --plot_mean_std