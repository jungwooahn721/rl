#!/bin/bash
log_folder_1="q2_pg_cheetah_baseline_HalfCheetah-v4_30-04-2024_10-52-56"
#log_folder_2="q2_pg_reacher_ppo_Reacher-v4_30-04-2024_11-18-55"

title="Baseline Loss"

python aai4160/scripts/parse_tensorboard.py \
    --input_log_files data/$log_folder_1 \
    --human_readable_names "Baseline Loss" \
    --data_key "Baseline_Loss" \
    --title "$title" \
    --x_label_name "Train Environment Steps" \
    --y_label_name "Loss" \
    --output_file "$title.png" \
    # --plot_mean_std
