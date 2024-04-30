#!/bin/bash
log_folder_1="q2_pg_cheetah_HalfCheetah-v4_30-04-2024_10-52-33"
log_folder_2="q2_pg_cheetah_baseline_HalfCheetah-v4_30-04-2024_10-52-56"
log_folder_3="q2_pg_cheetah_baseline_blr0.005_HalfCheetah-v4_30-04-2024_10-58-20"
log_folder_4="q2_pg_cheetah_baseline_bgs2_HalfCheetah-v4_30-04-2024_10-58-31"

title="with, without baseline + lower learning rate, gradient step"

python aai4160/scripts/parse_tensorboard.py \
    --input_log_files data/$log_folder_1 data/$log_folder_2 data/$log_folder_3 data/$log_folder_4 \
    --human_readable_names "w/o baseline" "w/ baseline" "w/ baseline, blr: 0.005" " w/ baseline, bgs: 2" \
    --data_key "Eval_AverageReturn" \
    --title "$title" \
    --x_label_name "Train Environment Steps" \
    --y_label_name "Eval Return" \
    --output_file "$title.png" \
    # --plot_mean_std
