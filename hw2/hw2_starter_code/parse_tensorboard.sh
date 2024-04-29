#!/bin/bash
log_folder_1="q2_pg_cheetah_HalfCheetah-v4_29-04-2024_16-01-36"
log_folder_2="q2_pg_lunar_lander_lambda0.95_LunarLander-v2_29-04-2024_16-38-24"
log_folder_3="q2_pg_lunar_lander_lambda0.98_LunarLander-v2_29-04-2024_17-11-55"
log_folder_4="q2_pg_lunar_lander_lambda0.99_LunarLander-v2_29-04-2024_17-40-10"

title="w/, w/o Baseline Comparison on HalfCheetah-v4"

python aai4160/scripts/parse_tensorboard.py \
    --input_log_files data/$log_folder_1 data/$log_folder_2 data/$log_folder_3 data/$log_folder_4 \
    --human_readable_names "Vanilla" "Baseline" \
    --data_key "Eval_AverageReturn" \
    --title "$title" \
    --x_label_name "Train Environment Steps" \
    --y_label_name "Eval Return" \
    --output_file "$title.png" \
    # --plot_mean_std
