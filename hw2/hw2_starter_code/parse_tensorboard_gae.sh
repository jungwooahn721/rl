#!/bin/bash
log_folder_1="q2_pg_lunar_lander_lambda0_LunarLander-v2_30-04-2024_10-59-51"
log_folder_2="q2_pg_lunar_lander_lambda0.95_LunarLander-v2_30-04-2024_11-22-01"
log_folder_3="q2_pg_lunar_lander_lambda0.98_LunarLander-v2_30-04-2024_11-49-00"
log_folder_4="q2_pg_lunar_lander_lambda0.99_LunarLander-v2_30-04-2024_11-49-09"
log_folder_5="q2_pg_lunar_lander_lambda1_LunarLander-v2_30-04-2024_11-49-16"

title="GAE Result with different λ"

python aai4160/scripts/parse_tensorboard.py \
    --input_log_files data/$log_folder_1 data/$log_folder_2 data/$log_folder_3 data/$log_folder_4 data/$log_folder_5 \
    --human_readable_names "lambda: 0" "lambda: 0.95" "lambda: 0.98" "lambda: 0.99" "lambda: 1" \
    --data_key "Eval_AverageReturn" \
    --title "$title" \
    --x_label_name "Train Environment Steps" \
    --y_label_name "Eval Return" \
    --output_file "$title.png" \
    # --plot_mean_std
