#!/bin/bash
log_folder_1="q2_pg_cartpole_lb_CartPole-v0_29-04-2024_15-41-08"
log_folder_2="q2_pg_cartpole_lb_rtg_CartPole-v0_29-04-2024_15-41-43"
log_folder_3="q2_pg_cartpole_lb_na_CartPole-v0_29-04-2024_15-42-13"
log_folder_4="q2_pg_cartpole_lb_rtg_na_CartPole-v0_29-04-2024_15-42-39"

title="Learning Curves (large batch experiments)"

python aai4160/scripts/parse_tensorboard.py \
    --input_log_files data/$log_folder_1 data/$log_folder_2 data/$log_folder_3 data/$log_folder_4 \
    --human_readable_names "Vanilla" "RewToGo" "NormAdv" "RewToGo+NormAdv" \
    --data_key "Eval_AverageReturn" \
    --title "Learning Curves (small batch experiments)" \
    --x_label_name "Train Environment Steps" \
    --y_label_name "Eval Return" \
    --output_file "$title.png" \
    # --plot_mean_std
