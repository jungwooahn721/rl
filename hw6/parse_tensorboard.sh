python aai4160/scripts/parse_tensorboard.py \
--input_log_files data/hw5_cql_alpha_0.0_expert_PointmassHard-v0_08-06-2024_14-02-56 \
data/hw5_cql_alpha_0.1_expert_PointmassHard-v0_08-06-2024_14-02-55 \
--human_readable_names "α = 0.0" "α = 0.1" \
--data_key "Overestimation" \
--title "Overestimation of Q-values" \
--x_label_name "Iterations" \
--y_label_name "Overestimation" \
--output_file "./4_overestimation.png"