#!/bin/bash
log_folder_1="hw3_dqn_dqn_MsPacmanNoFrameskip-v0_d0.99_tu2000_lr0.0001_doubleq_clip10.0_11-05-2024_12-02-07"
naeme_1="ddqn, Pacman"

# log_folder_2="hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_11-05-2024_11-15-34"
# name_2="lr = 5e-2"

# title="average return with different learning rates"

# data_key="Eval_AverageReturn"
# data_key="critic_loss"
# data_key="q_values"
data_key="eval_return"
# data_key="train_return"
title="$data_key on MsPacmanNoFrameskip-v0 with ddqn"


python aai4160/scripts/parse_tensorboard.py \
    --input_log_files data/$log_folder_1 \
    --human_readable_names "$naeme_1" \
    --data_key "$data_key" \
    --title "$title" \
    --x_label_name "Train Environment Steps" \
    --y_label_name "$data_key" \
    --output_file "$title.png" \
    # --plot_mean_std

log_folder_1="hw3_dqn_dqn_MsPacmanNoFrameskip-v0_d0.99_tu2000_lr0.0001_doubleq_clip10.0_11-05-2024_12-02-07"
naeme_1="ddqn, Pacman"

# log_folder_2="hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_11-05-2024_11-15-34"
# name_2="lr = 5e-2"

# title="average return with different learning rates"

# data_key="Eval_AverageReturn"
# data_key="critic_loss"
# data_key="q_values"
data_key="eval_return"
# data_key="train_return"
title="train, eval return on MsPacmanNoFrameskip-v0 with ddqn"


python aai4160/scripts/parse_tensorboard.py \
    --input_log_files data/$log_folder_1 \
    --human_readable_names "$naeme_1" \
    --data_key "$data_key" "train_return" \
    --title "$title" \
    --x_label_name "Train Environment Steps" \
    --y_label_name "return" \
    --output_file "$title.png" \
    # --plot_mean_std