#! /usr/bin/env bash
SAVE_FOLDER="/netscratch/xie/normal_weights/"
LOG_FOLDER="/netscratch/xie/logs/"
BACKBONE_FOLDER="/netscratch/xie/weights/"
SAVE_INTERVAL=5700

python3 train.py --config=NormalNet_base_config \
--batch_size=16 \
--save_folder=$SAVE_FOLDER \
--log_folder=$LOG_FOLDER \
--backbone_folder=$BACKBONE_FOLDER \
--save_interval=$SAVE_INTERVAL \
