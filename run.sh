#! /usr/bin/env bash
SAVE_FOLDER="/netscratch/xie/normal_weights/no_rotation/"
LOG_FOLDER="/netscratch/xie/logs/"
BACKBONE_FOLDER="/netscratch/xie/weights/"
SAVE_INTERVAL=4738

pip install webdataset
python3 train.py --config=NormalNet_base_config \
--dataset=dataset_server \
--batch_size=16 \
--save_folder=$SAVE_FOLDER \
--log_folder=$LOG_FOLDER \
--backbone_folder=$BACKBONE_FOLDER \
--save_interval=$SAVE_INTERVAL \
--validation_epoch=2 \
--reproductablity
