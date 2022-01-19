#! /usr/bin/env bash
SAVE_FOLDER="/netscratch/xie/normal_weights/"
LOG_FOLDER="/netscratch/xie/logs/"
BACKBONE_FOLDER="/netscratch/xie/weights/"

pip install webdataset
python3 eval.py --config=NormalNet_base_config \
--dataset=dataset_server \
--trained_model=/netscratch/xie/normal_weights/NormalNet_base_19_114000.pth