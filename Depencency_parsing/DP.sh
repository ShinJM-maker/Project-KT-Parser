#!/bin/bash

set +x

OUTPUT_DIR="model_output"
DATA_DIR="data/klue_benchmark" 
VERSION="v1.1"


# KLUE-DP
task="dp"


python dp_main.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/klue-dp-${VERSION}  --model_name_or_path monologg/koelectra-base-v3-discriminator --learning_rate 5e-5 --num_train_epochs 5 --warmup_ratio 0.1 --train_batch_size 16 --patience 10000 --max_seq_length 510 --metric_key train/loss --gpus 0 --num_workers 4
