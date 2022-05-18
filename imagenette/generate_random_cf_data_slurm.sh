#!/bin/bash

N_DATA=10000
WEIGHTS="../imagenet/weights/cgn.pth"
TRUNC=1

SUP_NAME="random"

RUN_NAME="$SUP_NAME/train_cf"
python generate_data.py --mode random --n_data $N_DATA_PER_CLASS --run_name $RUN_NAME --weights_path $WEIGHTS --truncation $TRUNC

RUN_NAME="$SUP_NAME/val_cf"
python generate_data.py --mode random --n_data $N_DATA_PER_CLASS --run_name $RUN_NAME --weights_path $WEIGHTS --truncation $TRUNC