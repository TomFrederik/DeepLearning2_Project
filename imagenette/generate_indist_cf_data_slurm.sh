#!/bin/bash

N_DATA_PER_CLASS=1000
WEIGHTS="../imagenet/weights/cgn.pth"
TRUNC=1

SUP_NAME="indist"

RUN_NAME="$SUP_NAME/train_cf"
for i in 0 217 482 491 497 566 569 571 574 701; do
    CLASS=$i
    python generate_data.py --mode fixed_classes --n_data $N_DATA_PER_CLASS --run_name $RUN_NAME --weights_path $WEIGHTS --classes $CLASS $CLASS $CLASS --truncation $TRUNC
done


RUN_NAME="$SUP_NAME/val_cf"
for i in 0 217 482 491 497 566 569 571 574 701; do
    CLASS=$i
    python generate_data.py --mode fixed_classes --n_data $N_DATA_PER_CLASS --run_name $RUN_NAME --weights_path $WEIGHTS --classes $CLASS $CLASS $CLASS --truncation $TRUNC
done