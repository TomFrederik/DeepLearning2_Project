#!/bin/bash

N_DATA=100
WEIGHTS="../imagenet/weights/cgn.pth"
RUN_NAME="train_cf"
TRUNC=0.5
# for i in 0 217 482 491 497 566 569 571 574 701; do
#     CLASS=$i
#     python generate_data.py --mode fixed_classes --n_data $N_DATA --run_name $RUN_NAME --weights_path $WEIGHTS --classes $CLASS $CLASS $CLASS --truncation $TRUNC
# done


RUN_NAME="val_cf"
for i in 0 217 482 491 497 566 569 571 574 701; do
    CLASS=$i
    python generate_data.py --mode fixed_classes --n_data $N_DATA --run_name $RUN_NAME --weights_path $WEIGHTS --classes $CLASS $CLASS $CLASS --truncation $TRUNC
done