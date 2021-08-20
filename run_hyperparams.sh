#!/usr/bin/env bash

a=${1}
b=${2}
c=${3}
d=${4}
e=${5}
f=${6}
g=${7}
h=${8}


source ~/.bashrc

conda activate testenv3

python run_hyperparams.py  "[0.1]" --epochs 100 --partial_train_data "/home/jkotary/fultr/transformed_datasets/german/Train/partial_train_5k.pkl"  --partial_val_data "/home/jkotary/fultr/transformed_datasets/german/Train/partial_valid_5k.pkl" --full_test_data "/home/jkotary/fultr/transformed_datasets/german/full/test.pkl"    --log_dir "runs/default_JK"  --sample_size 32 --batch_size 32 --soft_train 1 --gpu > logs/run_${h}.log
#${a} --batch_size ${b} --sample_size ${c} --lr ${d} --DSM ${e} --DSMfair ${f} --random_seed ${f} --hidden_layer ${g} --index ${h} > logs/log_${h}.txt
