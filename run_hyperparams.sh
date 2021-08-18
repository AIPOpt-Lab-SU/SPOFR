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

python run_hyperparams.py ${a} --batch_size ${b} --sample_size ${c} --lr ${d} --DSM ${e} --DSMfair ${f} --random_seed ${f} --hidden_layer ${g} --index ${h} > logs/log_${h}.txt
