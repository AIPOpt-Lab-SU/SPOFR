#!/usr/bin/env bash

a=${1}
b=${2}
c=${3}
d=${4}
e=${5}
f=${6}
g=${7}
h=${8}
i=${9}
j=${10}
k=${11}
l=${12}
m=${13}
n=${14}
o=${15}
p=${16}
q=${17}
r=${18}
s=${19}
t=${20}
u=${21}
v=${22}
w=${23}

source ~/.bashrc
conda activate pyenv1
python run_hyperparams.py --lambda_group_fairness ${a} --epochs ${b} --lr ${c} --hidden_layer ${d} --optimizer ${e} --dropout ${f} --partial_train_data ${g}  --partial_val_data ${h} --full_test_data ${i}    --log_dir runs/default_JK  --gme_new ${j} --sample_size ${k} --batch_size ${l} --soft_train ${m} --index ${n}  --allow_unfairness ${o} --fairness_gap ${p} --embed_groups ${q} --multi_groups ${r} --entreg_decay ${s} --evaluate_interval ${t} --output_tag ${u} --disparity_type ${v} --indicator_type square --reward_type dcg > logs/run_${n}.log #--seed ${w} > logs/run_${n}.log
