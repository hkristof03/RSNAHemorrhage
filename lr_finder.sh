#!/usr/bin/env bash

###########################################
# Batch size
tr_batch_size=48

models=(
    rsna-efficientnet-b0_b_0
)

for i in "${!models[@]}"
do
    model=${models[$i]}
    IFS=_ read net_model data_dataset data_fold <<< "$model"
    python lr_finder.py --n_jobs=16 --dev_enabled=0 \
    --data_dataset=${data_dataset} --data_fold=${data_fold} --data_input_folder=./input/preprocessed/stage_1_train_images \
    --data_train_transform='train_intermediate.json' \
    --net_model=${net_model} --net_pretrained=1 \
    --optim=radam --optim_lr=0.000001 --optim_weight_decay=0.01 \

done

#    --net_model=${net_model} --net_pretrained=1 \
#    --optim=sgd --optim_lr=0.01 --optim_weight_decay=0.0001 --optim_swa_enabled=1 \
#    --tr_transform=train_complex.json --vl_transform=valid_basic.json \
#     --cp_enabled=1 --cp_metric_name=val_output_loss --cp_minimize=1 \
# --optim=radam --optim_lr=0.0001 --optim_weight_decay=0 --optim_lookahead_enabled=0 \