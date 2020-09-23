#!/usr/bin/env bash

###########################################
# Batch size
tr_batch_size=32

# EXP id
exp_id=304

models=(
    # rsna-resnet-18_b_0
    rsna-efficientnet-b0_b_0
)

for i in "${!models[@]}"
do
    model=${models[$i]}
    IFS=_ read net_model data_dataset data_fold <<< "$model"
    python predict_any.py --n_jobs=8 --dev_enabled=0 \
    --pr_dataframe=./input/preprocessed/stage_1_test.csv --data_input_folder=./input/preprocessed/stage_1_test_images \
    --net_model=${net_model} --net_weight_file=./output/results/${exp_id}/ds_b__fold_0__best_in_stage_7.pt \
    --tr_use_amp=0 --tr_batch_size=${tr_batch_size} --net_num_classes=1

done

#    --net_model=${net_model} --net_pretrained=1 \
#    --optim=sgd --optim_lr=0.01 --optim_weight_decay=0.0001 --optim_swa_enabled=1 \
#    --tr_transform=train_complex.json --vl_transform=valid_basic.json \
#     --cp_enabled=1 --cp_metric_name=val_output_loss --cp_minimize=1 \
#
#     predict val set
#     --pr_dataframe=./input/datasets/b/dataset_b_fold_0.csv --data_input_folder=./input/preprocessed/stage_1_train_images \
#