#!/usr/bin/env bash

###########################################
# Batch size
tr_batch_size=48

exp_id=513

models=(
    # rsna-resnet-18_b_0
    # rsna-efficientnet-b0_b_0
    # rsna-efficientnet-b5_b_0

    # rsna-seresnext-50.32.4d_b_0
    rsna-efficientnet-b2_c_0
)

for i in "${!models[@]}"
do
    model=${models[$i]}
    IFS=_ read net_model data_dataset data_fold <<< "$model"
    python predict.py --n_jobs=16 --dev_enabled=0 \
    --pr_dataframe=./input/preprocessed/stage_1_test.csv --data_input_folder=./input2/stage_1_test_images \
    --data_test_transform='test_intermediate_tta_410_norm.json' \
    --net_model=${net_model} --net_weight_file=./output/results/${exp_id}/ds_c__fold_0__best_log_loss__iter_141999__0_827866.pt \
    --pr_tta=5 --tr_batch_size=${tr_batch_size}

done

#    --net_model=${net_model} --net_pretrained=1 \
#    --optim=sgd --optim_lr=0.01 --optim_weight_decay=0.0001 --optim_swa_enabled=1 \
#    --tr_transform=train_complex.json --vl_transform=valid_basic.json \
#     --cp_enabled=1 --cp_metric_name=val_output_loss --cp_minimize=1 \
#
#     predict val set
#     --pr_dataframe=./input/datasets/b/dataset_b_fold_0.csv --data_input_folder=./input/preprocessed/stage_1_train_images \
#