#!/usr/bin/env bash

###########################################
# Batch size
tr_batch_size=64

# num of samples: 608517 (b dataset)
# ~ 20 epochs = 19016 * 20
tr_iteration_num=380500
# tr_iteration_num=95500

# Validation after # of iterations
tr_iteration_valid=500

models=(
    # rsna-resnet-18_b_0
    rsna-efficientnet-b0_b_0
)

for i in "${!models[@]}"
do
    model=${models[$i]}
    IFS=_ read net_model data_dataset data_fold <<< "$model"
    python train_any.py --n_jobs=16 --nml_enabled=0 --dev_enabled=1 \
    --data_dataset=${data_dataset} --data_fold=${data_fold} --data_input_folder=./input2/preprocessed_b/stage_1_train_images \
    --data_train_transform='train_intermediate.json' \
    --net_model=${net_model} --net_pretrained=1 --net_num_classes=1 \
    --tr_use_amp=0 --tr_iteration_num=${tr_iteration_num} --tr_batch_size=${tr_batch_size} --tr_accumulation_step=1 \
    --tr_iteration_valid=${tr_iteration_valid} \
    --optim=radam --optim_lr=0.001 --optim_weight_decay=0 \
    --optim_scheduler=warmup_cosine --optim_scheduler_max_lr=0.001 --optim_scheduler_min_lr=0.000001 --optim_scheduler_warmup=0.1

done

#    --net_model=${net_model} --net_pretrained=1 \
#    --optim=sgd --optim_lr=0.01 --optim_weight_decay=0.0001 --optim_swa_enabled=1 \
#    --tr_transform=train_complex.json --vl_transform=valid_basic.json \
#     --cp_enabled=1 --cp_metric_name=val_output_loss --cp_minimize=1 \
# --optim=radam --optim_lr=0.0001 --optim_weight_decay=0 --optim_lookahead_enabled=0 \
#     --net_weight_file=./output/results/174/ds_b__fold_0__best_in_stage_15.pt \
# --optim_scheduler=warmup_cosine --optim_scheduler_max_lr=0.00005 --optim_scheduler_warmup=0.025 \
#     --cb_telemetry_enabled=1 --cb_telemetry_module_name=backbone._blocks