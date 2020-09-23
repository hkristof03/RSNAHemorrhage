#!/usr/bin/env bash

###########################################
# Batch size
tr_batch_size=30

# num of samples: 608517 (b dataset)
# num of (train) samples: 532356 (c dataset)
# num of (train) samples: 625858 (c#2 dataset)
# num of (train) samples: 647666 (c#2 + cq500)
# ~ 20 epochs = 19016 * 20 (batch size=32)
# tr_iteration_num=380500

# ~ 30 epochs, batch size: 64
# tr_iteration_num=285000

# ~ 20 epochs, batch size 64
# tr_iteration_num=190000

# ~ 5 epochs, batch size 64
# tr_iteration_num=95000

# ~5 epochs, batch size 24
tr_iteration_num=416500

# tr_iteration_num=209000

# Validation after # of iterations
tr_iteration_valid=10exi00

models=(
    # rsna-resnet-18_b_0
    # rsna-efficientnet-b0_b_0 c dataset!!
    # rsna-efficientnet-b5_b_0 c dataset!!

    # SeResnexts
    # rsna-seresnext-50.32.4d_c_0
    rsna-seresnext-50.32.4d_c_1

    # EfficientNets
    # rsna-efficientnet-b2_c_0
)

for i in "${!models[@]}"
do
    model=${models[$i]}
    IFS=_ read net_model data_dataset data_fold <<< "$model"
    python train.py --n_jobs=16 --nml_enabled=1 --dev_enabled=0 \
    --data_dataset=${data_dataset} --data_fold=${data_fold} --data_input_folder=./input2/stage_1_train_images \
    --data_img_size=410 --data_train_transform='train_complex_410_norm.json' --data_valid_transform='valid_simple_410_norm.json' \
    --net_model=${net_model} --net_pretrained=1 --net_num_classes=6 \
    --tr_use_amp=1 --tr_amp_level=O2 --tr_iteration_num=${tr_iteration_num} --tr_batch_size=${tr_batch_size} --tr_accumulation_step=2 \
    --tr_iteration_valid=${tr_iteration_valid} \
    --optim=radam --optim_lr=0.0001 --optim_weight_decay=0 \
    --optim_scheduler=one_cycle_lr --optim_scheduler_max_lr=0.0001 --optim_scheduler_pct_start=0.3



done

#    --net_model=${net_model} --net_pretrained=1 \
#    --optim=sgd --optim_lr=0.01 --optim_weight_decay=0.0001 --optim_swa_enabled=1 \
#    --tr_transform=train_complex.json --vl_transform=valid_basic.json \
#     --cp_enabled=1 --cp_metric_name=val_output_loss --cp_minimize=1 \
# --optim=radam --optim_lr=0.0001 --optim_weight_decay=0 --optim_lookahead_enabled=0 \
#  --data_sampler=pos_range
#--data_valid_transform='validate_simple.json' --data_sampler=pos_range
# --optim_scheduler=warmup_cosine --optim_scheduler_max_lr=0.00006 --optim_scheduler_min_lr=0.0000005 --optim_scheduler_warmup=0.05
# --optim_scheduler=multistep --optim_scheduler_gamma=0.66667
# --optim_scheduler=one_cycle_lr --optim_scheduler_max_lr=0.00005 --optim_scheduler_pct_start=0.1
# --optim_scheduler=multistep --optim_scheduler_gamma=0.66667
# --optim_scheduler=one_cycle_lr --optim_scheduler_max_lr=0.001 --optim_scheduler_pct_start=0.3
