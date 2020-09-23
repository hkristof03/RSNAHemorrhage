#!/bin/bash
FILES=./input2/cq500-dst/stage_1_train_images/*.dcm
# FILES=./input2/cq500-tmp/*.dcm
i=1
sp="/-\|"
echo -n ' '
for f in $FILES; do
    echo -ne "Decoding DICOM pixel data \033[0K\r"
    printf "\b${sp:i++%${#sp}:1}"
    gdcmconv --raw $f $f
done