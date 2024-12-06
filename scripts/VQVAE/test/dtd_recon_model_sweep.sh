#!/usr/bin/env bash

set -x

VERSION='dtd_recon_model_sweep'
DATA='filelist'
DATAROOT='./datasets/dtd/images'
DATALIST='./datasets/dtd/test_list.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
RES_DIR='./VQVAE/results/'$VERSION
WORKERS=8

for model in resnet18 resnet34 resnet50 resnet101 resnet152; do
    python ./VQVAE/test.py \
        --dataset $DATA \
        --dataroot $DATAROOT \
        --datalist $DATALIST \
        --WORKERS $WORKERS \
        --batchSize 1 \
        --imageSize 224 \
        --nz 128 \
        --nblk 1 \
        --expf $EXP_DIR-$model \
        --manualSeed 1112 \
        --epoch_test 5 \
        --eval \
        --resf $RES_DIR-$model \
        --show_input
done 