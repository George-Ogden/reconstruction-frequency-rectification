#!/usr/bin/env bash

set -x

VERSION='model-sweep'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/test.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
RES_DIR='./VQVAE/results/'$VERSION
WORKER=8
W0=0.01
W1=0.01
for model in resnet18 resnet34 resnet50 resnet101 resnet152; do
    python ./VQVAE/test.py \
        --dataset $DATA \
        --dataroot $DATAROOT \
        --datalist $DATALIST \
        --workers $WORKER \
        --batchSize 1 \
        --imageSize 224 \
        --nz 128 \
        --nblk 1 \
        --expf $EXP_DIR-$model-$W0-$W1 \
        --manualSeed 1112 \
        --epoch_test 5 \
        --eval \
        --resf $RES_DIR-$model-$W0-$W1 \
        --show_input \
        --model $model
done