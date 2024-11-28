#!/usr/bin/env bash

set -x

VERSION='dtd-model-sweep'
DATA='filelist'
DATAROOT='./datasets/dtd/images'
DATALIST='./datasets/dtd/train_list.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
WORKERS=8
W0=0.01
W1=0.01
for model in resnet152; do
    python ./VQVAE/train.py \
        --dataset $DATA \
        --dataroot $DATAROOT \
        --datalist $DATALIST \
        --workers $WORKERS \
        --batchSize 128 \
        --imageSize 224 \
        --nz 128 \
        --nblk 1 \
        --nepoch 5 \
        --expf $EXP_DIR-$model \
        --manualSeed 1112 \
        --log_iter 50 \
        --visualize_iter 500 \
        --ckpt_save_epoch 1 \
        --mse_w 1.0 \
        --ffl_w 0.0 \
        --cnn_loss_w0 $W0 \
        --cnn_loss_w1 $W1 \
        --model $model
done
