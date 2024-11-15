#!/usr/bin/env bash

set -x

VERSION='coarse-weight-sweep'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/train.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
WORKERS=8

for W0 in 0.001 0.01 0.1 1 10; do
    for W1 in 0.001 0.01 0.1 1 10; do
        python ./VQVAE/train.py \
            --dataset $DATA \
            --dataroot $DATAROOT \
            --datalist $DATALIST \
            --workers $WORKERS \
            --batchSize 128 \
            --imageSize 224 \
            --nz 256 \
            --nblk 2 \
            --nepoch 5 \
            --expf $EXP_DIR-$W0-$W1 \
            --manualSeed 1112 \
            --log_iter 50 \
            --visualize_iter 500 \
            --ckpt_save_epoch 1 \
            --mse_w 1.0 \
            --ffl_w 0.0 \
            --cnn_loss_w0 $W0 \
            --cnn_loss_w1 $W1
    done
done