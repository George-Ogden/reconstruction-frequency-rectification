#!/usr/bin/env bash

set -x

VERSION='coarse-weight-sweep'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/train.txt'
EXP_DIR='./VanillaAE/experiments/'$VERSION
WORKER=8

for W0 in 1e-2 1e-1 1 100; do
    for W1 in 1e-3 1e-2 1e-1 1 10; do
        echo "$W0 $W1"
    done
done | parallel --colsep ' ' -j $WORKER --dry-run "python ./VanillaAE/train.py \
    --no_cuda \
    --dataset $DATA \
    --dataroot $DATAROOT \
    --datalist $DATALIST \
    --workers 1 \
    --batchSize 128 \
    --imageSize 224 \
    --nz 256 \
    --nblk 2 \
    --nepoch 20 \
    --expf $EXP_DIR-{1}-{2} \
    --manualSeed 1112 \
    --log_iter 50 \
    --visualize_iter 500 \
    --ckpt_save_epoch 1 \
    --mse_w 1.0 \
    --ffl_w 1.0 \
    --cnn_loss_w0 {1} \
    --cnn_loss_w1 {2}"
