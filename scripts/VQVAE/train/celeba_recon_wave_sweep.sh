#!/usr/bin/env bash

set -x

VERSION='celeba_recon_wave_sweep'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/train.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
WORKER=8

for W in 0.001 0.01 0.1 1 10 100; do
    python ./VQVAE/train.py \
        --dataset $DATA \
        --dataroot $DATAROOT \
        --datalist $DATALIST \
        --workers $WORKER \
        --batchSize 128 \
        --imageSize 224 \
        --nz 128 \
        --nblk 1 \
        --nepoch 5 \
        --expf $EXP_DIR-$W \
        --manualSeed 1112 \
        --log_iter 50 \
        --visualize_iter 500 \
        --ckpt_save_epoch 1 \
        --mse_w 1.0 \
        --ffl_w 0.0 \
        --wavelet_w 1.0
done