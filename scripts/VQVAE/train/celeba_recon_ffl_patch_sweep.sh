#!/usr/bin/env bash

set -x

VERSION='celeba_recon_ffl_patch_sweep'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/train.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
WORKER=8

for P in 1 2 4 8 16 32; do
    python3 ./VQVAE/train.py \
        --dataset $DATA \
        --dataroot $DATAROOT \
        --datalist $DATALIST \
        --workers $WORKER \
        --batchSize 128 \
        --imageSize 224 \
        --nz 128 \
        --nblk 1 \
        --nepoch 5 \
        --expf $EXP_DIR-$P \
        --manualSeed 1112 \
        --log_iter 50 \
        --visualize_iter 500 \
        --ckpt_save_epoch 1 \
        --mse_w 1.0 \
        --ffl_w 100.0 \
        --patch_factors 1 $P
done