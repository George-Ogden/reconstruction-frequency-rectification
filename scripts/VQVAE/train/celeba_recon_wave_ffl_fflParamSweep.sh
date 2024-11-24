#!/usr/bin/env bash

set -x

VERSION='celeba_recon_wave_ffl_fflParamSweep'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/train.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
WORKER=8

for P in 2; do
    python3 ./VQVAE/train.py \
        --no_cuda \
        --dataset $DATA \
        --dataroot $DATAROOT \
        --datalist $DATALIST \
        --workers $WORKER \
        --batchSize 8 \
        --imageSize 224 \
        --nz 256 \
        --nblk 2 \
        --nepoch 5 \
        --expf $EXP_DIR \
        --manualSeed 1112 \
        --log_iter 50 \
        --visualize_iter 500 \
        --ckpt_save_epoch 1 \
        --mse_w 1.0 \
        --ffl_w 100.0 \
        --wavelet_w 1.0\
        --patch_factors 1 $P
done