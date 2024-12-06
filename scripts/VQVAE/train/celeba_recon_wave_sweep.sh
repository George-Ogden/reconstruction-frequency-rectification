#!/usr/bin/env bash

set -x

VERSION='celeba_recon_wave_sweep'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/train.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
WORKERS=8

for W0 in 0.001 0.003 0.01 0.03 0.1; do
    for W1 in 0.001 0.003 0.01 0.03 0.1; do
        for level in 1 2 3 4; do
            python ./VQVAE/train.py \
                --dataset $DATA \
                --dataroot $DATAROOT \
                --datalist $DATALIST \
                --WORKERS $WORKERS \
                --batchSize 128 \
                --imageSize 224 \
                --nz 128 \
                --nblk 1 \
                --nepoch 5 \
                --expf $EXP_DIR-$W0-$W1-$level \
                --manualSeed 1112 \
                --log_iter 50 \
                --visualize_iter 500 \
                --ckpt_save_epoch 1 \
                --mse_w 1.0 \
                --ffl_w 0.0 \
                --wavelet_w0 $W0 \
                --wavelet_w1 $W1 \
                --wavelet_level $level
        done
    done
done