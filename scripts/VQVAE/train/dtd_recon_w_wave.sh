#!/usr/bin/env bash

set -x

VERSION='dtd_recon_w_wave'
DATA='filelist'
DATAROOT='./datasets/dtd/images'
DATALIST='./datasets/dtd/train_list.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
WORKERS=8
W0=0.001
W1=0.03
LEVEL=1

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
    --expf $EXP_DIR \
    --manualSeed 1112 \
    --log_iter 50 \
    --visualize_iter 500 \
    --ckpt_save_epoch 1 \
    --mse_w 1.0 \
    --ffl_w 0.0 \
    --wavelet_w0 $W0 \
    --wavelet_w1 $W1 \
    --wavelet_level $LEVEL

