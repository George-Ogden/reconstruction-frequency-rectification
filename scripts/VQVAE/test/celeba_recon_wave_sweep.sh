#!/usr/bin/env bash

set -x

VERSION='celeba_recon_wave_sweep'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/test.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
RES_DIR='./VQVAE/results/'$VERSION
WORKER=8

for W in 0.0001 0.001 0.01 0.1 1; do
    python ./VQVAE/test.py \
        --dataset $DATA \
        --dataroot $DATAROOT \
        --datalist $DATALIST \
        --workers $WORKER \
        --batchSize 1 \
        --imageSize 224 \
        --nz 128 \
        --nblk 1 \
        --expf $EXP_DIR-$W \
        --manualSeed 1112 \
        --epoch_test 5 \
        --eval \
        --resf $RES_DIR-$W \
        --show_input 
done