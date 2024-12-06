#!/usr/bin/env bash

set -x

VERSION='celeba_recon_wave_sweep'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/test.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
RES_DIR='./VQVAE/results/'$VERSION
WORKERS=8

for W0 in 0.001 0.003 0.01 0.03 0.1; do
    for W1 in 0.001 0.003 0.01 0.03 0.1; do
        for level in 1 2 3 4; do
            python ./VQVAE/test.py \
                --dataset $DATA \
                --dataroot $DATAROOT \
                --datalist $DATALIST \
                --WORKERS $WORKERS \
                --batchSize 1 \
                --imageSize 224 \
                --nz 128 \
                --nblk 1 \
                --expf $EXP_DIR-$W0-$W1-$level \
                --manualSeed 1112 \
                --epoch_test 5 \
                --eval \
                --resf $RES_DIR-$W0-$W1-$level \
                --show_input 
        done
    done
done