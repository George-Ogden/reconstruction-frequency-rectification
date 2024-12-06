#!/usr/bin/env bash

set -x

VERSION='celeba_recon_model_weight_sweep'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/test.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
RES_DIR='./VQVAE/results/'$VERSION
WORKERS=8

for W0 in 0.001 0.01 0.1 1 10; do
    for W1 in 0.001 0.01 0.1 1 10; do
        python ./VQVAE/test.py \
            --dataset $DATA \
            --dataroot $DATAROOT \
            --datalist $DATALIST \
            --WORKERS $WORKERS \
            --batchSize 1 \
            --imageSize 224 \
            --nz 256 \
            --nblk 2 \
            --expf $EXP_DIR-$W0-$W1 \
            --manualSeed 1112 \
            --epoch_test 5 \
            --eval \
            --resf $RES_DIR-$W0-$W1 \
            --show_input 
    done
done