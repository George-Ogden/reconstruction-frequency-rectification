#!/usr/bin/env bash

set -x

VERSION='celeba_recon_j_ffl'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/test.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
RES_DIR='./VQVAE/results/'$VERSION
WORKERS=8

python ./VQVAE/test.py \
    --dataset $DATA \
    --dataroot $DATAROOT \
    --datalist $DATALIST \
    --WORKERS $WORKERS \
    --batchSize 1 \
    --imageSize 224 \
    --nz 128 \
    --nblk 1 \
    --expf $EXP_DIR \
    --manualSeed 1112 \
    --epoch_test 5 \
    --eval \
    --resf $RES_DIR \
    --show_input
