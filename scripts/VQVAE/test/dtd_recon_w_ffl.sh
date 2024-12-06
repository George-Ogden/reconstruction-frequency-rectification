#!/usr/bin/env bash

set -x

VERSION='dtd_recon_w_ffl'
DATA='filelist'
DATAROOT='./datasets/dtd/images'
DATALIST='./datasets/dtd/test_list.txt'
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
