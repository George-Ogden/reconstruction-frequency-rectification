#!/usr/bin/env bash

set -x

VERSION='celeba_recon_wo_ffl'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/test.txt'
EXP_DIR='./VQVAE/experiments/'$VERSION
RES_DIR='./VQVAE/results/'$VERSION
WORKER=0

python ./VQVAE/test.py \
    --dataset $DATA \
    --dataroot $DATAROOT \
    --datalist $DATALIST \
    --workers $WORKER \
    --batchSize 1 \
    --imageSize 224 \
    --nz 256 \
    --nblk 2 \
    --expf $EXP_DIR \
    --manualSeed 1112 \
    --epoch_test 10 \
    --eval \
    --resf $RES_DIR \
    --show_input