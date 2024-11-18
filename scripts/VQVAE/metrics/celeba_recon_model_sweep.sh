#!/usr/bin/env bash

set -x

VERSION='model-sweep'
METRICS='psnr ssim lpips fid lfd'
EPOCH='epoch_005_seed_1112_with_input'
W0=0.01
W1=0.01

for model in resnet18 resnet34 resnet50 resnet101 resnet152; do
    FDRF='./VQVAE/results/'$VERSION-$model-$W0-$W1'/'$EPOCH
    LOGS='./VQVAE/results/'$VERSION-$model-$W0-$W1'/metrics_'$EPOCH'.txt'
    python ./metrics/calc_metrics.py \
        --metrics $METRICS \
        --fdrf $FDRF \
        --logs $LOGS
done