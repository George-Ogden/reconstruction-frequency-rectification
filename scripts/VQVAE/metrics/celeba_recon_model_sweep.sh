#!/usr/bin/env bash

set -x

VERSION='celeba_recon_model_sweep'
METRICS='psnr ssim lpips fid lfd'
EPOCH='epoch_005_seed_1112_with_input'

for model in resnet18 resnet34 resnet50 resnet101 resnet152; do
    FDRF='./VQVAE/results/'$VERSION-$model'/'$EPOCH
    LOGS='./VQVAE/results/'$VERSION-$model'/metrics_'$EPOCH'.txt'
    python ./metrics/calc_metrics.py \
        --metrics $METRICS \
        --fdrf $FDRF \
        --logs $LOGS
done
