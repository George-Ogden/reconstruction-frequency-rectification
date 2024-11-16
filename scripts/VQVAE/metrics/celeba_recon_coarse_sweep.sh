#!/usr/bin/env bash

set -x

VERSION='coarse-weight-sweep'
METRICS='psnr ssim lpips fid lfd'
EPOCH='epoch_005_seed_1112_with_input'

for W0 in 0.001 0.01 0.1 1 10; do
    for W1 in 0.001 0.01 0.1 1 10; do
        FDRF='./VQVAE/results/'$VERSION-$W0-$W1'/'$EPOCH
        LOGS='./VQVAE/results/'$VERSION-$W0-$W1'/metrics_'$EPOCH'.txt'
        python ./metrics/calc_metrics.py \
            --metrics $METRICS \
            --fdrf $FDRF \
            --logs $LOGS
    done
done