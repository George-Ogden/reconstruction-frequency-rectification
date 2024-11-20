#!/usr/bin/env bash

set -x

VERSION='celeba_recon_wave_sweep'
METRICS='psnr ssim lpips fid lfd'
EPOCH='epoch_005_seed_1112_with_input'

for W in 0.0001 0.001 0.01 0.1 1; do
    FDRF='./VQVAE/results/'$VERSION-$W'/'$EPOCH
    LOGS='./VQVAE/results/'$VERSION-$W'/metrics_'$EPOCH'.txt'
    python ./metrics/calc_metrics.py \
        --metrics $METRICS \
        --fdrf $FDRF \
        --logs $LOGS
done