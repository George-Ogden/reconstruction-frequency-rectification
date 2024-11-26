#!/usr/bin/env bash

set -x

VERSION='celeba_recon_wave_sweep'
METRICS='psnr ssim lpips fid lfd'
EPOCH='epoch_005_seed_1112_with_input'

for W0 in 0.001 0.003 0.01 0.03 0.1; do
    for W1 in 0.001 0.003 0.01 0.03 0.1; do
        for level in 1 2 3 4; do
            FDRF='./VQVAE/results/'$VERSION-$W0-$W1-$level'/'$EPOCH
            LOGS='./VQVAE/results/'$VERSION-$W0-$W1-$level'/metrics_'$EPOCH'.txt'
            python ./metrics/calc_metrics.py \
                --metrics $METRICS \
                --fdrf $FDRF \
                --logs $LOGS
        done
    done
done
