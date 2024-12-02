#!/usr/bin/env bash

set -x

VERSION='celeba_recon_ffl_patch_sweep'
METRICS='psnr ssim lpips fid lfd'
EPOCH='epoch_005_seed_1112_with_input'

for P in 1 2 4 8 16 32; do
    FDRF='./VQVAE/results/'$VERSION-$P'/'$EPOCH
    LOGS='./VQVAE/results/'$VERSION-$P'/metrics_'$EPOCH'.txt'
    python ./metrics/calc_metrics.py \
        --metrics $METRICS \
        --fdrf $FDRF \
        --logs $LOGS
done
