#!/usr/bin/env bash

set -x

VERSION='dtd_recon_w_ffl'
EPOCH='epoch_005_seed_1112_with_input'
METRICS='psnr ssim lpips fid lfd'
FDRF='./VQVAE/results/'$VERSION'/'$EPOCH
LOGS='./VQVAE/results/'$VERSION'/metrics_'$EPOCH'.txt'

python ./metrics/calc_metrics.py \
    --metrics $METRICS \
    --fdrf $FDRF \
    --logs $LOGS
