#!/usr/bin/env bash

set -x

VERSION='celeba_recon_wo_ffl'
EPOCH='epoch_010_seed_1112_with_input'
METRICS='psnr ssim lpips fid lfd'
FDRF='./VQVAE/results/'$VERSION'/'$EPOCH
LOGS='./VQVAE/results/'$VERSION'/metrics_'$EPOCH'.txt'

python ./metrics/calc_metrics.py \
    --metrics $METRICS \
    --fdrf $FDRF \
    --logs $LOGS
