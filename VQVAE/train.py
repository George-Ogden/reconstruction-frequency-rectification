from __future__ import print_function
import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision.models as vmodels
from tqdm import tqdm

from models import EncoderDecoder
from utils import get_dataloader, print_and_write_log, set_random_seed


parser = argparse.ArgumentParser()

# data-args
parser.add_argument('--dataset', required=True, help='folderall | filelist | pairfilelist')
parser.add_argument('--dataroot', default='', help='path to dataset')
parser.add_argument('--datalist', default='', help='path to dataset file list')
parser.add_argument('--WORKERS', type=int, help='number of data loading WORKERS', default=4)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size (default 128)')
parser.add_argument('--imageSize', type=int, default=64, help='dimensions of network\'s input image (height = width = imageSize)')

# model-args
parser.add_argument('--nz', type=int, default=256, help='dimension of the latent layers')
parser.add_argument('--nblk', type=int, default=2, help='number of blocks')
parser.add_argument('--nepoch', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default=0.0003)')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam (default=0.9)')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam (default=0.999)')
parser.add_argument('--no_cuda', action='store_true', help='disable cuda flag (if only using CPU)')
parser.add_argument('--netG', default='', help='path to netG (to continue training)')
parser.add_argument('--manualSeed', type=int, help='set manual seed')

# display-save-args
parser.add_argument('--expf', default='./experiments', help='folder to save visualized images and model checkpoints')
parser.add_argument('--log_iter', type=int, default=50, help='log interval (iterations)')
parser.add_argument('--visualize_iter', type=int, default=500, help='visualization interval (iterations)')
parser.add_argument('--ckpt_save_epoch', type=int, default=1, help='checkpoint save interval (epochs)')

# default-loss-args
parser.add_argument('--mse_w', type=float, default=1.0, help='weight for mse (L2) spatial loss')
parser.add_argument('--latent_w', type=float, default=0.25, help='weight for latent loss')

# cnn-loss-args
parser.add_argument('--cnn_loss_w0', type=float, help='weight to use for the early layer CNN loss', default=0.0)
parser.add_argument('--cnn_loss_w1', type=float, help='weight to use for the mid layer CNN loss', default=0.0)
parser.add_argument('--model', type=str, help='type of model to use for CNN loss', default='resnet50')

# wavelet-loss-args
parser.add_argument('--wavelet_w0', type=float, default=0.0, help='Wavelet loss weight for low frequency terms')
parser.add_argument('--wavelet_w1', type=float, default=0.0, help='Wavelet loss weight for high frequency terms')
parser.add_argument('--wavelet_level', type=int, default=0, help='decomposition level for wavelet loss')

# ffl-loss-args
parser.add_argument('--freq_start_epoch', type=int, default=1, help='the start epoch to add focal frequency loss')
parser.add_argument('--ffl_w', type=float, default=0.0, help='weight for focal frequency loss')
parser.add_argument('--ave_spectrum', action='store_true', help='whether to use minibatch average spectrum')
parser.add_argument('--alpha', type=float, default=1.0, help='the scaling factor alpha of the spectrum weight matrix for flexibility')
parser.add_argument('--log_matrix', action='store_true', help='whether to adjust the spectrum weight matrix by logarithm')
parser.add_argument('--batch_matrix', action='store_true', help='whether to calculate the spectrum weight matrix using batch-based statistics')
parser.add_argument('--patch_factors', type=int, nargs='+', default=[1], help='unique list of img subdivision levels to use in multi-FFL\ne.g. train.py --patch_factors 2 4') 

opt = parser.parse_args()

opt.is_train = True

os.makedirs(os.path.join(opt.expf, 'images'), exist_ok=True)
os.makedirs(os.path.join(opt.expf, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(opt.expf, 'logs'), exist_ok=True)
train_log_file = os.path.join(opt.expf, 'logs', 'train_log.txt')
opt.train_log_file = train_log_file

# [FFL-PARAM-SWEEP DEBUG] handle patch_factors args
if opt.patch_factors:
    for factor in opt.patch_factors:
        if (opt.imageSize % factor != 0 ): 
            parser.error("ERROR: One or more --patch_factors were not factors of --imageSize")

    if len(opt.patch_factors) > 1:
        opt.patch_factors = sorted(list(set(opt.patch_factors)))
        print_and_write_log(train_log_file, f"DEBUG: Multi-patching used.\nFactors:{opt.patch_factors}")

cudnn.benchmark = True

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print_and_write_log(train_log_file, "Random Seed: %d" % opt.manualSeed)
set_random_seed(opt.manualSeed)

if torch.cuda.is_available() and opt.no_cuda:
    print_and_write_log(train_log_file, "WARNING: You have a CUDA device, so you should probably run without --no_cuda")
    
print("Successfully parsed args")
    

dataloader, nc = get_dataloader(opt)
opt.nc = nc

print_and_write_log(train_log_file, opt)

model = EncoderDecoder(opt)

num_epochs = opt.nepoch
iters = 0

matrix = None
for epoch in tqdm(range(1, num_epochs + 1)):
    for i, data in enumerate(tqdm(dataloader), 0):
        if opt.dataset == 'pairfilelist':
            img, matrix = data
            data = img

        # main training code
        errG_pix, errG_freq, errG_wavelet, latent_loss, errG_cnn = model.gen_update(data, epoch, matrix)

        # logs
        if i % opt.log_iter == 0:
            print_and_write_log(train_log_file,
                                '[%d/%d][%d/%d] LossPixel: %.10f LossFreq: %.10f LossWavelet %.10f LossLatent %.10f LossCNN %.10f' %
                                (epoch, num_epochs, i, len(dataloader), errG_pix.item(), errG_freq.item(), errG_wavelet.item(), latent_loss.item(), errG_cnn.item()))

        # write images for visualization
        if (iters % opt.visualize_iter == 0) or ((epoch == num_epochs) and (i == len(dataloader) - 1)):
            real_cpu = data.cpu()
            recon = model.sample(real_cpu)
            visual = torch.cat([real_cpu[:16], recon.detach().cpu()[:16]], 0)
            visual = visual * torch.tensor((0.299, 0.244, 0.225)).reshape((1, 3, 1, 1)) + torch.tensor((0.485, 0.456, 0.406)).reshape((1, 3, 1, 1))
            visual = visual.clamp(0.0, 1.0)
            vutils.save_image(visual, '%s/images/epoch_%03d_real_recon.png' % (opt.expf, epoch), normalize=False, nrow=16)

        iters += 1

    # save checkpoints
    if epoch % opt.ckpt_save_epoch == 0 or epoch == num_epochs:
        model.save_checkpoints('%s/checkpoints' % opt.expf, epoch)

print_and_write_log(train_log_file, 'Finish training.')
