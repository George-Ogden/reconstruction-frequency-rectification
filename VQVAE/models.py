import os

import torch
import torch.nn as nn
import torch.optim as optim
from focal_frequency_loss import FocalFrequencyLoss as FFL
from torchvision.models import resnet50

from utils import print_and_write_log, weights_init
import torch
from torch import nn
from torch.nn import functional as F


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch
# Modified from https://github.com/rosinality/vq-vae-2-pytorch/blob/ef5f67c46f93624163776caec9e0d95063910eca/train_vqvae.py


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

class ResNetSubset(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1

        for param in self.parameters():
            param.requires_grad_(False)
    
    def forward(self, x):
        early_features = self.conv1(x)
        x = self.bn1(early_features)
        x = self.relu(x)
        x = self.maxpool(x)

        mid_features = self.layer1(x)
        return early_features, mid_features


class EncoderDecoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = torch.device("cuda:0" if not opt.no_cuda else "cpu")
        nc = int(opt.nc)
        nz = int(opt.nz)
        nblk = int(opt.nblk)
        self.netG = VQVAE(n_embed=nz, n_res_block=nblk, in_channel=nc).to(self.device)
        self.latent_loss_w = float(opt.latent_w)

        weights_init(self.netG)
        if opt.netG != '':
            self.netG.load_state_dict(torch.load(opt.netG, map_location=self.device))
        print_and_write_log(opt.train_log_file, 'netG:')
        print_and_write_log(opt.train_log_file, str(self.netG))

        # losses
        self.criterion = nn.MSELoss()
        # define focal frequency loss
        self.criterion_freq = FFL(loss_weight=opt.ffl_w,
                                  alpha=opt.alpha,
                                  patch_factor=opt.patch_factor,
                                  ave_spectrum=opt.ave_spectrum,
                                  log_matrix=opt.log_matrix,
                                  batch_matrix=opt.batch_matrix).to(self.device)
        self.cnn_loss_ws = float(opt.cnn_loss_w0), float(opt.cnn_loss_w1)
        self.resnet = ResNetSubset(resnet50(weights=opt.resnet_weights))

        # misc
        self.to(self.device)

        # optimizer
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    def forward(self):
        pass

    def criterion_cnn(self, recon, real):
        recon_early_features, recon_mid_features = self.resnet(recon)
        real_early_features, real_mid_features = self.resnet(real)

        early_loss_w, mid_loss_w = self.cnn_loss_ws
        
        return F.mse_loss(recon_early_features, real_early_features) * early_loss_w + F.mse_loss(recon_mid_features, real_mid_features) * mid_loss_w

    def gen_update(self, data, epoch, matrix=None):
        self.netG.zero_grad()
        real = data.to(self.device)
        if matrix is not None:
            matrix = matrix.to(self.device)
        recon, latent_loss = self.netG(real)
        latent_loss *= self.latent_loss_w

        # apply pixel-level loss
        errG_pix = self.criterion(recon, real) * self.opt.mse_w

        # apply focal frequency loss
        if epoch >= self.opt.freq_start_epoch:
            errG_freq = self.criterion_freq(recon, real, matrix)
        else:
            errG_freq = torch.tensor(0.0).to(self.device)

        # apply CNN loss
        errG_cnn = self.criterion_cnn(recon, real)

        errG = errG_pix + errG_freq + latent_loss + errG_cnn
        errG.backward()
        self.optimizerG.step()

        return errG_pix, errG_freq, latent_loss, errG_cnn

    def sample(self, x):
        x = x.to(self.device)
        self.netG.eval()
        with torch.no_grad():
            recon, _ = self.netG(x)
        self.netG.train()

        return recon

    def save_checkpoints(self, ckpt_dir, epoch):
        torch.save(self.netG.state_dict(), '%s/netG_epoch_%03d.pth' % (ckpt_dir, epoch))
