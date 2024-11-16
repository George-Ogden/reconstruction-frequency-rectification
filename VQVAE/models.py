import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from focal_frequency_loss import FocalFrequencyLoss as FFL
from torchvision import models

from networks import VQVAE, ResNetSubset
from utils import print_and_write_log, weights_init

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
                                  batch_matrix=opt.batch_matrix).to(self.device) if opt.ffl_w != 0 else (
                                        lambda tensor, *tensors: torch.zeros((), device=tensor.device, dtype=tensor.dtype))
        self.cnn_loss_ws = float(opt.cnn_loss_w0), float(opt.cnn_loss_w1)
        self.resnet = ResNetSubset(getattr(models, opt.model)(weights=opt.resnet_weights))

        # misc
        self.to(self.device)

        # optimizer
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    def forward(self):
        pass

    def criterion_cnn(self, recon, real):
        early_loss_w, mid_loss_w = self.cnn_loss_ws
        if early_loss_w == 0.0 and mid_loss_w == 0.0:
            return torch.zeros((), dtype=recon.dtype, device=recon.device)

        recon_early_features, recon_mid_features = self.resnet(recon)
        real_early_features, real_mid_features = self.resnet(real)
        
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
