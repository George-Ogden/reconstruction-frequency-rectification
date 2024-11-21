from pytorch_wavelets import DWTForward  # Discrete Wavelet Transform Forward

import torch.nn as nn

class WaveletLoss(nn.Module):
    def __init__(self, wavelet='haar', level=1, loss_fn=nn.MSELoss()):
        super(WaveletLoss, self).__init__()
        self.dwt = DWTForward(J=level, wave=wavelet)  # Wavelet Transform
        self.loss_fn = loss_fn

    def forward(self, pred, target):
        # Perform Wavelet Transform on predicted and target images
        pred_coeffs = self.dwt(pred)
        target_coeffs = self.dwt(target)

        # Compute loss for each subband
        total_loss = 0
        pred_yl, pred_yh = pred_coeffs
        target_yl, target_yh = target_coeffs

        for pred_coeffs, targte_coeffs in zip(pred_yl, target_yl):
            total_loss += self.loss_fn(pred_coeffs, targte_coeffs)

        for pred_coeffs, targte_coeffs in zip(pred_yh, target_yh):
            total_loss += self.loss_fn(pred_coeffs, targte_coeffs)

        return total_loss