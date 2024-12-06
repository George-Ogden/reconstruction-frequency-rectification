
## Modified VQVAE

This repository builds on and explores frequency-space losses. 
VQVAE has been used as a baseline model; the starting point is taken from the official PyTorch implementation for the following paper:

**Focal Frequency Loss for Image Reconstruction and Synthesis**<br>
[Liming Jiang](https://liming-jiang.com/), [Bo Dai](http://daibo.info/), [Wayne Wu](https://wywu.github.io/) and [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/)<br>
In ICCV 2021.<br>
[**Project Page**](https://www.mmlab-ntu.com/project/ffl/index.html) | [**Paper**](https://arxiv.org/abs/2012.12821) | [**Poster**](https://liming-jiang.com/projects/FFL/resources/poster.pdf) | [**Slides**](https://liming-jiang.com/projects/FFL/resources/slides.pdf) | [**YouTube Demo**](https://www.youtube.com/watch?v=RNTnDtKvcpc)
> **Abstract:** *Image reconstruction and synthesis have witnessed remarkable progress thanks to the development of generative models. Nonetheless, gaps could still exist between the real and generated images, especially in the frequency domain. In this study, we show that narrowing gaps in the frequency domain can ameliorate image reconstruction and synthesis quality further. We propose a novel focal frequency loss, which allows a model to adaptively focus on frequency components that are hard to synthesize by down-weighting the easy ones. This objective function is complementary to existing spatial losses, offering great impedance against the loss of important frequency information due to the inherent bias of neural networks. We demonstrate the versatility and effectiveness of focal frequency loss to improve popular models, such as VAE, pix2pix, and SPADE, in both perceptual quality and quantitative performance. We further show its potential on StyleGAN2.*

## Updates

- [09/2021] [Originator's Work] The **code** of Focal Frequency Loss is **released**.

- [07/2021] [Originator's Work] The [paper](https://arxiv.org/abs/2012.12821) detailing Focal Frequency Loss is accepted by **ICCV 2021**.

- [11/2024] [Our Project Work] Our modifications + extensions to the VQVAE model + FFL implementation are provided **here**. Includes WaveletLoss and CNNLoss implementations. 

## Quick Start

Standalone loss classes are available as pip-installable packages at [**j-muoneke/image-losses**](https://github.com/j-muoneke/image-losses)

To get a copy of this repository, perform the following steps:

```bash
git clone https://github.com/George-Ogden/reconstruction-frequency-rectification.git
cd reconstruction-frequency-rectification
pip install -r VQVAE/requirements.txt
```

## Example - Initialising + Using Losses

```python
from focal_frequency_loss import FocalFrequencyLoss as FFL
from wavelet_loss import WaveletLoss as WVL
from cnn_loss import CNNLoss as CNNL

# [WaveletLoss example] initialise nn.Module class
# High- and low-band weights	 	specified with w0 | w1 ( float )
# Single-level calc			specified with level   ( int )
# An internal loss function		specified with loss_fn ( nn.Module )
# Wavelet function set 			specified with wavelet (string | pywt Wavelet object)
wvl = WVL(wavelet='haar', level=1, loss_fn=nn.MSELoss(), w0=0.001, w1=0.01)


# [FFLoss Example] initialize nn.Module class
# Relative loss weight			specified with loss_weight 	( float ) 
# No loss exponentiation		specified with alpha 		( float )
# Whole image + 4x4 patch loss 		specified with patch_factors 	( List[int] )
ffl = FFL(loss_weight=1.0, alpha=1.0, patch_factors = [1,4])

# [CNNLoss Example] initialize nn.Module class
# With early- and mid-feature weights 	specified with  w0 | w1 ( float )
# ResNet50 as extractor net		specified with model 	( string )
cnnl = CNNL(model = "Resnet50", w0 = 0.1, w1 = 0.1)

# Fabricate data
import torch
fake = torch.randn(4, 3, 64, 64)  # replace it with the predicted tensor of shape (N, C, H, W)
real = torch.randn(4, 3, 64, 64)  # replace it with the target tensor of shape (N, C, H, W)

losses = [ffl, wvl, cnnl]
for loss in losses:
	# Calc + print loss values
	print(loss(fake, real))
```

### Dataset Preparation

We've sourced a cropped, face-aligned version of the CelebFaces dataset prepared by the originators here: [img\_align\_celeba.zip](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ)
[official website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Variants of this dataset should go in their own folder (for this example, inside `datasets/celeba/img_align_celeba`). See below as an example:

```
├── datasets
│    ├── celeba
│    │    ├── img_align_celeba  
│    │    │    ├── 000001.jpg
│    │    │    ├── 000002.jpg
│    │    │    ├── 000003.jpg
│    │    │    ├── ...
```
We've also used a normed version of the Describable Textures Dataset [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

### Testing

Download the [pretrained models](https://drive.google.com/file/d/1YIH09eoDyP2JLmiYJpju4hOkVFO7M3b_/view?usp=sharing) and unzip them to `./VQVAE/experiments`.

Along with originator examples, we've provided some more [test scripts](https://github.com/George-Ogden/reconstruction-frequency-rectification/tree/master/scripts/VQVAE/test). 
If you only have a CPU environment, please specify `--no_cuda` in the script. Run:

```bash
bash scripts/VQVAE/test/celeba_recon_wo_ffl.sh
bash scripts/VQVAE/test/celeba_recon_w_ffl.sh
```

Any image reconstruction results will be saved at `./VQVAE/results` by default.

### Training

Originators have provided example [training scripts](https://github.com/EndlessSora/focal-frequency-loss/tree/master/scripts/VanillaAE/train), which we have modified to observe baseline training for VQVAE. 
To recreate our experiments and hyperparameter tuning, we've also provided an exemplar [parameter sweep script](https://github.com/George-Ogden/reconstruction-frequency-rectification/blob/master/scripts/VQVAE/train/celeba_recon_wave_sweep.sh)
For CPU-only environments please specify `--no_cuda` in the script. Run:

```bash
bash scripts/VQVAE/train/celeba_recon_wo_ffl.sh
bash scripts/VQVAE/train/celeba_recon_w_ffl.sh 
bash scripts/VQVAE/train/celeba_recon_wave_sweep.sh
```

## Metrics

We've added scripts and Python methods for regenerating the plots and images seen in our Report.
These can be found and used [here](https://github.com/George-Ogden/reconstruction-frequency-rectification/tree/master/scripts/VQVAE/metrics) after running the train/test scripts for the the relevant model

## Acknowledgments

The code of Vanilla AE is inspired by [PyTorch DCGAN](https://github.com/pytorch/examples/tree/master/dcgan) and [MUNIT](https://github.com/NVlabs/MUNIT). Part of the evaluation metric code is borrowed from [MMEditing](https://github.com/open-mmlab/mmediting). We also apply [LPIPS](https://github.com/richzhang/PerceptualSimilarity) and [pytorch-fid](https://github.com/mseitzer/pytorch-fid) as evaluation metrics.
Full credit is given to originators for baseline [VQVAE model and frequency loss](https://github.com/EndlessSora/focal-frequency-loss) implementations
Inspiration for methods mainly derivated from [FrePolad](https://github.com/Chenliang-Zhou/FrePolad), a point-cloud diffusion model. Guidance on changes and approaches to loss formulation was kindly provided by one of its authors. 
