# VideoSR

As part of our Deep Learning project, our problem statement was benchmarking different state-of-the-art video super resolution architectures and compare their pro's and con's. 

These are the benchmarks adopted from the parent repository, which have been added as a submodule inside this repo. 

## Project Tree: 
```
.
├── config.py
├── data
│   ├── base.py
│   ├── __init__.py
│   ├── paired.py
│   └── unpaired.py
├── LICENSE
├── metrics
│   ├── evaluate.py
│   ├── __init__.py
│   ├── LPIPS
│   │   ├── metrics.py
│   │   ├── models
│   │   │   ├── base.py
│   │   │   ├── dist.py
│   │   │   ├── __init__.py
│   │   │   ├── networks.py
│   │   │   ├── pretrained.py
│   │   │   ├── util.py
│   │   │   └── v0.1
│   │   │       ├── alex.pth
│   │   │       ├── squeeze.pth
│   │   │       └── vgg.pth
│   │   ├── source.txt
│   │   └── summary.py
│   └── metrics.py
├── models
│   ├── base.py
│   ├── custom.py
│   ├── espcn.py
│   ├── net.py
│   ├── sofvsr.py
│   └── vespcn.py
├── networks
│   ├── base.py
│   ├── custom.py
│   ├── egvsr.py
│   ├── espcn.py
│   ├── __init__.py
│   ├── sofvsr.py
│   ├── tecogan.py
│   └── vespcn.py
├── README.md
├── requirements.txt
├── train_egvsr.py
├── train_net.py
├── utils.py
└── weights
    ├── egvsr.pth
    ├── espcn.pth
    ├── frvsr.pth
    ├── sofvsr.pth
    ├── tecogan.pth
    └── vespcn.pth
```

The project is structured into the following directories:

* `data/`: Contains data loading and processing modules, including paired (LR-HR) and unpaired data loaders for GAN training.
* `metrics/`: Contains evaluation metrics implementation, including LPIPS (Learned Perceptual Image Patch Similarity) with multiple backbone options.
* `models/`: Contains implementations of various VSR models including ESPCN, SOFVSR, and VESPCN.
* `networks/`: Contains network architectures for different VSR approaches (TecoGAN, EGVSR, ESPCN, SOFVSR, VESPCN).
* `weights/`: Contains pretrained weights for all implemented architectures.

The project tree also contains several individual files:

* `config.py`: Configuration settings for the project.
* `train_egvsr.py`: Training script specifically for EGVSR model.
* `train_net.py`: Training script for other architectures.
* `utils.py`: Utility functions used across the project.

## Implmented Architectures: 
The repository currently supports the following VSR architectures. 
* **TecoGAN**: Temporarily Coherent GAN
* **EGVSR**: Enhanced/Extended GAN VSR
* **ESPCN**: Efficient Sub-Pixel CNN
* **SOFVSR**: Spatio-temporal Optical Flow VSR
* **VESPCN**: Very Deep ESPCN

## Installation and Usage: 
### Step 1: Setup a virtualenv / conda environment. 
For conda: 
```bash
$ conda create -n <env_name> 
```

For virtualenv:
```bash
$ python -m venv <env_name>
```
where `env_name` will be the name of the directory that you will use as the virtual environment. 

### Step 2: Activate the environment. 
For conda: 
```bash
$ conda install --file requirements.txt
```

For virualenv: 
```bash
$ pip install -r requirements.txt
```

### Step 3: Training
You can run any of the two python (train_egvsr, train_net) files for running inference.

train_egvsr:- runs inference for TecoGAN and the custom model. 

train_net:- runs inference for ESPCN. 

## Benchmark One: 
We will be comparing objective metrics for visual quality evaluation. This will be against a handful of other architectures like VESPCN, SOFVSR and DUF. 

Here's the table converted to markdown format:

| Sequence Name | Metric | VESPCN | SOFVSR | DUF | EGVSR |
|--------------|--------|---------|---------|-----|------|
| Calendar | PSNR↑ | 14.67 | 18.39 | 23.59 | 23.60 |
| | SSIM↑ | 0.19 | 0.50 | 0.80 | 0.80 |
| | LPIPS↓ | 0.57 | 0.41 | 0.33 | 0.17 |
| City | PSNR↑ | 19.38 | 22.03 | 27.63 | 27.31 |
| | SSIM↑ | 0.14 | 0.69 | 0.79 | 0.79 |
| | LPIPS↓ | 0.48 | 0.21 | 0.27 | 0.16 |
| Foliage | PSNR↑ | 16.22 | 22.96 | 26.15 | 24.79 |
| | SSIM↑ | 0.09 | 0.46 | 0.77 | 0.73 |
| | LPIPS↓ | 0.54 | 0.36 | 0.35 | 0.14 |
| Walk | PSNR↑ | 15.28 | 20.91 | 29.90 | 27.84 |
| | SSIM↑ | 0.32 | 0.45 | 0.91 | 0.86 |
| | LPIPS↓ | 0.34 | 0.44 | 0.14 | 0.09 |
| Average | PSNR↑ | 16.20 | 21.02 | 26.82 | 25.88 |
| | SSIM↑ | 0.19 | 0.53 | 0.82 | 0.80 |
| | LPIPS↓ | 0.48 | 0.36 | 0.27 | 0.14 |

## Benchmark Two: 
The next set of benchmarks will be specifically comparing TecoGAN and EGVSR, the two tight competitors for being state 
of the art in video super-resolution. 

What custom architecture are we using? 

We are using a variant of the "TecoGAN" model, which uses a FNet architecture for the generator and a "spatio-temporal" discriminator. We modified the FNet 
architecture to use Residual Blocks instead as they have better performance in terms of carrying the weights and maintaining gradient flow. The benchmarks for
the following datasets are listed below: 

| Dataset | Metric | TecoGAN | EGVSR | Our Model |
|---------|--------|---------|-------|-----------|
| Vid4 | Avg_LPIPS↓ | 0.156 | **0.138** | 0.150 |
| | Avg_tOF↓ | 0.199 | **0.188** | 0.192 |
| | Avg_tLP100↓ | 0.510 | **0.447** | 0.480 |
| ToS3 | Avg_LPIPS↓ | **0.109** | 0.121 | 0.110 |
| | Avg_tOF↓ | **0.134** | 0.138 | 0.138 |
| | Avg_tLP100↓ | 0.211 | 0.221 | 0.217 |
| GvT72 | Avg_LPIPS↓ | 0.069 | 0.072 | **0.068** |
| | Avg_tOF↓ | 0.233 | **0.232** | 0.236 |
| | Avg_tLP100↓ | **0.429** | 0.438 | 0.435 |

## License: 
This project is covered by the MIT License, granting anyone to use any parts of this repository for any use whatsoever. Original credit of the code goes to the authors that developed the submodules. 

We give credits to the original authors who authored the papers refered to in the literature review. 