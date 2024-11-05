# VideoSR

As part of our Deep Learning project, our problem statement was benchmarking different state-of-the-art video super resolution architectures and compare their pro's and con's. 

These are the benchmarks adopted from the parent repository, which have been added as a submodule inside this repo. 

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

| Dataset | Metric | TecoGAN | EGVSR |
|---------|---------|----------|--------|
| Vid4 | Avg_LPIPS↓ | 0.156 | 0.138 |
| | Avg_tOF↓ | 0.199 | 0.188 |
| | Avg_tLP100↓ | 0.510 | 0.447 |
| ToS3 | Avg_LPIPS↓ | 0.109 | 0.121 |
| | Avg_tOF↓ | 0.134 | 0.138 |
| | Avg_tLP100↓ | 0.211 | 0.221 |
| GvT72 | Avg_LPIPS↓ | 0.069 | 0.072 |
| | Avg_tOF↓ | 0.233 | 0.232 |
| | Avg_tLP100↓ | 0.429 | 0.438 |

## License: 
This project is covered by the MIT License, granting anyone to use any parts of this repository for any use whatsoever. Original credit of the code goes to the authors that developed the submodules. 