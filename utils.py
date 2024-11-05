import os
import torch
import requests
import zipfile
import logging
import functools
import numpy as np
from math import log10
import os.path as osp
import torch.nn as nn
from os import listdir
from PIL import Image
import tqdm as tqdm
from os.path import join
from torchnet.meter import meter
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, CenterCrop, Scale


class PSNRMeter(meter.Meter):
    def __init__(self):
        super(PSNRMeter, self).__init__()
        self.reset()

    def reset(self):
        self.n = 0
        self.sesum = 0.0

    def add(self, output, target):
        if not torch.is_tensor(output) and not torch.is_tensor(target):
            output = torch.from_numpy(output)
            target = torch.from_numpy(target)
        output = output.cpu()
        target = target.cpu()
        self.n += output.numel()
        self.sesum += torch.sum((output - target) ** 2)

    def value(self):
        mse = self.sesum / max(1, self.n)
        psnr = 10 * log10(1 / mse)
        return psnr


class BicubicUpsample(nn.Module):
    """ A bicubic upsampling class with similar behavior to that in TecoGAN-Tensorflow

        Note that it's different from torch.nn.functional.interpolate and
        matlab's imresize in terms of bicubic kernel and sampling scheme

        Theoretically it can support any scale_factor >= 1, but currently only
        scale_factor = 4 is tested

        References:
            The original paper: http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf
            https://stackoverflow.com/questions/26823140/imresize-trying-to-understand-the-bicubic-interpolation
    """

    def __init__(self, scale_factor, a=-0.75):
        super(BicubicUpsample, self).__init__()

        # calculate weights
        cubic = torch.FloatTensor([
            [0, a, -2 * a, a],
            [1, 0, -(a + 3), a + 2],
            [0, -a, (2 * a + 3), -(a + 2)],
            [0, 0, a, -a]
        ])  # accord to Eq.(6) in the reference paper

        kernels = [
            torch.matmul(cubic, torch.FloatTensor([1, s, s ** 2, s ** 3]))
            for s in [1.0*d/scale_factor for d in range(scale_factor)]
        ]  # s = x - floor(x)

        # register parameters
        self.scale_factor = scale_factor
        self.register_buffer('kernels', torch.stack(kernels))

    def forward(self, input):
        n, c, h, w = input.size()
        s = self.scale_factor

        # pad input (left, right, top, bottom)
        input = F.pad(input, (1, 2, 1, 2), mode='replicate')

        # calculate output (height)
        kernel_h = self.kernels.repeat(c, 1).view(-1, 1, s, 1)
        output = F.conv2d(input, kernel_h, stride=1, padding=0, groups=c)
        output = output.reshape(
            n, c, s, -1, w + 3).permute(0, 1, 3, 2, 4).reshape(n, c, -1, w + 3)

        # calculate output (width)
        kernel_w = self.kernels.repeat(c, 1).view(-1, 1, 1, s)
        output = F.conv2d(output, kernel_w, stride=1, padding=0, groups=c)
        output = output.reshape(
            n, c, s, h * s, -1).permute(0, 1, 3, 4, 2).reshape(n, c, h * s, -1)

        return output


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = dataset_dir + '/SRF_' + str(upscale_factor) + '/data'
        self.target_dir = dataset_dir + '/SRF_' + \
            str(upscale_factor) + '/target'
        self.image_filenames = [join(self.image_dir, x) for x in listdir(
            self.image_dir) if is_image_file(x)]
        self.target_filenames = [join(self.target_dir, x) for x in listdir(
            self.target_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, _, _ = Image.open(
            self.image_filenames[index]).convert('YCbCr').split()
        target, _, _ = Image.open(
            self.target_filenames[index]).convert('YCbCr').split()
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.image_filenames)


class STN(nn.Module):
    """Spatial transformer network.
      For optical flow based frame warping.

    Args:
      mode: sampling interpolation mode of `grid_sample`
      padding_mode: can be `zeros` | `borders`
      normalized: flow value is normalized to [-1, 1] or absolute value
    """

    def __init__(self, mode='bilinear', padding_mode='zeros', normalize=False):
        super(STN, self).__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.norm = normalize

    def forward(self, inputs, u, v=None, gain=1):
        batch = inputs.size(0)
        device = inputs.device
        mesh = nd_meshgrid(*inputs.shape[-2:], permute=[1, 0])
        mesh = torch.tensor(mesh, dtype=torch.float32, device=device)
        mesh = mesh.unsqueeze(0).repeat_interleave(batch, dim=0)
        # add flow to mesh
        if v is None:
            assert u.shape[1] == 2, "optical flow must have 2 channels"
            _u, _v = u[:, 0], u[:, 1]
        else:
            _u, _v = u, v
        if not self.norm:
            # flow needs to normalize to [-1, 1]
            h, w = inputs.shape[-2:]
            _u = _u / w * 2
            _v = _v / h * 2
        flow = torch.stack([_u, _v], dim=-1) * gain
        assert flow.shape == mesh.shape, f"Shape mis-match: {flow.shape} != {mesh.shape}"
        mesh = mesh + flow
        return F.grid_sample(inputs,
                             mesh,
                             mode=self.mode,
                             padding_mode=self.padding_mode,
                             align_corners=False)


class CoarseFineFlownet(nn.Module):
    def __init__(self, channel):
        """Coarse to fine flow estimation network

        Originally from paper "Real-Time Video Super-Resolution with Spatio-Temporal
        Networks and Motion Compensation".
        See Vespcn.py
        """

        super(CoarseFineFlownet, self).__init__()
        in_c = channel * 2
        # Coarse Flow
        conv1 = nn.Sequential(nn.Conv2d(in_c, 24, 5, 2, 2), nn.ReLU(True))
        conv2 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
        conv3 = nn.Sequential(nn.Conv2d(24, 24, 5, 2, 2), nn.ReLU(True))
        conv4 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
        conv5 = nn.Sequential(nn.Conv2d(24, 32, 3, 1, 1), nn.Tanh())
        up1 = nn.PixelShuffle(4)
        self.coarse_flow = nn.Sequential(
            conv1, conv2, conv3, conv4, conv5, up1)
        # Fine Flow
        in_c = channel * 3 + 2
        conv1 = nn.Sequential(nn.Conv2d(in_c, 24, 5, 2, 2), nn.ReLU(True))
        conv2 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
        conv3 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
        conv4 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
        conv5 = nn.Sequential(nn.Conv2d(24, 8, 3, 1, 1), nn.Tanh())
        up2 = nn.PixelShuffle(2)
        self.fine_flow = nn.Sequential(conv1, conv2, conv3, conv4, conv5, up2)
        self.warp_c = STN(padding_mode='border')

    def forward(self, target, ref, gain=1):
        """Estimate optical flow from `ref` frame to `target` frame"""

        flow_c = self.coarse_flow(torch.cat((ref, target), 1))
        wc = self.warp_c(ref, flow_c[:, 0], flow_c[:, 1])
        flow_f = self.fine_flow(
            torch.cat((ref, target, flow_c, wc), 1)) + flow_c
        flow_f *= gain
        return flow_f


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in ['.mp4', '.avi', '.mpg', '.mkv', '.wmv', '.flv'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Scale(crop_size // upscale_factor, interpolation=Image.BICUBIC)
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size)
    ])


def download_file(url, path, logger):
    """Download a file from a URL to a specified path in chunks."""
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Check for download errors
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
        logger.info(f"File downloaded to {path}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file: {e}")
        raise


def extract_zip(zip_path, extract_dir, logger):
    """Extract a ZIP file to a specified directory."""
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"Files extracted to {extract_dir}")
    except zipfile.BadZipFile as e:
        logger.error(f"Error extracting file: {e}")
        raise


def setup_logger(name):
    # create a logger
    base_logger = logging.getLogger(name=name)
    base_logger.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s]: %(message)s')
    # create a stream handler & set format
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    # add handlers
    base_logger.addHandler(sh)


def float32_to_uint8(inputs):
    """ Convert np.float32 array to np.uint8

        Parameters:
            :param input: np.float32, (NT)CHW, [0, 1]
            :return: np.uint8, (NT)CHW, [0, 255]
    """
    return np.uint8(np.clip(np.round(inputs * 255), 0, 255))


def space_to_depth(x, scale=4):
    """ Equivalent to tf.space_to_depth()
    """

    n, c, in_h, in_w = x.size()
    out_h, out_w = in_h // scale, in_w // scale

    x_reshaped = x.reshape(n, c, out_h, scale, out_w, scale)
    x_reshaped = x_reshaped.permute(0, 3, 5, 1, 2, 4)
    output = x_reshaped.reshape(n, scale * scale * c, out_h, out_w)

    return output


def backward_warp(x, flow, mode='bilinear', padding_mode='border'):
    """ Backward warp `x` according to `flow`

        Both x and flow are pytorch tensor in shape `nchw` and `n2hw`

        Reference:
            https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
    """

    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    # normalize flow to [-1, 1]
    flow = torch.cat([
        flow[:, 0:1, ...] / ((w - 1.0) / 2.0),
        flow[:, 1:2, ...] / ((h - 1.0) / 2.0)], dim=1)

    # add flow to grid and reshape to nhw2
    grid = (grid + flow).permute(0, 2, 3, 1)

    # bilinear sampling
    # Note: `align_corners` is set to `True` by default in PyTorch version
    #        lower than 1.4.0
    if int(''.join(torch.__version__.split('.')[:2])) >= 14:
        output = F.grid_sample(
            x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    else:
        output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    return output


def get_upsampling_func(scale=4, degradation='BI'):
    if degradation == 'BI':
        upsample_func = functools.partial(
            F.interpolate, scale_factor=scale, mode='bilinear',
            align_corners=False)

    elif degradation == 'BD':
        upsample_func = BicubicUpsample(scale_factor=scale)

    else:
        raise ValueError('Unrecognized degradation: {}'.format(degradation))

    return upsample_func


def initialize_weights(net_l, init_type='kaiming', scale=1):
    """ Modify from BasicSR/MMSR
    """

    if not isinstance(net_l, list):
        net_l = [net_l]

    for net in net_l:
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                else:
                    raise NotImplementedError(init_type)

                m.weight.data *= scale  # to stabilize training

                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)


def pad_if_divide(x: torch.Tensor, value, mode='constant'):
    """pad tensor if its width and height couldn't be divided by `value`.

    Args:
        x: a tensor at least has 3 dimensions.
        value: value to divide width and height.
        mode: a string, representing padding mode.
    Return:
        padded tensor.
    """

    shape = x.shape
    assert 3 <= x.dim() <= 4, f"Dim of x is not 3 or 4, which is {x.dim()}"
    h = shape[-2]
    w = shape[-1]
    dh = h + (value - h % value) % value - h
    dw = w + (value - w % value) % value - w
    pad = [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2]
    return F.pad(x, pad, mode)


def nd_meshgrid(*size, permute=None):
    _error_msg = ("Permute index must match mesh dimensions, "
                  "should have {} indexes but got {}")
    size = np.array(size)
    ranges = []
    for x in size:
        ranges.append(np.linspace(-1, 1, x))
    mesh = np.stack(np.meshgrid(*ranges, indexing='ij'))
    if permute is not None:
        if len(permute) != len(size):
            raise ValueError(_error_msg.format(len(size), len(permute)))
        mesh = mesh[permute]
    return mesh.transpose(*range(1, mesh.ndim), 0)


def retrieve_files(dir, suffix='png|jpg'):
    """ retrive files with specific suffix under dir and sub-dirs recursively
    """

    def retrieve_files_recursively(dir, file_lst):
        for d in sorted(os.listdir(dir)):
            dd = osp.join(dir, d)

            if osp.isdir(dd):
                retrieve_files_recursively(dd, file_lst)
            else:
                if osp.splitext(d)[-1].lower() in ['.' + s for s in suffix]:
                    file_lst.append(dd)

    if not dir:
        return []

    if isinstance(suffix, str):
        suffix = suffix.split('|')

    file_lst = []
    retrieve_files_recursively(dir, file_lst)
    file_lst.sort()

    return file_lst


def generate_dataset(data_type, upscale_factor):
    images_name = [x for x in listdir(
        'data/VOC2012/' + data_type) if is_image_file(x)]
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    lr_transform = input_transform(crop_size, upscale_factor)
    hr_transform = target_transform(crop_size)

    root = 'data/' + data_type
    if not os.path.exists(root):
        os.makedirs(root)
    path = root + '/SRF_' + str(upscale_factor)
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = path + '/data'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    target_path = path + '/target'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for image_name in tqdm(images_name, desc='generate ' + data_type + ' dataset with upscale factor = '
                           + str(upscale_factor) + ' from VOC2012'):
        image = Image.open('data/VOC2012/' + data_type + '/' + image_name)
        target = image.copy()
        image = lr_transform(image)
        target = hr_transform(target)

        image.save(image_path + '/' + image_name)
        target.save(target_path + '/' + image_name)
