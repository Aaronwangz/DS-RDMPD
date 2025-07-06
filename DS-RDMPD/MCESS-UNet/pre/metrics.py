import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import lpips
from numpy import cov, trace, iscomplexobj
from scipy.linalg import sqrtm


# Gaussian and window functions for SSIM calculation
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(pred, gt):
    pred = torch.clamp(pred, 0, 1)
    gt = torch.clamp(gt, 0, 1)
    mse = F.mse_loss(pred, gt)
    psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr_val.item()


def lpips(pred, gt, model):
    return model(pred, gt).item()


def compute_mscn(image, C=1 / 255):
    padding = 3  # 使卷积操作后图像大小不变
    mu = F.avg_pool2d(image, 7, 1, padding)
    mu_sq = mu * mu
    sigma = F.avg_pool2d(image * image, 7, 1, padding) - mu_sq
    sigma = torch.sqrt(sigma + C)
    mscn = (image - mu) / (sigma + 1)
    return mscn


def compute_pair_product(image):
    shifts = [(1, 0), (0, 1), (1, 1), (1, -1)]
    return [image * torch.roll(image, shift, dims=(2, 3)) for shift in shifts]


def niqe(image):
    mscn = compute_mscn(image)
    pair_products = compute_pair_product(mscn)
    niqe_score = torch.mean(torch.stack(pair_products))
    return niqe_score.item()


# Calculate Fréchet Inception Distance (FID)
def calculate_fid(act1, act2):
    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid
