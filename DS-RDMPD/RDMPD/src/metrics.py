import torch
import torch.nn.functional as F


def create_window(window_size, channel):
    """ 创建一个高斯窗口，用于 SSIM 计算 """
    _1D_window = torch.hann_window(window_size).float().unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size)


def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算图像对的 SSIM（结构相似度）
    :param img1: 输入图像1 (B, C, H, W)
    :param img2: 输入图像2 (B, C, H, W)
    :param window_size: 高斯窗大小
    :param size_average: 是否返回平均值
    :return: SSIM值（按批次计算）
    """
    (_, channel, _, _) = img1.size()

    # 创建高斯窗口
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())

    window = window.type_as(img1)

    # 计算均值和方差
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

    # 计算 SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 保证返回 SSIM 的维度
    if size_average:
        ssim_map = ssim_map.mean([1, 2, 3])  # 平均化后返回每个样本的 SSIM
        return ssim_map.unsqueeze(1)  # 保留维度 [B, 1]
    else:
        return ssim_map  # 返回原始 SSIM 图，保留批次维度
