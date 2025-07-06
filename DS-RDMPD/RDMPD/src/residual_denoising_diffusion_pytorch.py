import copy
import glob
import math
import os
import random
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from dataset import myImageFlodertrain, myImageFlodertest
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from torch import einsum, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])


# helpers functions


def set_seed(SEED):  # 每次运行相同的代码和数据时能够得到相同的结果。
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def exists(x):
    # 　检测ｘ是否为空
    return x is not None


def default(val, d):
    # 函数用于提供一个默认值 d
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):  # 输入的参数原封不动地返回
    return t


def cycle(dl):  # 接收一个数据加载器（dl）并无限次地循环遍历它
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):  # 整数平方根
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):  # 将 num 分成多个部分，每个部分的大小是 divisor，如果有剩余部分，则最后一部分的大小为剩余部分。
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions


def normalize_to_neg_one_to_one(img):
    # 　从 [0, 1] 范围映射到 [-1, 1] 范围
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5


# small helper modules


class Residual(nn.Module):  # 构建具有跳跃连接的网络模块
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    # 　上采样模块，用于上采样输入的特征图。
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    # 计算权重的均值和方差，对卷积层的权重进行标准化
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):  # 每个样本的特征维度上进行归一化
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):  # 首先对输入数据进行归一化（LayerNorm），然后将其传递给后续的函数（fn）
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    # 　这个模块生成基于正弦函数的时间 / 位置嵌入
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    # 　根据需要生成随机的或学习的正弦位置嵌入
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules


class Block(nn.Module):
    # 构建网络的基础模块
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    # 通过添加时间信息，实现动态调整特征图的生成。
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):  # 线性注意力
    # 线性注意力机制的实现
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        v = v / (h * w)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    # 注意力机制
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        # 标准的注意力机制
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            self_condition=False,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            condition=False,
            input_condition=False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels + channels * \
                         (1 if self_condition else 0) + channels * \
                         (1 if condition else 0) + channels * (1 if input_condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class ConditionalFeatureFusionLayer(torch.nn.Module):
    def __init__(self, nf=64, n_condition=64):
        super(ConditionalFeatureFusionLayer, self).__init__()
        self.nf = nf
        self.n_condition = n_condition

        # 保持特征提取网络
        self.feature_conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.1, True),
            torch.nn.Conv2d(64, nf, kernel_size=3, padding=1)
        )
        self.condition_conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.1, True),
            torch.nn.Conv2d(64, n_condition, kernel_size=3, padding=1)
        )

        self.mul_conv1 = torch.nn.Conv2d(nf + n_condition, 128, kernel_size=3, padding=1)
        self.mul_conv2 = torch.nn.Conv2d(128, nf, kernel_size=3, padding=1)
        self.add_conv1 = torch.nn.Conv2d(nf + n_condition, 128, kernel_size=3, padding=1)
        self.add_conv2 = torch.nn.Conv2d(128, nf, kernel_size=3, padding=1)
        self.lrelu = torch.nn.LeakyReLU(0.1, True)

    def forward(self, features, conditions, target_size=None):
        # 特征提取
        x = self.feature_conv(features)
        cond = self.condition_conv(conditions)

        # 如果提供了目标尺寸，调整到目标尺寸
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
            cond = F.interpolate(cond, size=target_size, mode='bilinear', align_corners=True)
        # 否则确保条件特征与输入特征尺寸匹配
        elif x.shape[2:] != cond.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 特征调制
        cat_input = torch.cat((x, cond), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.lrelu(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.lrelu(self.add_conv1(cat_input)))

        return x * mul + add


class UnetRes(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            self_condition=False,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            share_encoder=1,
            condition=False,
            input_condition=False
    ):
        super().__init__()
        self.condition = condition
        self.input_condition = input_condition
        self.share_encoder = share_encoder
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = out_dim if out_dim is not None else default_out_dim
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.self_condition = self_condition
        # self.fcb = FCB(channel=channels)

        # Determine dimensions
        if self.share_encoder == 1:
            input_channels = channels + channels * \
                             (1 if self_condition else 0) + \
                             channels * (1 if condition else 0) + channels * \
                             (1 if input_condition else 0)
            init_dim = init_dim if init_dim is not None else dim
            self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

            dims = [init_dim] + [dim * m for m in dim_mults]
            in_out = list(zip(dims[:-1], dims[1:]))

            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

            # Time embeddings
            time_dim = dim * 4

            if self.random_or_learned_sinusoidal_cond:
                sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                    learned_sinusoidal_dim, random_fourier_features)
                fourier_dim = learned_sinusoidal_dim + 1
            else:
                sinu_pos_emb = SinusoidalPosEmb(dim)
                fourier_dim = dim

            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )

            # Layers
            self.downs = nn.ModuleList([])
            self.ups = nn.ModuleList([])
            self.ups_no_skip = nn.ModuleList([])
            num_resolutions = len(in_out)

            for ind, (dim_in, dim_out) in enumerate(in_out):
                is_last = ind >= (num_resolutions - 1)

                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                        dim_in, dim_out, 3, padding=1)
                ]))

            mid_dim = dims[-1]
            self.mid_block1 = block_klass(
                mid_dim, mid_dim, time_emb_dim=time_dim)
            self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
            self.mid_block2 = block_klass(
                mid_dim, mid_dim, time_emb_dim=time_dim)

            reversed_dim_mults = list(reversed(dim_mults))
            dims_reversed = list(reversed(dims))

            self.cff_layers = nn.ModuleList([
                ConditionalFeatureFusionLayer(
                    nf=dims_reversed[i],  # 当前层的特征通道数
                    n_condition=dims_reversed[i]  # 条件的通道数应该匹配当前层
                )
                for i in range(len(reversed_dim_mults))
            ])

            self.cff_layers_no_skip = nn.ModuleList([
                ConditionalFeatureFusionLayer(
                    nf=dims_reversed[i],
                    n_condition=dims_reversed[i]
                )
                for i in range(len(reversed_dim_mults))
            ])

            for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
                is_last = ind == (len(in_out) - 1)

                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out,
                                time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in, dim_out,
                                time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                        dim_out, dim_in, 3, padding=1)
                ]))

                self.ups_no_skip.append(nn.ModuleList([
                    block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                        dim_out, dim_in, 3, padding=1)
                ]))

            self.final_res_block_1 = block_klass(
                dim, dim, time_emb_dim=time_dim)
            self.final_conv_1 = nn.Conv2d(dim, self.out_dim, 1)

            self.final_res_block_2 = block_klass(
                dim * 2, dim, time_emb_dim=time_dim)
            self.final_conv_2 = nn.Conv2d(dim, self.out_dim, 1)
        elif self.share_encoder == 0:
            self.unet0 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition)
            self.unet1 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition)
        elif self.share_encoder == -1:
            self.unet0 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition)

    def forward(self, x, time, x_self_cond=None):
        if self.share_encoder == 1:  # 共享编码器
            if self.self_condition:  # 自条件
                x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
                x = torch.cat((x_self_cond, x), dim=1)

            C = self.channels
            x_input = x[:, C:2 * C, :, :]  # 提取 x_input
            if self.input_condition:
                x_input_condition = x[:, 2 * C:, :, :]  # 提取 x_input_condition
            else:
                x_input_condition = None

            x = self.init_conv(x)  # 初始卷积
            r = x.clone()  # 跳跃连接
            # r = self.fcb(r)
            t = self.time_mlp(time)  # 时间嵌入
            h = []

            # 下采样阶段
            for block1, block2, attn, downsample in self.downs:
                x = block1(x, t)
                h.append(x)
                x = block2(x, t)
                x = attn(x)
                h.append(x)
                x = downsample(x)

            # 瓶颈层
            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)

            out_res = x

            # 上采样阶段（不使用跳跃连接）
            for idx, (block1, block2, attn, upsample) in enumerate(self.ups_no_skip):
                out_res = block1(out_res, t)
                out_res = block2(out_res, t)
                out_res = attn(out_res)

                if self.input_condition and x_input_condition is not None:
                    cff_no_skip = self.cff_layers_no_skip[idx]

                    modulated = cff_no_skip(x_input, x_input_condition, target_size=out_res.shape[2:])
                    out_res = out_res + modulated
                else:
                    cff_no_skip = self.cff_layers_no_skip[idx]
                    modulated = cff_no_skip(x_input, x_input, target_size=out_res.shape[2:])
                    out_res = out_res + modulated

                out_res = upsample(out_res)

            out_res = self.final_res_block_1(out_res, t)
            out_res = self.final_conv_1(out_res)


            for idx, (block1, block2, attn, upsample) in enumerate(self.ups):
                # 从下采样阶段弹出对应的特征图
                skip = h.pop()
                x = torch.cat((x, skip), dim=1)
                x = block1(x, t)


                if self.input_condition and x_input_condition is not None:
                    cff = self.cff_layers[idx]
                    # 传入目标尺寸
                    modulated = cff(x_input, x_input_condition, target_size=x.shape[2:])
                    x = x + modulated
                else:
                    cff = self.cff_layers[idx]
                    modulated = cff(x_input, x_input, target_size=x.shape[2:])
                    x = x + modulated

                x = torch.cat((x, h.pop()), dim=1)
                x = block2(x, t)
                x = attn(x)
                x = upsample(x)

            # 最终处理
            x = torch.cat((x, r), dim=1)
            x = self.final_res_block_2(x, t)
            out_res_add_noise = self.final_conv_2(x)

            return out_res, out_res_add_noise
        elif self.share_encoder == 0:
            return self.unet0(x, time, x_self_cond=x_self_cond), self.unet1(x, time, x_self_cond=x_self_cond)
        elif self.share_encoder == -1:
            return [self.unet0(x, time, x_self_cond=x_self_cond)]


# gaussian diffusion trainer class


def extract(a, t, x_shape):  # 张量 a 中根据给定的时间步长 t 提取相应的值，并将其调整为与 x_shape 形状匹配的输出。
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gen_coefficients(timesteps, schedule="increased", sum_scale=1):  # 加噪-噪声衰减系数
    if schedule == "increased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        scale = 0.5 * timesteps * (timesteps + 1)
        alphas = x / scale
    elif schedule == "decreased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        x = torch.flip(x, dims=[0])
        scale = 0.5 * timesteps * (timesteps + 1)
        alphas = x / scale
    elif schedule == "average":
        alphas = torch.full([timesteps], 1 / timesteps, dtype=torch.float64)
    else:
        alphas = torch.full([timesteps], 1 / timesteps, dtype=torch.float64)
    assert alphas.sum() - torch.tensor(1) < torch.tensor(1e-10)

    return alphas * sum_scale


class ResidualDiffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            objective='pred_res',
            ddim_sampling_eta=0.,
            condition=False,
            sum_scale=None,
            input_condition=False,
            input_condition_mask=False,
    ):
        super().__init__()
        assert not (
                type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.input_condition = input_condition
        self.input_condition_mask = input_condition_mask

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            ddim_sampling_eta = 0.
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        alphas = gen_coefficients(timesteps, schedule="decreased")
        alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2 = gen_coefficients(timesteps, schedule="increased", sum_scale=self.sum_scale)
        betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)
        betas_cumsum = torch.sqrt(betas2_cumsum)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        posterior_variance = betas2 * betas2_cumsum_prev / betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val):
            return self.register_buffer(
                name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1 - alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev / betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                                                 alphas_cumsum_prev - betas2_cumsum_prev * alphas) / betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2 / betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):  # 从残差中反推出噪声
        return (
                (x_t - x_input - (extract(self.alphas_cumsum, t, x_t.shape) - 1)
                 * pred_res) / extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):  # 从噪声和输入图像中估计出图像的无噪声版本
        return (
                (x_t - extract(self.alphas_cumsum, t, x_t.shape) * x_input -
                 extract(self.betas_cumsum, t, x_t.shape) * noise) / extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):  # 输入图像 x_t、残差 x_res 和噪声，预测恢复的图像起始状态
        return (
                x_t - extract(self.alphas_cumsum, t, x_t.shape) * x_res -
                extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):  # 用于采样过程中，将预测的噪声和残差转化为概率意义上的后验图像状态
        return (x_t - extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape) / extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        # 计算图像的后验均值和方差，这通常涉及当前图像、预测的残差、恢复图像以及时间步长。
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
                extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
                extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, x_input_condition=0, x_self_cond=None, clip_denoised=True):
        if not self.condition:
            x_in = x
        else:
            if self.input_condition:
                x_in = torch.cat((x, x_input, x_input_condition), dim=1)
            else:
                x_in = torch.cat((x, x_input), dim=1)
        model_output = self.model(x_in, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            pred_res = model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(x, t, pred_res, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_res_add_noise':
            pred_res = model_output[0]
            pred_noise = model_output[1] - model_output[0]
            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(x, t, pred_res, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0_noise':
            pred_res = x_input - model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == 'pred_x0_add_noise':
            x_start = model_output[0]
            pred_noise = model_output[1] - model_output[0]
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t, x_input_condition=0, x_self_cond=None):
        # 采样过程中使用，提供每个时间步的采样均值和方差，供去噪步骤使用
        preds = self.model_predictions(
            x_input, x, t, x_input_condition, x_self_cond)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int, x_input_condition=0, x_self_cond=None):
        # 从当前图像状态生成一个去噪后的版本，逐步逼近目标图像。
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x_input, x=x, t=batched_times, x_input_condition=x_input_condition, x_self_cond=x_self_cond)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        # 生成过程的主要循环逻辑，负责调用 p_sample 函数完成逐步去噪
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]

        batch, device = shape[0], self.betas.device

        if self.condition:
            img = x_input + math.sqrt(self.sum_scale) * \
                  torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None

        if not last:
            img_list = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(
                x_input, img, t, x_input_condition, self_cond)

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise] + img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def ddim_sample(self, x_input, shape, last=True):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]

        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            img = x_input + math.sqrt(self.sum_scale) * \
                  torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_x_start"

        if not last:
            img_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(
                x_input, img, time_cond, x_input_condition, self_cond)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum - alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum - betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2 * betas2_cumsum_next / betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                                                                                betas2_cumsum_next - sigma2).sqrt() / betas_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)

            if type == "use_pred_noise":
                img = img - alpha * pred_res - \
                      (betas_cumsum - (betas2_cumsum_next - sigma2).sqrt()) * \
                      pred_noise + sigma2.sqrt() * noise
            elif type == "use_x_start":
                img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum * img + \
                      (1 - sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * x_start + \
                      (
                                  alpha_cumsum_next - alpha_cumsum * sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * pred_res + \
                      sigma2.sqrt() * noise
            elif type == "special_eta_0":
                img = img - alpha * pred_res - \
                      (betas_cumsum - betas_cumsum_next) * pred_noise
            elif type == "special_eta_1":
                img = img - alpha * pred_res - betas2 / betas_cumsum * pred_noise + \
                      betas * betas2_cumsum_next.sqrt() / betas_cumsum * noise

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise] + img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True):  # 内部会选择调用 p_sample_loop 或 ddim_sample 完成采样过程。
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if self.condition:
            if self.input_condition and self.input_condition_mask:
                x_input[0] = normalize_to_neg_one_to_one(x_input[0])
            else:
                x_input = normalize_to_neg_one_to_one(x_input)
            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)
        return sample_fn(x_input, size, last=last)

    def q_sample(self, x_start, x_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))  # 加噪过程

        return (x_start + extract(self.alphas_cumsum, t, x_start.shape) * x_res +
                extract(self.betas_cumsum, t, x_start.shape) * noise)  # alphas_cumsum 累计的系数

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, imgs, t, noise=None):
        if not isinstance(imgs, list) or len(imgs) < 2:
            raise ValueError("Expected imgs to be a list with at least two tensors: GT and input images.")

        if isinstance(imgs, list):  # Condition
            if self.input_condition:
                x_input_condition = imgs[2]
            else:
                x_input_condition = 0
            x_input = imgs[1]
            x_start = imgs[0]
        else:  # Generation
            x_input = 0
            x_start = imgs
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_res = x_input - x_start

        b, c, h, w = x_start.shape

        # noise sample
        x = self.q_sample(x_start, x_res, t, noise=noise)  # 加噪

        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x_input, x, t,
                                                     x_input_condition if self.input_condition else 0).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        if not self.condition:
            x_in = x
        else:
            if self.input_condition:
                x_in = torch.cat((x, x_input, x_input_condition), dim=1)
            else:
                x_in = torch.cat((x, x_input), dim=1)

        model_out = self.model(x_in, t, x_self_cond)

        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res)
            target.append(noise)

            pred_res = model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_res_add_noise':
            target.append(x_res)
            target.append(x_res + noise)

            pred_res = model_out[0]
            pred_noise = model_out[1] - model_out[0]
        elif self.objective == 'pred_x0_noise':
            target.append(x_start)
            target.append(noise)

            pred_res = x_input - model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_x0_add_noise':
            target.append(x_start)
            target.append(x_start + noise)

            pred_res = x_input - model_out[0]
            pred_noise = model_out[1] - model_out[0]
        elif self.objective == "pred_noise":
            target.append(noise)

            pred_noise = model_out[0]
        elif self.objective == "pred_res":
            target.append(x_res)
            pred_res = model_out[0]
        else:
            raise ValueError(f'unknown objective {self.objective}')

        u_loss = False
        if u_loss:
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, x, t)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, x, t)
            loss = 10000 * self.loss_fn(x_u, u_gt, reduction='none')
        else:
            if self.objective == "pred_res":
                loss = self.loss_fn(model_out[0], target[0], reduction='none')
            else:
                loss = 0
                for i in range(min(len(model_out), len(target))):  # 防止索引越界
                    loss += self.loss_fn(model_out[i], target[i], reduction='none')

        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        if isinstance(img, list):
            b, c, h, w, device, img_size, = * \
                img[0].shape, img[0].device, self.image_size
        else:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        if self.input_condition and self.input_condition_mask:
            img[0] = normalize_to_neg_one_to_one(img[0])
            img[1] = normalize_to_neg_one_to_one(img[1])
        else:
            img = normalize_to_neg_one_to_one(img)

        return self.p_losses(img, t, *args, **kwargs)


# trainer class
class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            train_folder,
            test_folder,
            train_batch_size=16,
            train_lr=8e-5,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results/sample',
            fp16=False,
            split_batches=True,
            equalizeHist=False,
            crop_patch=False,
    ):
        # Accelerator settings
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.model = diffusion_model
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition_type = 3  # Always use three input tensors

        # Dataset settings
        transform = T.Compose([T.ToTensor()])

        # Define test and training data loaders
        self.sample_dataset = myImageFlodertest(
            test_folder, transform=transform, resize=True, resize_size=self.image_size)
        self.sample_loader = self.accelerator.prepare(
            DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True, pin_memory=True, num_workers=4))

        self.dl = self.accelerator.prepare(
            DataLoader(myImageFlodertrain(train_folder, transform=transform, crop=crop_patch, resize=True,
                                          resize_size=self.image_size),
                       batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=4))

        # Optimizer and EMA settings
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.set_results_folder(results_folder)

        # Training step initialization
        self.step = 0
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.device = self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
        }

        if hasattr(self.accelerator, "scaler") and self.accelerator.scaler is not None:
            data['scaler'] = self.accelerator.scaler.state_dict()

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        path = Path(self.results_folder / f'model-{milestone}.pt')
        if path.exists():
            data = torch.load(str(path), map_location=self.device)
            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])
            self.step = data['step']
            self.opt.load_state_dict(data['opt'])
            self.ema.load_state_dict(data['ema'])
            if hasattr(self.accelerator, "scaler") and "scaler" in data:
                self.accelerator.scaler.load_state_dict(data['scaler'])
            print("Model loaded - " + str(path))

        self.ema.to(self.device)

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not self.accelerator.is_main_process) as pbar:
            data_iter = iter(self.dl)
            while self.step < self.train_num_steps:
                try:
                    data = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dl)
                    data = next(data_iter)
                data = [item.to(self.device) for item in data]

                total_loss = 0.
                for _ in range(2):
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / 2
                        total_loss += loss.item()
                    self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()
                self.step += 1
                if self.accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()
                    if self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.sample(milestone)
                        if self.step % (self.save_and_sample_every * 10) == 0:
                            self.save(milestone)
                pbar.set_description(f'Loss: {total_loss:.4f}')
                pbar.update(1)
        self.accelerator.print('Training complete')

    def sample(self, milestone, last=True, FID=False):
        self.ema.ema_model.eval()  # 评估模式

        with torch.no_grad():
            x_input_sample = next(iter(self.sample_loader))
            x_input_sample = [item.to(self.device) for item in x_input_sample]
            show_x_input_sample = x_input_sample  # Display three tensors

            model_output = self.ema.ema_model.sample(x_input_sample[1:], batch_size=self.num_samples, last=last)
            if not isinstance(model_output, list) or len(model_output) == 0:
                print(f"Error: Model output is empty or scalar at step {self.step}")
                return

            all_images_list = show_x_input_sample + list(model_output)
            all_images = torch.cat(all_images_list, dim=0)
            nrow = int(math.sqrt(self.num_samples)) if last else all_images.shape[0]

            file_name = f'sample-{milestone}.png'
            utils.save_image(all_images, str(self.results_folder / file_name), nrow=nrow)
            print("Sample saved - " + file_name)
        return milestone

    def test(self, sample=False, last=True, FID=False):
        print("Testing start")
        self.ema.ema_model.eval()
        loader = DataLoader(dataset=self.sample_dataset, batch_size=1)
        i = 0

        for items in loader:
            file_name = f"test_sample_{i}.png"
            i += 1

            with torch.no_grad():
                x_input_sample = [item.to(self.device) for item in items]
                show_x_input_sample = x_input_sample

                if sample:
                    all_images_list = show_x_input_sample + list(
                        self.ema.ema_model.sample(x_input_sample[1:], batch_size=self.num_samples))
                else:
                    all_images_list = list(self.ema.ema_model.sample(
                        x_input_sample[1:], batch_size=self.num_samples, last=last))
                    all_images_list = [all_images_list[-1]]

                all_images = torch.cat(all_images_list, dim=0)
                nrow = int(math.sqrt(self.num_samples)) if last else all_images.shape[0]
                utils.save_image(all_images, str(self.results_folder / file_name), nrow=nrow)
                print("Test sample saved - " + file_name)
        print("Testing end")

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)