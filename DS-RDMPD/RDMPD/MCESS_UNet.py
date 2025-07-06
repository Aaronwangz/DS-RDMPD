import math
import torch
from torch import nn
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class cSE(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, in_channel // 2, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel // 2, 1, kernel_size=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.residual = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')
        return x * y.expand_as(x) + self.residual(x)


class sSE(nn.Module):
    def __init__(self, in_channel):
        super(sSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel // 2, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel // 2, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.residual = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y + self.residual(x)


class scSE(nn.Module):
    def __init__(self, in_channel):
        super(scSE, self).__init__()
        self.cSE = cSE(in_channel)
        self.sSE = sSE(in_channel)
        # self.fourier_attn = FourierAttention(in_channel)
        self.residual = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        # U_fattn = self.fourier_attn(U)
        U_res = self.residual(U)
        return torch.max(U_cse + U_res, U_sse + U_res)


class Efficient_Attention_Gate(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_groups=32):
        super(Efficient_Attention_Gate, self).__init__()
        self.num_groups = num_groups
        self.grouped_conv_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )

        self.grouped_conv_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.grouped_conv_g(g)
        x1 = self.grouped_conv_x(x)
        psi = self.psi(self.relu(x1 + g1))
        out = x * psi
        out += x

        return out


class EfficientChannelAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return v


class ESS(nn.Module):
    def __init__(self, in_dim, is_bottom=False):
        super().__init__()
        self.is_bottom = is_bottom
        if not is_bottom:
            self.EAG = Efficient_Attention_Gate(in_dim, in_dim, in_dim)
        else:
            self.EAG = nn.Identity()
        self.ECA = EfficientChannelAttention(in_dim * 2)
        self.SE = scSE(in_channel=in_dim * 2)

        self.SA = SpatialAttention()

    def forward(self, x, skip):
        if not self.is_bottom:
            EAG_skip = self.EAG(x, skip)
            x = torch.cat((EAG_skip, x), dim=1)
            # x = EAG_skip + x
        else:
            x = self.EAG(x)
        x = self.ECA(x) * x
        x = self.SE(x)
        x = self.SA(x) * x

        return x


class ASFusion(nn.Module):
    def __init__(self, in_features):
        super(ASFusion, self).__init__()

        self.shallow = nn.Sequential(
            nn.Conv2d(
                in_features,
                in_features,
                kernel_size=3,
                stride=1,
                padding=3,
                padding_mode="reflect",
                dilation=2,
                groups=in_features,
            ),
            nn.BatchNorm2d(in_features),
            nn.ReLU(True),
            nn.Conv2d(
                in_features,
                in_features,
                kernel_size=3,
                stride=1,
                padding=3,
                padding_mode="reflect",
                dilation=2,
                groups=in_features,
            ),
            nn.BatchNorm2d(in_features),
            nn.ReLU(True),
        )
        self.MLP_A = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, in_features * 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_features * 2, in_features, 1, bias=False),
            nn.ReLU(True),
        )
        self.MLP_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, in_features * 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_features * 2, in_features, 1, bias=False),
            nn.ReLU(True),
        )

    def forward(self, x):
        mid = self.shallow(x)

        t = self.MLP_T(mid)
        b, c, h, w = t.shape
        a = self.MLP_A(mid)
        one = torch.ones((b, c, h, w)).to(x.device)
        return x * t + a * (one - t)


class MSM(nn.Module):
    """多分支特征提取模块"""
    def __init__(self, in_channels, filters):
        super(MSM, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.LeakyReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        o1 = self.branch1(x)
        o2 = self.branch2(x)
        o3 = self.branch3(x)
        o4 = self.branch4(x)
        return torch.cat([o1, o2, o3, o4], dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class CAM(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction_ratio=4):
        super(CAM, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.Sigmoid(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attention_weights = self.module(x)
        attention_weights = attention_weights.unsqueeze(2).unsqueeze(3)
        return x * attention_weights.expand_as(x)


class AttentionModule(nn.Module):
    """组合注意力模块，结合多分支特征提取和通道注意力"""
    def __init__(self, in_channels, filters, reduction_ratio=4):
        super(AttentionModule, self).__init__()
        self.inc = MSM(in_channels=in_channels, filters=filters)
        self.adjust_channel = nn.Conv2d(filters * 4, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.cam = CAM(in_channels=in_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        inc_out = self.inc(x)
        adjusted_out = self.adjust_channel(inc_out)
        attention_out = self.cam(adjusted_out)
        return attention_out


class MSCABlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MSCABlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu2 = nn.ReLU(inplace=True)
        self.attention = AttentionModule(in_channels=ch_out, filters=ch_out // 4, reduction_ratio=4)
        self.residual_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.attention(out)
        return out + residual


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class U_Net(nn.Module):
    def __init__(self, dim=32):
        super(U_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Encoding path
        self.Conv1 = MSCABlock(3, dim)
        self.Conv2 = MSCABlock(dim, dim*2)
        self.Conv3 = MSCABlock(dim*2, dim*4)
        self.Conv4 = MSCABlock(dim*4, dim*8)
        self.Conv5 = MSCABlock(dim*8, dim*16)

        # Decoding path
        self.Up5 = UpConv(dim*16, dim*8)
        self.Up_conv5 = MSCABlock(dim*16, dim*8)

        self.Up4 = UpConv(dim*8, dim*4)
        self.Up_conv4 = MSCABlock(dim*8, dim*4)

        self.Up3 = UpConv(dim*4, dim*2)
        self.Up_conv3 = MSCABlock(dim*4, dim*2)

        self.Up2 = UpConv(dim*2, dim)
        self.Up_conv2 = MSCABlock(dim*2, dim)

        # Final 1x1 Convolution to get the desired number of classes
        self.Final_conv = nn.Conv2d(dim, 3, kernel_size=1, stride=1, padding=0)

        self.neck = ASFusion(dim*16)

        self.eff1 = ESS(in_dim=dim*8, is_bottom=False)
        self.eff2 = ESS(in_dim=dim*4, is_bottom=False)
        self.eff3 = ESS(in_dim=dim*2, is_bottom=False)
        self.eff4 = ESS(in_dim=dim, is_bottom=False)

        # 用于调整通道数以匹配残差连接
        self.adjust_channel4 = nn.Conv2d(dim*8, dim*16, kernel_size=1, stride=1, padding=0, bias=True)
        self.adjust_channel3 = nn.Conv2d(dim*4, dim*8, kernel_size=1, stride=1, padding=0, bias=True)
        self.adjust_channel2 = nn.Conv2d(dim*2, dim*4, kernel_size=1, stride=1, padding=0, bias=True)
        self.adjust_channel1 = nn.Conv2d(dim, dim*2, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        x5 = self.neck(x5)

        d5 = self.Up5(x5)
        d5 = self.eff1(x4, d5) + self.adjust_channel4(x4)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.eff2(x3, d4) + self.adjust_channel3(x3)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.eff3(x2, d3) + self.adjust_channel2(x2)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.eff4(x1, d2) + self.adjust_channel1(x1)
        d2 = self.Up_conv2(d2)

        output = self.Final_conv(d2)

        output = output + x

        return output


# Example of instantiation
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化和使用模型
    model = U_Net(dim=64).to(device)
    input_tensor = torch.randn(1, 3, 512, 512).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print("Output size:", output.size())

    # Average SSIM: 0.9322641082876836, Average PSNR: 22.723670734110293
    # epochs:800 lr?

