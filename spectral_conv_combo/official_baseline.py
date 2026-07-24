import torch
import torch.nn as nn
import torch.fft
import time
import numpy as np

class SpectralConv2d(nn.Module):
    """2D Spectral Convolution 算子 - FNO核心组件

    计算流程: FFT -> 频域复数矩阵乘(截断低频) -> IFFT
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # 第一维频率截断模态数
        self.modes2 = modes2  # 第二维频率截断模态数

        scale = 1.0 / (in_channels * out_channels)
        # 可学习的频域权重 (两组: 正频率区域 + 负频率区域)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        """复数批量矩阵乘: (B, C_in, H, W) x (C_in, C_out, H, W) -> (B, C_out, H, W)"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        前向计算
        Args:
            x: [B, C_in, H, W] 实数张量
        Returns:
            [B, C_out, H, W] 实数张量
        """
        B, C_in, H, W = x.shape

        # Step 1: 2D实数FFT (rfft2), 输出shape: [B, C_in, H, W//2+1] (复数)
        x_ft = torch.fft.rfft2(x)

        # Step 2: 在频域进行低频模态的复数矩阵乘
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                            dtype=torch.cfloat, device=x.device)

        # 正频率部分 [:modes1, :modes2]
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        # 负频率部分 [-modes1:, :modes2]
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Step 3: 逆FFT回空间域
        x_out = torch.fft.irfft2(out_ft, s=(H, W))
        return x_out


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution 算子 (进阶)"""
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(B, self.out_channels, D, H, W // 2 + 1,
                            dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        x_out = torch.fft.irfftn(out_ft, s=(D, H, W))
        return x_out