"""
Impala CNN网络架构的PyTorch实现
基于OpenAI baselines的build_impala_cnn，用于Procgen环境
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from loguru import logger


class ResidualBlock(nn.Module):
    """Impala CNN的残差块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果通道数不同，需要1x1卷积来调整residual
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        # 如果通道数不同，调整residual
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        return out + residual


class ImpalaCNN(BaseFeaturesExtractor):
    """
    Impala CNN特征提取器
    基于OpenAI baselines的build_impala_cnn实现

    :param observation_space: 观察空间
    :param depths: 每个阶段的通道数，默认[16, 32, 32]
    :param emb_size: 最终特征维度，默认256
    """
    def __init__(self, observation_space: spaces.Box, depths=[16, 32, 32], emb_size=256):
        super().__init__(observation_space, emb_size)

        n_input_channels = observation_space.shape[0]
        logger.info(f"ImpalaCNN: Input channels={n_input_channels}, depths={depths}, emb_size={emb_size}")

        # 构建卷积层
        layers = []
        in_channels = n_input_channels

        for depth in depths:
            # 每个阶段包含一个残差块
            layers.append(ResidualBlock(in_channels, depth))
            # 最大池化层，将空间维度减半
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            in_channels = depth

        self.conv_layers = nn.Sequential(*layers)

        # 添加Flatten层（与stable-baselines3的NatureCNN一致）
        self.flatten = nn.Flatten()

        # 计算卷积后的特征维度
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            conv_out = self.conv_layers(sample_input)
            n_flatten = conv_out.shape[1] * conv_out.shape[2] * conv_out.shape[3]

        logger.info(f"ImpalaCNN: Conv output shape={conv_out.shape}, flattened size={n_flatten}")

        # 全连接层
        self.fc = nn.Linear(n_flatten, emb_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 卷积层
        x = self.conv_layers(observations)
        # 展平（使用Flatten层，自动处理非连续张量）
        x = self.flatten(x)
        # 全连接层
        x = F.relu(self.fc(x))
        return x
