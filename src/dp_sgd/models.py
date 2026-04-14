from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


def _group_norm(num_channels: int) -> nn.GroupNorm:
    groups = min(32, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)



class MnistConvNet(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, padding=1, bias=False),
            _group_norm(width),
            nn.ReLU(inplace=False),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            _group_norm(width * 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1, bias=False),
            _group_norm(width * 4),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(width * 4, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        outputs = torch.flatten(outputs, start_dim=1)
        outputs = self.dropout(outputs)
        return self.classifier(outputs)


class _CifarBasicBlock(nn.Module):
    """BasicBlock with GroupNorm and non-inplace residual add (Opacus-compatible)."""

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = _group_norm(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = _group_norm(planes)
        self.relu2 = nn.ReLU(inplace=False)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                _group_norm(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)  # non-inplace add — required by Opacus
        return self.relu2(out)


class _CifarResNet18(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32×32) with GroupNorm and no inplace ops."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = _group_norm(64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_planes: int, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        cur = in_planes
        for s in strides:
            layers.append(_CifarBasicBlock(cur, planes, s))
            cur = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _build_cifar_resnet18(num_classes: int) -> nn.Module:
    return _CifarResNet18(num_classes=num_classes)


def build_model(model_config: ModelConfig, dataset_name: str) -> nn.Module:
    model_name = model_config.name.lower()
    if model_name == "auto":
        model_name = "mnist_cnn" if dataset_name.lower() == "mnist" else "cifar_resnet18"

    if model_name in {"mnist_cnn", "cnn"}:
        return MnistConvNet(
            num_classes=model_config.num_classes,
            width=model_config.width,
            dropout=model_config.dropout,
        )
    if model_name in {"cifar_resnet18", "resnet18"}:
        return _build_cifar_resnet18(num_classes=model_config.num_classes)
    raise ValueError(f"Unsupported model '{model_config.name}'.")
