"""CNN pequeña entrenada desde cero (recomendación de la consigna).

Arquitectura (entrada 3 x 128 x 128):

    [Conv3x3(32) - BN - ReLU - MaxPool2]   -> 32 x 64 x 64
    [Conv3x3(64) - BN - ReLU - MaxPool2]   -> 64 x 32 x 32
    [Conv3x3(128) - BN - ReLU - MaxPool2]  -> 128 x 16 x 16
    GlobalAvgPool                          -> 128
    Dropout(0.3) - Linear(128, 64) - ReLU - Linear(64, K)

Cada convolución calcula  y = sum_{c,i,j} w_{c,i,j} * x_{c, u+i, v+j} + b,
compartiendo pesos espacialmente (pocos parámetros, invariancia a traslación).
GlobalAvgPool en vez de Flatten reduce los parámetros de ~4M a ~100k, lo
cual es clave con un dataset pequeño para limitar el sobreajuste.
"""
from __future__ import annotations

import torch
from torch import nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 3, widths=(32, 64, 128), dropout: float = 0.3):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers.append(conv_block(c, w))
            c = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, 64), nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # devuelve logits
        return self.head(self.pool(self.features(x)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
