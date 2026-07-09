"""DCRM bottleneck module."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class LayerNormFunction(torch.autograd.Function):
    """Channel-wise LayerNorm for 4D feature maps: [N, C, H, W]."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        """Normalize each spatial location across the channel dimension."""
        _, channels, _, _ = x.size()

        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        y = (x - mean) / torch.sqrt(variance + eps)

        ctx.eps = eps
        ctx.save_for_backward(y, variance, weight)

        return weight.view(1, channels, 1, 1) * y + bias.view(1, channels, 1, 1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass for the custom channel-wise LayerNorm."""
        eps = ctx.eps
        y, variance, weight = ctx.saved_tensors

        grad = grad_output * weight.view(1, -1, 1, 1)
        mean_grad = grad.mean(dim=1, keepdim=True)
        mean_grad_y = (grad * y).mean(dim=1, keepdim=True)

        grad_x = (grad - y * mean_grad_y - mean_grad) / torch.sqrt(variance + eps)
        grad_weight = (grad_output * y).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_x, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    """A 2D LayerNorm layer for CNN feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Split channels into two halves and multiply them element-wise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class DepthwiseConv(nn.Module):
    """Depthwise convolution branch with configurable dilation."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DCRMBottleneck(nn.Module):
    """DCRM/DCR bottleneck with multi-dilation depthwise branches.

    The block contains two main parts:
        1. CAU-like part: normalization, channel expansion, and multi-dilation
           depthwise convolution branches.
        2. ARU-like part: SimpleGate-based channel interaction, lightweight
           channel attention, projection, and residual scaling.
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        dilations: Sequence[int] = (1, 4, 9),
        extra_depthwise: bool = True,
    ) -> None:
        super().__init__()

        expanded_channels = channels * dw_expand
        gated_channels = expanded_channels // 2

        if expanded_channels % 2 != 0:
            raise ValueError("expanded_channels must be even for SimpleGate.")
        if len(dilations) == 0:
            raise ValueError("At least one dilation value must be provided.")

        self.expanded_channels = expanded_channels
        self.norm = LayerNorm2d(channels)

        # Channel expansion before depthwise branch aggregation.
        self.expand_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=expanded_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Optional local depthwise convolution before multi-dilation branches.
        self.extra_depthwise_conv = (
            nn.Conv2d(
                in_channels=expanded_channels,
                out_channels=expanded_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=expanded_channels,
                bias=True,
            )
            if extra_depthwise
            else nn.Identity()
        )

        # Multi-scale depthwise branches with different dilation rates.
        self.cau_branches = nn.ModuleList(DepthwiseConv(expanded_channels, dilation=dilation) for dilation in dilations)

        # Simple channel interaction and lightweight attention.
        self.simple_gate = SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=gated_channels,
                out_channels=gated_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        # Project features back to the original channel dimension.
        self.project_conv = nn.Conv2d(
            in_channels=gated_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Learnable residual scaling. It starts from zero for stable training.
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # CAU-like feature extraction.
        x = self.norm(x)
        x = self.expand_conv(x)
        x = self.extra_depthwise_conv(x)

        cau = torch.zeros_like(x)
        for branch in self.cau_branches:
            cau = cau + branch(x)

        # ARU-like gated channel reweighting.
        aru = self.simple_gate(cau)
        aru = self.channel_attention(aru) * aru
        aru = self.project_conv(aru)

        return identity + self.beta * aru


class DCRM_bottleneck(DCRMBottleneck):
    """Backward-compatible wrapper for the original class name and arguments."""

    def __init__(
        self,
        c1: int,
        DW_Expand: int = 2,
        dilations: Sequence[int] = (1, 4, 9),
        extra_depth_wise: bool = True,
    ) -> None:
        super().__init__(
            channels=c1,
            dw_expand=DW_Expand,
            dilations=dilations,
            extra_depthwise=extra_depth_wise,
        )


# Optional alias for the original depthwise-convolution class name.
Conv_DW = DepthwiseConv
