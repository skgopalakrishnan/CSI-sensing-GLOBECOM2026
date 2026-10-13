"""Tiny frequency-stride autoencoder tailored for SenseFi-style CSI grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["TinyFStrideAutoencoder", "TinyFStrideConfig"]


def _compute_padding(kernel: int, dilation: int) -> int:
    return (kernel - 1) // 2 * dilation


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution followed by pointwise projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        padding = (
            _compute_padding(kernel_size[0], dilation[0]),
            _compute_padding(kernel_size[1], dilation[1]),
        )
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.activation(x)


class TinyTCNLayer(nn.Module):
    """Single dilated temporal convolution with residual connection."""

    def __init__(
        self,
        channels: int,
        kernel_time: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = _compute_padding(kernel_time, dilation)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_time),
            stride=1,
            padding=(0, padding),
            dilation=(1, dilation),
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual


class TinyTCNBlock(nn.Module):
    """Stack of temporal residual layers."""

    def __init__(
        self,
        channels: int,
        kernel_time: int,
        dilations: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TinyTCNLayer(channels, kernel_time, dilation, dropout) for dilation in dilations]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class TinyFStrideConfig:
    in_channels: int
    hidden_dims: Tuple[int, int] = (24, 48)
    encoder_kernels: Tuple[Tuple[int, int], Tuple[int, int]] = ((5, 3), (3, 3))
    encoder_strides: Tuple[Tuple[int, int], Tuple[int, int]] = ((2, 1), (2, 1))
    temporal_kernel: int = 3
    temporal_dilations: Tuple[Sequence[int], Sequence[int]] = ((1, 2), (2, 4))
    dropout: float = 0.05
    output_activation: str | None = None
    latent_channels: Optional[int] = None
    latent_freq: Optional[int] = None


class TinyFStrideAutoencoder(nn.Module):
    """Compact frequency-stride autoencoder with lightweight temporal blocks."""

    def __init__(self, config: TinyFStrideConfig) -> None:
        super().__init__()
        self.config = config
        (h1, h2) = config.hidden_dims

        self.enc_conv1 = DepthwiseSeparableConv(
            config.in_channels,
            h1,
            kernel_size=config.encoder_kernels[0],
            stride=config.encoder_strides[0],
        )
        self.enc_tcn1 = TinyTCNBlock(
            h1,
            kernel_time=config.temporal_kernel,
            dilations=config.temporal_dilations[0],
            dropout=config.dropout,
        )
        self.enc_conv2 = DepthwiseSeparableConv(
            h1,
            h2,
            kernel_size=config.encoder_kernels[1],
            stride=config.encoder_strides[1],
        )
        self.enc_tcn2 = TinyTCNBlock(
            h2,
            kernel_time=config.temporal_kernel,
            dilations=config.temporal_dilations[1],
            dropout=config.dropout,
        )

        self.latent_channels = config.latent_channels or h2
        self.latent_proj = nn.Conv2d(h2, self.latent_channels, kernel_size=1, bias=False)
        self.latent_pool = None
        if config.latent_freq is not None:
            self.latent_pool = nn.AdaptiveAvgPool2d((int(config.latent_freq), None))
        if self.latent_channels == h2:
            self.channel_expand = nn.Identity()
        else:
            self.channel_expand = nn.Conv2d(self.latent_channels, h2, kernel_size=1, bias=False)

        self.dec_deconv1 = nn.ConvTranspose2d(
            h2,
            h1,
            kernel_size=(4, 3),
            stride=config.encoder_strides[1],
            padding=(1, 1),
            bias=False,
        )
        self.dec_norm1 = nn.BatchNorm2d(h1)
        self.dec_tcn1 = TinyTCNBlock(
            h1,
            kernel_time=config.temporal_kernel,
            dilations=config.temporal_dilations[0],
            dropout=config.dropout,
        )
        self.dec_deconv0 = nn.ConvTranspose2d(
            h1,
            h1,
            kernel_size=(4, 3),
            stride=config.encoder_strides[0],
            padding=(1, 1),
            bias=False,
        )
        self.dec_norm0 = nn.BatchNorm2d(h1)
        self.dec_tcn0 = TinyTCNBlock(
            h1,
            kernel_time=config.temporal_kernel,
            dilations=(1,),
            dropout=config.dropout,
        )
        self.output_proj = nn.Conv2d(h1, config.in_channels, kernel_size=1)
        activation = (config.output_activation or "").lower()
        if activation == "tanh":
            self.output_activation = nn.Tanh()
        elif activation in {"sigmoid", "logistic"}:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

    def _encode_impl(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
        """Run the encoder stack and return intermediate tensors."""

        x = self.enc_conv1(x)
        x = self.enc_tcn1(x)
        x = self.enc_conv2(x)
        latent_pre = self.enc_tcn2(x)
        freq_before_pool = latent_pre.shape[-2]

        latent = self.latent_proj(latent_pre)
        pooled_latent = latent
        if self.latent_pool is not None:
            pooled_latent = self.latent_pool(pooled_latent)
        return latent_pre, latent, pooled_latent, freq_before_pool

    def encode(
        self,
        x: Tensor,
        *,
        pool: bool = True,
        return_unpooled: bool = False,
        return_pre_pool: bool = False,
    ):
        """Encode an input tensor without invoking the decoder.

        Args:
            x: Input tensor shaped [B, C, F, T].
            pool: If True, apply latent frequency pooling before returning.
            return_unpooled: When True, append the unpooled latent to outputs.
            return_pre_pool: When True, append the tensor prior to the 1x1 projection.

        Returns:
            The pooled latent tensor by default. If `return_unpooled` and/or
            `return_pre_pool` are enabled, the additional tensors are appended to
            the return tuple in (pooled, unpooled, pre_pool) order.
        """

        latent_pre, latent, pooled_latent, _ = self._encode_impl(x)
        primary = pooled_latent if pool else latent

        outputs = [primary]
        if return_unpooled:
            outputs.append(latent)
        if return_pre_pool:
            outputs.append(latent_pre)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        input_shape = x.shape[-2:]

        latent_pre, latent, pooled_latent, freq_before_pool = self._encode_impl(x)

        decode_input = pooled_latent
        if self.latent_pool is not None:
            decode_input = F.interpolate(
                decode_input,
                size=(freq_before_pool, decode_input.shape[-1]),
                mode="nearest",
            )
        decode_input = self.channel_expand(decode_input)

        x = self.dec_deconv1(decode_input)
        x = self.dec_norm1(x)
        x = F.gelu(x)
        x = self.dec_tcn1(x)
        x = self.dec_deconv0(x)
        x = self.dec_norm0(x)
        x = F.gelu(x)
        x = self.dec_tcn0(x)
        x = self.output_proj(x)
        x = self.output_activation(x)

        if x.shape[-2:] != input_shape:
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        if not self.training and not getattr(self, "_printed_compression", False):
            inp = latent_pre.shape
            lat = pooled_latent.shape
            inp_elems = int(inp[1] * inp[2] * inp[3])
            lat_elems = int(lat[1] * lat[2] * lat[3])
            if lat_elems > 0:
                print(
                    f"[AE] input {inp[1:]} elems={inp_elems:,} -> latent {lat[1:]} elems={lat_elems:,} "
                    f"(×{inp_elems / lat_elems:.2f} compression)"
                )
            self._printed_compression = True
        return x, pooled_latent
