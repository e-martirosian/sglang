# SPDX-License-Identifier: Apache-2.0
"""Native ``AutoencoderKLConv3D`` VAE for HunyuanImage-3.0.

Ported from the official ``autoencoder_kl_3d.py`` shipped with
``tencent/HunyuanImage-3.0-Instruct`` (reference: FLUX autoencoder + MIT-Han-Lab
DCAE), with the diffusers ``ModelMixin``/``ConfigMixin`` machinery, the
multi-process distributed decode variant and gradient checkpointing removed so
it can be instantiated on a meta device and loaded directly from the
checkpoint's sharded safetensors (weights live under the ``vae.*`` prefix).

Module naming is kept identical to the official implementation so the
safetensors tensors map 1:1:

* ``encoder.conv_in / encoder.down.N.block.M.* / encoder.down.N.downsample.*``
* ``encoder.mid.{block_1, attn_1, block_2} / encoder.norm_out / encoder.conv_out``
* ``decoder.*`` (mirrored layout)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        if parameters.ndim == 3:
            dim = 2  # (B, L, C)
        elif parameters.ndim in (4, 5):
            dim = 1  # (B, C, T, H, W) / (B, C, H, W)
        else:
            raise NotImplementedError
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> Tensor:
        sample = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * sample

    def mode(self) -> Tensor:
        return self.mean


@dataclass
class DecoderOutput:
    sample: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution] = None


class Conv3d(nn.Conv3d):
    """Conv3d with temporal-chunked fallback for large inputs.

    Numerically matches ``nn.Conv3d`` within 1e-5; only symmetric temporal
    padding is supported (same as the official implementation).
    """

    def forward(self, input):
        B, C, T, H, W = input.shape
        memory_count = (C * T * H * W) * 2 / 1024**3
        if memory_count > 2:
            n_split = math.ceil(memory_count / 2)
            assert n_split >= 2
            chunks = torch.chunk(input, chunks=n_split, dim=-3)
            padded_chunks = []
            for i in range(len(chunks)):
                if self.padding[0] > 0:
                    padded_chunk = F.pad(
                        chunks[i],
                        (0, 0, 0, 0, self.padding[0], self.padding[0]),
                        mode=(
                            "constant"
                            if self.padding_mode == "zeros"
                            else self.padding_mode
                        ),
                        value=0,
                    )
                    if i > 0:
                        padded_chunk[:, :, : self.padding[0]] = chunks[i - 1][
                            :, :, -self.padding[0] :
                        ]
                    if i < len(chunks) - 1:
                        padded_chunk[:, :, -self.padding[0] :] = chunks[i + 1][
                            :, :, : self.padding[0]
                        ]
                else:
                    padded_chunk = chunks[i]
                padded_chunks.append(padded_chunk)
            padding_bak = self.padding
            self.padding = (0, self.padding[1], self.padding[2])
            outputs = []
            for padded_chunk in padded_chunks:
                outputs.append(super().forward(padded_chunk))
            self.padding = padding_bak
            return torch.cat(outputs, dim=-3)
        return super().forward(input)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.q = Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        q = q.permute(0, 2, 3, 4, 1).reshape(b, 1, f * h * w, c).contiguous()
        k = k.permute(0, 2, 3, 4, 1).reshape(b, 1, f * h * w, c).contiguous()
        v = v.permute(0, 2, 3, 4, 1).reshape(b, 1, f * h * w, c).contiguous()
        h_ = F.scaled_dot_product_attention(q, k, v)
        return (
            h_.reshape(b, f, h, w, c)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        self.add_temporal_downsample = add_temporal_downsample
        stride = (2, 2, 2) if add_temporal_downsample else (1, 2, 2)  # THW
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = Conv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=0)

    def forward(self, x: Tensor):
        spatial_pad = (0, 1, 0, 1, 0, 0)  # WHT
        x = F.pad(x, spatial_pad, mode="constant", value=0)

        temporal_pad = (0, 0, 0, 0, 0, 1) if self.add_temporal_downsample else (0, 0, 0, 0, 1, 1)
        x = F.pad(x, temporal_pad, mode="replicate")

        x = self.conv(x)
        return x


class DownsampleDCAE(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True
    ):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = Conv3d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)

        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_downsample else 1
        b, c, tf, th, tw = x.shape
        f, h, w = tf // r1, th // 2, tw // 2
        h_conv = self.conv(x)
        cout = h_conv.shape[1]
        # einops: 'b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w'
        h_conv = h_conv.reshape(b, cout, f, r1, h, 2, w, 2)
        h_conv = h_conv.permute(0, 3, 5, 7, 1, 2, 4, 6).reshape(
            b, r1 * 2 * 2 * cout, f, h, w
        )
        shortcut = x.reshape(b, c, f, r1, h, 2, w, 2)
        shortcut = shortcut.permute(0, 3, 5, 7, 1, 2, 4, 6).reshape(
            b, r1 * 2 * 2 * c, f, h, w
        )

        B, C, T, H, W = shortcut.shape
        shortcut = shortcut.view(B, h_conv.shape[1], self.group_size, T, H, W).mean(dim=2)
        return h_conv + shortcut


class Upsample(nn.Module):
    def __init__(self, in_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        self.add_temporal_upsample = add_temporal_upsample
        self.scale_factor = (2, 2, 2) if add_temporal_upsample else (1, 2, 2)  # THW
        self.conv = Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class UpsampleDCAE(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True
    ):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = Conv3d(in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1)

        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: Tensor) -> Tensor:
        r1 = 2 if self.add_temporal_upsample else 1
        b, cin, f, h, w = x.shape
        h_conv = self.conv(x)
        out_c = h_conv.shape[1] // (r1 * 2 * 2)
        # einops: 'b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)'
        h_conv = h_conv.reshape(b, r1, 2, 2, out_c, f, h, w)
        h_conv = h_conv.permute(0, 4, 5, 1, 6, 2, 7, 3).reshape(
            b, out_c, f * r1, h * 2, w * 2
        )
        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
        shortcut = shortcut.reshape(b, r1, 2, 2, out_c, f, h, w)
        shortcut = shortcut.permute(0, 4, 5, 1, 6, 2, 7, 3).reshape(
            b, out_c, f * r1, h * 2, w * 2
        )
        return h_conv + shortcut


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = tuple(block_out_channels)
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = Conv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < math.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(
                i_level >= math.log2(ffactor_spatial // ffactor_temporal)
            )
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = (
                    block_out_channels[i_level + 1]
                    if downsample_match_channel
                    else block_in
                )
                down.downsample = DownsampleDCAE(block_in, block_out, add_temporal_downsample)
                block_in = block_out
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = Conv3d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        h = self.conv_in(x)
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        b, _, f, hh, ww = h.shape
        shortcut = (
            h.reshape(b, -1, group_size, f, hh, ww).mean(dim=2)
        )
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h += shortcut
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = tuple(block_out_channels)
        self.num_res_blocks = num_res_blocks

        # z to block_in
        block_in = block_out_channels[0]
        self.conv_in = Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block

            add_spatial_upsample = bool(i_level < math.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < math.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = (
                    block_out_channels[i_level + 1]
                    if upsample_match_channel
                    else block_in
                )
                up.upsample = UpsampleDCAE(block_in, block_out, add_temporal_upsample)
                block_in = block_out
            self.up.append(up)

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = Conv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # z to block_in
        repeats = self.block_out_channels[0] // self.z_channels
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)
        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


@dataclass
class AutoencoderKLConv3DConfig:
    """Minimal stand-in for the diffusers config used by the decode stage."""

    _class_name: str = "AutoencoderKLConv3D"
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 32
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024, 1024)
    layers_per_block: int = 2
    ffactor_spatial: int = 16
    ffactor_temporal: int = 4
    sample_size: int = 384
    sample_tsize: int = 96
    downsample_match_channel: bool = True
    upsample_match_channel: bool = True
    scaling_factor: Optional[float] = None
    shift_factor: Optional[float] = None


class AutoencoderKLConv3D(nn.Module):
    """Native HunyuanImage-3.0 VAE (DCAE-style Conv3D autoencoder).

    Weight names match the official implementation exactly, so the
    ``vae.*`` tensors from the checkpoint's safetensors load 1:1 after
    stripping the ``vae.`` prefix.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 32,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024, 1024),
        layers_per_block: int = 2,
        ffactor_spatial: int = 16,
        ffactor_temporal: int = 4,
        sample_size: int = 384,
        sample_tsize: int = 96,
        scaling_factor: Optional[float] = None,
        shift_factor: Optional[float] = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
        only_encoder: bool = False,
        only_decoder: bool = False,
    ):
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        self.config = AutoencoderKLConv3DConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            sample_size=sample_size,
            sample_tsize=sample_tsize,
            downsample_match_channel=downsample_match_channel,
            upsample_match_channel=upsample_match_channel,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
        )

        if not only_decoder:
            self.encoder = Encoder(
                in_channels=in_channels,
                z_channels=latent_channels,
                block_out_channels=block_out_channels,
                num_res_blocks=layers_per_block,
                ffactor_spatial=ffactor_spatial,
                ffactor_temporal=ffactor_temporal,
                downsample_match_channel=downsample_match_channel,
            )
        if not only_encoder:
            self.decoder = Decoder(
                z_channels=latent_channels,
                out_channels=out_channels,
                block_out_channels=list(reversed(block_out_channels)),
                num_res_blocks=layers_per_block,
                ffactor_spatial=ffactor_spatial,
                ffactor_temporal=ffactor_temporal,
                upsample_match_channel=upsample_match_channel,
            )

        self.use_slicing = False
        self.slicing_bsz = 1
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // ffactor_temporal
        self.tile_overlap_factor = 0.125

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "AutoencoderKLConv3D":
        """Build from the ``"vae"`` sub-dict of the checkpoint config.json."""
        kwargs = {
            k: cfg[k]
            for k in (
                "in_channels",
                "out_channels",
                "latent_channels",
                "block_out_channels",
                "layers_per_block",
                "ffactor_spatial",
                "ffactor_temporal",
                "sample_size",
                "sample_tsize",
                "scaling_factor",
                "shift_factor",
                "downsample_match_channel",
                "upsample_match_channel",
            )
            if k in cfg
        }
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Tiling toggles
    # ------------------------------------------------------------------
    def enable_temporal_tiling(self, use_tiling: bool = True):
        self.use_temporal_tiling = use_tiling

    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        self.enable_spatial_tiling(use_tiling)

    def disable_tiling(self):
        self.disable_spatial_tiling()

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    # ------------------------------------------------------------------
    # Tile blending
    # ------------------------------------------------------------------
    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = (
                a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent)
                + b[:, :, :, :, x] * (x / blend_extent)
            )
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = (
                a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent)
                + b[:, :, :, y, :] * (y / blend_extent)
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = (
                a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent)
                + b[:, :, x, :, :] * (x / blend_extent)
            )
        return b

    # ------------------------------------------------------------------
    # Tiled encode / decode (single-device serial paths)
    # ------------------------------------------------------------------
    def spatial_tiled_encode(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = x[
                    :, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size
                ]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def temporal_tiled_encode(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_latent_min_tsize - blend_extent

        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i : i + self.tile_sample_min_tsize, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_sample_min_size
                or tile.shape[-2] > self.tile_sample_min_size
            ):
                tile = self.spatial_tiled_encode(tile)
            else:
                tile = self.encoder(tile)
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        return torch.cat(result_row, dim=-3)

    def spatial_tiled_decode(self, z: torch.Tensor):
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = z[
                    :, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size
                ]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def temporal_tiled_decode(self, z: torch.Tensor):
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - blend_extent
        assert 0 < overlap_size < self.tile_latent_min_tsize

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i : i + self.tile_latent_min_tsize, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_latent_min_size
                or tile.shape[-2] > self.tile_latent_min_size
            ):
                decoded = self.spatial_tiled_decode(tile)
            else:
                decoded = self.decoder(tile)
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        return torch.cat(result_row, dim=-3)

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------
    def encode(self, x: Tensor, return_dict: bool = True):
        def _encode(x):
            if self.use_temporal_tiling and x.shape[-3] > self.tile_sample_min_tsize:
                return self.temporal_tiled_encode(x)
            if self.use_spatial_tiling and (
                x.shape[-1] > self.tile_sample_min_size
                or x.shape[-2] > self.tile_sample_min_size
            ):
                return self.spatial_tiled_encode(x)
            return self.encoder(x)

        if len(x.shape) != 5:  # (B, C, T, H, W)
            x = x[:, :, None]
        assert len(x.shape) == 5  # (B, C, T, H, W)
        if x.shape[2] == 1:
            x = x.expand(-1, -1, self.ffactor_temporal, -1, -1)
        else:
            assert x.shape[2] != self.ffactor_temporal and x.shape[2] % self.ffactor_temporal == 0

        if self.use_slicing and x.shape[0] > 1:
            if self.slicing_bsz == 1:
                encoded_slices = [_encode(x_slice) for x_slice in x.split(1)]
            else:
                sections = [self.slicing_bsz] * (x.shape[0] // self.slicing_bsz)
                if x.shape[0] % self.slicing_bsz != 0:
                    sections.append(x.shape[0] % self.slicing_bsz)
                encoded_slices = [_encode(x_slice) for x_slice in x.split(sections)]
            h = torch.cat(encoded_slices)
        else:
            h = _encode(x)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return DecoderOutput(sample=h, posterior=posterior)

    def decode(self, z: Tensor, return_dict: bool = True, generator=None):
        def _decode(z):
            if self.use_temporal_tiling and z.shape[-3] > self.tile_latent_min_tsize:
                return self.temporal_tiled_decode(z)
            if self.use_spatial_tiling and (
                z.shape[-1] > self.tile_latent_min_size
                or z.shape[-2] > self.tile_latent_min_size
            ):
                return self.spatial_tiled_decode(z)
            return self.decoder(z)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [_decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = _decode(z)

        if z.shape[-3] == 1:
            decoded = decoded[:, :, -1:]
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
    ):
        posterior = self.encode(sample, return_dict=False)[0]
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z, return_dict=False)[0]
        return DecoderOutput(sample=dec, posterior=posterior) if return_dict else (dec, posterior)


EntryClass = [AutoencoderKLConv3D]
