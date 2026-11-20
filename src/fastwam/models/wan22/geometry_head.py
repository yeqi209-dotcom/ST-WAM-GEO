"""Training-only dense geometry supervision head.

The head consumes the hidden DINO video tokens produced by the existing MoT
and predicts a dense future depth sequence.  It is deliberately separate from
the diffusion experts: no geometry tokens are added to MoT and the head is not
called by ``infer_action``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryAuxiliaryHead(nn.Module):
    """Lightweight, single-scale spatial decoder for depth supervision.

    Args:
        hidden_dim: DINO Video DiT hidden dimension.
        decoder_dim: Feature width used by the dense decoder.
        num_layers: Number of per-frame 2D convolution blocks.
        output_channels: Number of geometry channels, normally one depth map.
    """

    def __init__(
        self,
        hidden_dim: int,
        decoder_dim: int = 256,
        num_layers: int = 2,
        output_channels: int = 1,
        output_activation: str = "identity",
    ):
        super().__init__()
        if hidden_dim <= 0 or decoder_dim <= 0:
            raise ValueError("`hidden_dim` and `decoder_dim` must be positive.")
        if num_layers <= 0:
            raise ValueError("`num_layers` must be positive.")
        if output_channels != 1:
            raise ValueError("This depth-supervision head requires `output_channels=1`.")

        self.hidden_dim = int(hidden_dim)
        self.decoder_dim = int(decoder_dim)
        self.num_layers = int(num_layers)
        self.output_channels = int(output_channels)
        self.output_activation = str(output_activation).strip().lower()
        if self.output_activation not in {"identity", "sigmoid", "softplus"}:
            raise ValueError(
                "`output_activation` must be one of identity, sigmoid, softplus; "
                f"got {output_activation!r}"
            )

        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.input_projection = nn.Linear(self.hidden_dim, self.decoder_dim)

        blocks = []
        for _ in range(self.num_layers):
            blocks.extend(
                [
                    nn.Conv2d(self.decoder_dim, self.decoder_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=math.gcd(32, self.decoder_dim), num_channels=self.decoder_dim),
                    nn.GELU(),
                ]
            )
        self.decoder = nn.Sequential(*blocks)
        self.output_projection = nn.Conv2d(self.decoder_dim, self.output_channels, kernel_size=1)

    def forward(
        self,
        tokens: torch.Tensor,
        grid_size: Sequence[int],
        output_size: Sequence[int] | None = None,
        num_views: int = 1,
    ) -> torch.Tensor:
        """Decode tokens into ``[B, C, T, H, W]`` geometry maps."""

        if tokens.ndim != 3:
            raise ValueError(f"`tokens` must be [B,S,D], got {tuple(tokens.shape)}")
        if len(grid_size) != 3:
            raise ValueError(f"`grid_size` must contain (T,H,W), got {grid_size}")
        grid_t, grid_h, grid_w = (int(v) for v in grid_size)
        if min(grid_t, grid_h, grid_w) <= 0:
            raise ValueError(f"`grid_size` values must be positive, got {grid_size}")
        num_views = int(num_views)
        if num_views <= 0:
            raise ValueError("`num_views` must be positive.")
        expected_tokens = grid_t * num_views * grid_h * grid_w
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                "Geometry token/grid mismatch: "
                f"tokens={tokens.shape[1]} vs grid={grid_size} ({expected_tokens})."
            )
        if tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Geometry token dimension must be {self.hidden_dim}, got {tokens.shape[-1]}"
            )

        x = self.input_projection(self.input_norm(tokens))
        # View-patched DINO tokens have order (T,view,H,W). Restore the
        # horizontal mosaic before spatial decoding; never interpolate time.
        x = x.reshape(tokens.shape[0], grid_t, num_views, grid_h, grid_w, self.decoder_dim)
        x = x.permute(0, 1, 5, 3, 2, 4).reshape(
            tokens.shape[0] * grid_t, self.decoder_dim, grid_h, num_views * grid_w
        )
        x = self.decoder(x)
        x = self.output_projection(x)
        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.output_activation == "softplus":
            x = F.softplus(x)
        if output_size is not None:
            output_size = tuple(int(v) for v in output_size)
            if len(output_size) != 3 or min(output_size) <= 0:
                raise ValueError(f"`output_size` must be a positive (T,H,W), got {output_size}")
            if output_size[0] != grid_t:
                raise ValueError("Geometry decoding requires one token frame per target frame (temporal patch size 1).")
            if tuple(x.shape[-2:]) != output_size[1:]:
                x = F.interpolate(x, size=output_size[1:], mode="bilinear", align_corners=False)
        return x.reshape(tokens.shape[0], grid_t, self.output_channels, *x.shape[-2:]).permute(0, 2, 1, 3, 4).contiguous()

