"""Depth sidecar parsing, frame selection and masked spatial resizing."""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _as_depth_sequence(value, name: str) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim == 3:
        value = value.unsqueeze(0)
    elif value.ndim == 4 and value.shape[0] != 1 and value.shape[1] == 1:
        value = value.permute(1, 0, 2, 3)
    if value.ndim != 4 or value.shape[0] != 1 or min(value.shape) <= 0:
        raise ValueError(f"{name} must be [T,H,W], [1,T,H,W] or [T,1,H,W]; got {tuple(value.shape)}")
    return value


def load_geometry_window(
    path: str,
    *,
    sample_index: int,
    sample_offsets: Sequence[int],
    full_frame_count: int,
    output_size: Sequence[int],
    normalize: str = "unit",
    max_depth: float = 10.0,
    min_depth: Optional[float] = 0.0,
    expected_frame_indices: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return float32 depth and bool validity, both [1,T,H,W].

    Input maps must already use the RGB camera mosaic and orientation. Labels
    may contain the full action window or only the selected video frames.
    Optional frame_indices metadata is checked against the dataset's actual,
    episode-clamped global frame indices.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing geometry cache for sample {sample_index}: {path}")
    payload = (np.load(path, allow_pickle=False) if path.suffix == ".npy"
               else torch.load(path, map_location="cpu", weights_only=True))
    valid, frame_indices = None, None
    if isinstance(payload, dict):
        if "sample_index" in payload and int(payload["sample_index"]) != sample_index:
            raise ValueError(f"Geometry sample_index mismatch in {path}")
        depth = payload.get("geometry", payload.get("depth", payload.get("data")))
        valid = payload.get("geometry_valid", payload.get("valid", payload.get("depth_valid")))
        frame_indices = payload.get("frame_indices")
    else:
        depth = payload
    if depth is None:
        raise ValueError(f"No geometry/depth data in {path}")
    depth = _as_depth_sequence(depth, "geometry").float()
    if valid is not None:
        valid = _as_depth_sequence(valid, "geometry_valid")
        if valid.shape != depth.shape:
            raise ValueError(f"Geometry mask shape {tuple(valid.shape)} != depth {tuple(depth.shape)} in {path}")
        valid = torch.isfinite(valid) & (valid > 0)
    else:
        valid = torch.ones_like(depth, dtype=torch.bool)

    selected_count = len(sample_offsets)
    original_count = depth.shape[1]
    if original_count != selected_count:
        if original_count != full_frame_count:
            raise ValueError(f"Geometry temporal length {original_count} must be {selected_count} selected or {full_frame_count} full frames in {path}")
        depth = depth[:, list(sample_offsets)]
        valid = valid[:, list(sample_offsets)]
    if frame_indices is not None:
        frame_indices = torch.as_tensor(frame_indices)
        if frame_indices.ndim != 1 or frame_indices.numel() != original_count:
            raise ValueError(f"Geometry frame_indices must have {original_count} entries in {path}")
        if original_count != selected_count:
            frame_indices = frame_indices[list(sample_offsets)]
        if expected_frame_indices is not None and frame_indices.tolist() != list(expected_frame_indices):
            raise ValueError(f"Geometry/RGB frame_indices mismatch in {path}")

    if len(output_size) != 2 or min(output_size) <= 0:
        raise ValueError(f"Invalid geometry output_size: {output_size}")
    out_h, out_w = map(int, output_size)
    if depth.shape[-2] * out_w != depth.shape[-1] * out_h:
        raise ValueError(f"Depth mosaic aspect ratio {tuple(depth.shape[-2:])} != RGB target {(out_h, out_w)} in {path}")
    valid = valid & torch.isfinite(depth)
    if min_depth is not None:
        valid = valid & (depth > float(min_depth))
    # Sanitize before resizing. Nearest preserves depth edges and the exact
    # validity of each sampled pixel, including NaN holes and camera seams.
    depth = torch.where(valid, depth, torch.zeros_like(depth))
    if tuple(depth.shape[-2:]) != (out_h, out_w):
        depth = F.interpolate(depth, size=(out_h, out_w), mode="nearest")
        valid = F.interpolate(valid.float(), size=(out_h, out_w), mode="nearest").bool()
    if normalize == "unit":
        if not np.isfinite(max_depth) or max_depth <= 0:
            raise ValueError("geometry_max_depth must be finite and positive")
        depth = depth.clamp(0.0, float(max_depth)) / float(max_depth)
    elif normalize != "none":
        raise ValueError("geometry_normalize must be 'none' or 'unit'")
    return depth.contiguous(), valid.contiguous()
