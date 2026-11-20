"""Pack RGB-aligned, globally indexed depth frames into training windows.

This script packs supplied labels; it does not run a depth teacher or render
simulator depth. Use --check-only to check every existing window before training.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from fastwam.datasets.geometry_targets import load_geometry_window
from fastwam.utils.config_resolvers import register_default_resolvers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["libero_geometry_aux", "robotwin_geometry_aux", "libero_geometry_mot", "robotwin_geometry_mot"])
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--frame-depths", help=".npy [global_frame_count,H,W], unnormalized aligned depth mosaic")
    parser.add_argument("--frame-valid", help="optional .npy validity bank with the same shape")
    parser.add_argument("--output", help="defaults to data.<split>.geometry_cache_dir")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args, overrides = parser.parse_known_args()
    register_default_resolvers()
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = hydra.compose(config_name="train", overrides=[f"task={args.task}", *overrides])
    data_cfg = cfg.data.get(args.split)
    if data_cfg is None:
        parser.error(f"data.{args.split} is not configured in this task")
    data_cfg = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
    output = Path(args.output or data_cfg.geometry_cache_dir)
    if int(data_cfg.get("global_sample_stride", 1)) != 1:
        parser.error("Geometry cache packing requires global_sample_stride=1")

    # Only dataset metadata and the episode index are needed. No RGB, model,
    # text embeddings, dataset statistics or existing latent caches are read.
    for key, value in {
        "processor": None, "load_text_context": False, "geometry_enabled": False,
        "geometry_cache_dir": None, "dino_latent_cache_dir": None,
        "vae_latent_cache_dir": None, "dino_latent_cache_required": False,
        "vae_latent_cache_required": False, "skip_video_load_if_latent_cached": False,
        "load_history_dino_latents": False, "history_dino_latent_cache_required": False,
    }.items():
        OmegaConf.update(data_cfg, key, value, force_add=True)
    dataset = instantiate(data_cfg)
    depths = valid_bank = None
    if not args.check_only:
        if not args.frame_depths:
            parser.error("--frame-depths is required when packing")
        depths = np.load(args.frame_depths, mmap_mode="r", allow_pickle=False)
        if depths.ndim != 3 or depths.shape[0] != len(dataset):
            parser.error(f"Depth bank must be [len(dataset)={len(dataset)},H,W], got {depths.shape}")
        if args.frame_valid:
            valid_bank = np.load(args.frame_valid, mmap_mode="r", allow_pickle=False)
            if valid_bank.shape != depths.shape:
                parser.error("Validity bank must have the same shape as the depth bank")
        output.mkdir(parents=True, exist_ok=True)
    if args.start < 0 or args.start >= len(dataset) or (args.limit is not None and args.limit <= 0):
        parser.error("--start/--limit must describe a nonempty in-range subset")
    end = min(len(dataset), args.start + args.limit) if args.limit is not None else len(dataset)
    valid_windows = 0
    for idx in range(args.start, end):
        indices = dataset._get_video_global_indices(idx)
        path = output / f"{idx:08d}.pt"
        if not args.check_only and (args.overwrite or not path.exists()):
            payload = {
                "sample_index": idx, "frame_indices": indices,
                "geometry": torch.from_numpy(np.array(depths[indices], dtype=np.float32, copy=True)),
            }
            if valid_bank is not None:
                payload["geometry_valid"] = torch.from_numpy(np.array(valid_bank[indices], copy=True))
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                torch.save(payload, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        if args.check_only and not path.exists():
            # Support externally prepared .npy/.pth window caches too.
            path = next((output / f"{idx:08d}{suffix}" for suffix in (".pth", ".npy")
                         if (output / f"{idx:08d}{suffix}").exists()), path)
        _, mask = load_geometry_window(
            str(path), sample_index=idx, sample_offsets=dataset.video_sample_indices,
            full_frame_count=dataset.num_frames, output_size=dataset.geometry_size,
            normalize=dataset.geometry_normalize, max_depth=dataset.geometry_max_depth,
            min_depth=dataset.geometry_min_depth, expected_frame_indices=indices,
        )
        valid_windows += int(mask[:, 1:].any())
        if (idx - args.start + 1) % 1000 == 0:
            print(f"Checked {idx - args.start + 1}/{end - args.start} windows", flush=True)
    report = {"checked": end - args.start, "windows_with_valid_future_depth": valid_windows,
              "cache_dir": str(output.resolve()), "task": args.task, "split": args.split}
    print(json.dumps(report, indent=2))
    if valid_windows == 0:
        raise ValueError("No valid future geometry labels in the checked subset")


if __name__ == "__main__":
    main()
