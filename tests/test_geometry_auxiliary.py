"""CPU checks without downloading the Wan/DINO/Qwen backbones.

The actual model/trainer class bodies are compiled from source in isolation,
so optional simulator and CUDA dependencies are unnecessary. Tiny experts
replace the large backbones; loss, input, mask and checkpoint logic is real.
"""
import ast
import importlib.util
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Optional, Sequence, Union
import zipfile

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "src/fastwam/models/wan22/fastwam_vae_dino_mot.py"


def load_file(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GeometryAuxiliaryHead = load_file(MODEL_PATH.with_name("geometry_head.py")).GeometryAuxiliaryHead
load_geometry_window = load_file(ROOT / "src/fastwam/datasets/geometry_targets.py").load_geometry_window
Scheduler = load_file(MODEL_PATH.parent / "schedulers/scheduler_continuous.py").WanContinuousFlowMatchScheduler


def isolated_class(path, name):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    scope = dict(globals())
    scope.update({key: nn.Module for key in (
        "ActionDiT", "DinoVideoDiT", "DinoVideoEncoder", "DINOHistoryIntentResampler",
        "QwenDINOHistoryActionAdapter", "MoT", "GeometryVideoDiT",
    )})
    scope.update(logger=logging.getLogger(name), WanContinuousFlowMatchScheduler=Scheduler,
                 DINO_FUTURE_MODES={"predict", "condition_only"},
                 INTENT_INJECTION_MODES={"context_after_proprio"})
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


Model = isolated_class(MODEL_PATH, "FastWAMVAEDinoMoT")
Trainer = isolated_class(ROOT / "src/fastwam/trainer.py", "Wan22Trainer")


class TinyVideo(nn.Module):
    hidden_dim = 8
    latent_patch_size = (1, 1, 1)
    dino_dim = 2
    fuse_vae_embedding_in_latents = False

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, 2)

    def pre_dit(self, x, context, context_mask, **kwargs):
        b, c, t, h, w = x.shape
        return dict(tokens=self.encoder(x.permute(0, 2, 3, 4, 1).reshape(b, -1, c)),
                    context=context, context_mask=context_mask, freqs=None, t_mod=None,
                    meta=dict(grid_size=(t, h, w), tokens_per_frame=h*w, latent_patch_mode="flat"))

    def post_dit(self, tokens, state):
        t, h, w = state["meta"]["grid_size"]
        return self.decoder(tokens).reshape(tokens.shape[0], t, h, w, 2).permute(0, 4, 1, 2, 3)

    def target_to_output_space(self, x):
        return x

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        mask = torch.ones(video_seq_len, video_seq_len, dtype=torch.bool, device=device)
        mask[:video_tokens_per_frame, video_tokens_per_frame:] = False
        return mask


class TinyAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = nn.Linear(2, 8), nn.Linear(8, 2)

    def pre_dit(self, action_tokens, context, context_mask, **kwargs):
        return dict(tokens=self.encoder(action_tokens), context=context, context_mask=context_mask,
                    freqs=None, t_mod=None)

    def post_dit(self, tokens, state):
        return self.decoder(tokens)


class TinyMoT(nn.Module):
    def __init__(self, video, dino, action):
        super().__init__()
        self.mixtures = nn.ModuleDict(dict(video=video, dino=dino, action=action))
        self.mix = nn.Linear(8, 8)

    def forward(self, embeds_all, **kwargs):
        return {key: self.mix(value) for key, value in embeds_all.items()}


class TinyVAE(nn.Module):
    temporal_downsample_factor = 4
    z_dim = 2


def make_model(enabled=True, model_cls=Model):
    video, dino, action = TinyVideo(), TinyVideo(), TinyAction()
    kwargs = dict(video_expert=video, dino_video_expert=dino, action_expert=action,
                  mot=TinyMoT(video, dino, action), vae=TinyVAE(), dino_encoder=None,
                  text_dim=8, device="cpu")
    if model_cls is Model:
        kwargs.update(geometry_config=dict(enabled=enabled, decoder_dim=10), loss_lambda_geometry=.05)
    return model_cls(**kwargs)


def sample():
    return dict(vae_latents=torch.randn(2, 2, 3, 2, 4),
                dino_latents=torch.randn(2, 2, 9, 2, 4), action=torch.randn(2, 32, 2),
                context=torch.randn(2, 3, 8), context_mask=torch.ones(2, 3, dtype=torch.bool),
                geometry=torch.rand(2, 1, 9, 4, 8),
                geometry_valid=torch.ones(2, 1, 9, 4, 8, dtype=torch.bool),
                image_is_pad=torch.zeros(2, 9, dtype=torch.bool),
                action_is_pad=torch.zeros(2, 32, dtype=torch.bool))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dense_head_gradient_and_no_temporal_mixing(dtype):
    head = GeometryAuxiliaryHead(8, decoder_dim=10).to(dtype=dtype)
    tokens = torch.randn(2, 3*2*4, 8, dtype=dtype, requires_grad=True)
    pred = head(tokens, (3, 2, 4), (3, 4, 8))
    assert pred.shape == (2, 1, 3, 4, 8)
    loss = Model._compute_geometry_loss_per_sample(pred, torch.ones_like(pred), None, None).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert tokens.grad[:, :8].count_nonzero() == 0  # f0 has no supervision
    assert tokens.grad[:, 8:].abs().sum() > 0
    assert head.output_projection.weight.grad.abs().sum() > 0


def test_view_tokens_restore_spatial_order():
    head = GeometryAuxiliaryHead(1, decoder_dim=1, num_layers=1)
    head.input_norm = head.input_projection = head.decoder = head.output_projection = nn.Identity()
    tokens = torch.arange(8.).reshape(1, 8, 1)  # T=1, V=2, H=2, W=2
    out = head(tokens, (1, 2, 2), num_views=2)
    assert out[0, 0, 0].tolist() == [[0., 1., 4., 5.], [2., 3., 6., 7.]]
    with pytest.raises(ValueError, match="one token frame"):
        head(tokens, (1, 2, 2), (2, 2, 4), num_views=2)


def test_future_loss_ignores_current_padding_invalid_and_nan():
    pred = torch.tensor([100., 2., 100., 100.]).reshape(1, 1, 4, 1, 1).requires_grad_()
    target = torch.tensor([0., 1., float("nan"), 0.]).reshape_as(pred)
    pad = torch.tensor([[False, False, False, True]])
    loss = Model._compute_geometry_loss_per_sample(pred, target, None, pad)
    assert loss.item() == 1.
    loss.sum().backward()
    assert pred.grad.flatten().tolist() == [0., 1., 0., 0.]
    invalid_loss = Model._compute_geometry_loss_per_sample(pred, target, torch.zeros_like(pred).bool(), pad)
    assert invalid_loss.item() == 0.


def cache_kwargs():
    return dict(sample_index=5, sample_offsets=[0, 4, 8], full_frame_count=9,
                output_size=[2, 4], max_depth=10., expected_frame_indices=[5, 9, 13])


def test_cache_full_and_selected_frames_and_finite_resize(tmp_path):
    full = torch.arange(1., 10.)[:, None, None].expand(-1, 4, 8).clone()
    full[4, 0, 0] = float("nan")
    full[8, 2, 4] = float("inf")
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save(dict(geometry=full, sample_index=5, frame_indices=list(range(5, 14))), a)
    torch.save(dict(depth=full[[0, 4, 8]], frame_indices=[5, 9, 13]), b)
    depth, valid = load_geometry_window(str(a), **cache_kwargs())
    depth_b, valid_b = load_geometry_window(str(b), **cache_kwargs())
    assert depth.shape == (1, 3, 2, 4) and depth.dtype == torch.float32
    assert torch.equal(depth, depth_b) and torch.equal(valid, valid_b)
    assert torch.isfinite(depth).all()
    assert not valid[0, 1, 0, 0] and valid[0, 1, 0, 1]
    assert depth[0, 1, 0, 1].item() == .5
    assert not valid[0, 2, 1, 2]


@pytest.mark.parametrize("payload,match", [
    ({"depth": torch.ones(4, 2, 4)}, "temporal length"),
    ({"depth": torch.ones(3, 2, 2)}, "aspect ratio"),
    ({"depth": torch.ones(3, 2, 4), "frame_indices": [5, 10, 13]}, "frame_indices mismatch"),
    ({"depth": torch.ones(3, 2, 4), "sample_index": 6}, "sample_index mismatch"),
    ({"depth": torch.ones(3, 2, 4), "valid": torch.ones(3, 1, 1)}, "mask shape"),
])
def test_bad_cache_fails_explicitly(tmp_path, payload, match):
    path = tmp_path / "cache.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match=match):
        load_geometry_window(str(path), **cache_kwargs())


def test_numpy_cache_and_metric_scaling(tmp_path):
    x = np.ones((3, 2, 4), np.float32) * 20
    x[1, 0, 0], x[2, 0, 0] = 0, -1
    path = tmp_path / "cache.npy"
    np.save(path, x)
    depth, valid = load_geometry_window(str(path), **cache_kwargs())
    assert depth.max() == 1 and not valid[0, 1, 0, 0] and not valid[0, 2, 0, 0]


def test_real_training_loss_backprop_and_original_loss_accounting():
    model, batch = make_model(), sample()
    Trainer._apply_dit_only_train_mode(model)
    loss, info = model.training_loss(batch)
    assert loss.item() == pytest.approx(sum(info[key] for key in (
        "loss_video", "loss_dino", "loss_action", "loss_geometry")), rel=1e-5)
    assert info["loss_geometry"] == pytest.approx(.05*info["loss_geometry_raw"])
    loss.backward()
    assert model.geometry_head.output_projection.weight.grad.abs().sum() > 0
    # Isolate the geometry objective to prove gradients reach the DINO expert.
    model.zero_grad()
    model.loss_lambda_video = model.loss_lambda_dino = model.loss_lambda_action = 0.
    model.training_loss(batch)[0].backward()
    assert model.dino_video_expert.encoder.weight.grad.abs().sum() > 0
    assert model.mot.mix.weight.grad.abs().sum() > 0
    assert model.geometry_head.training and all(p.requires_grad for p in model.geometry_head.parameters())


def test_build_inputs_masks_nonfinite_even_when_user_marks_valid():
    model, batch = make_model(), sample()
    batch["geometry"][0, 0, 2, 1, 1] = float("inf")
    built = model.build_inputs(batch)
    assert not built["geometry_valid"][0, 0, 2, 1, 1]
    assert torch.isfinite(built["geometry"]).all()
    batch.pop("geometry_valid")
    assert not model.build_inputs(batch)["geometry_valid"][0, 0, 2, 1, 1]
    batch["geometry_valid"] = torch.ones(2, 1, 9, 1, 1)
    with pytest.raises(ValueError, match="shape must match"):
        model.build_inputs(batch)


def test_eval_batch_preserves_labels():
    batch = sample()
    single = {key: value[0] for key, value in batch.items()}
    single["video_frame_count"] = 9
    result = Trainer._to_batched_eval_sample(single)
    assert result["geometry"].shape == (1, 1, 9, 4, 8)
    assert result["geometry_valid"].dtype == torch.bool
    assert make_model().training_loss(result)[1]["loss_geometry_raw"] >= 0


def test_geometry_checkpoint_and_baseline_initialization(tmp_path):
    model = make_model()
    path = tmp_path / "geometry.pt"
    model.save_checkpoint(path, step=7, safe_serialization=True)
    restored = make_model()
    assert restored.load_checkpoint(path) == 7
    for a, b in zip(model.geometry_head.parameters(), restored.geometry_head.parameters()):
        assert torch.equal(a, b)
    inference = make_model(enabled=False)
    inference.load_checkpoint(path)
    baseline_path = tmp_path / "baseline.pt"
    inference.save_checkpoint(baseline_path)
    before = {k: v.clone() for k, v in restored.geometry_head.state_dict().items()}
    restored.load_checkpoint(baseline_path)
    assert all(torch.equal(v, restored.geometry_head.state_dict()[k]) for k, v in before.items())
    incompatible = make_model()
    incompatible.geometry_config["output_activation"] = "sigmoid"
    with pytest.raises(ValueError, match="output_activation mismatch"):
        incompatible.load_checkpoint(path)


@pytest.mark.parametrize("task,size,resolution", [
    ("libero_geometry_aux", [56, 112], [224, 448]),
    ("robotwin_geometry_aux", [96, 80], [384, 320]),
])
def test_hydra_task_composition(task, size, resolution):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task}"])
    OmegaConf.to_container(cfg, resolve=True)
    assert cfg.model.geometry_config.enabled and cfg.data.train.geometry_enabled
    assert cfg.model.loss.lambda_geometry == .05
    assert cfg.model.loss.lambda_video == 1 and cfg.model.loss.lambda_dino == .02
    assert cfg.model.loss.lambda_action == 1
    assert list(cfg.data.train.geometry_size) == size
    assert list(cfg.model.dino_config.input_resolution) == resolution
    assert cfg.model.semantic_history_config.enabled  # original CAIR retained


def test_inference_unchanged_and_original_loader_accepts_geometry_checkpoint(tmp_path):
    baseline = ROOT.with_name("ST-WAM-Semantic-Temporal-World-Action-Model")
    if not baseline.exists():
        pytest.skip("Original checkout is optional for standalone use")
    relative = MODEL_PATH.relative_to(ROOT)
    old_tree, new_tree = ast.parse((baseline / relative).read_text()), ast.parse(MODEL_PATH.read_text())
    old = next(node for node in old_tree.body if isinstance(node, ast.ClassDef))
    new = next(node for node in new_tree.body if isinstance(node, ast.ClassDef))
    for name in ("infer_action", "_predict_action_noise", "_prefill_action_static_cache",
                 "_predict_action_noise_with_static_cache", "_build_tribranch_attention_mask"):
        a = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(a) == ast.dump(b), name
    path = tmp_path / "new_weights.pt"
    model = make_model()
    model.save_checkpoint(path)
    old_model = make_model(model_cls=isolated_class(baseline / relative, "FastWAMVAEDinoMoT"))
    old_model.load_checkpoint(path)
    assert all(torch.equal(a, b) for a, b in zip(model.mot.parameters(), old_model.mot.parameters()))


def test_disabled_geometry_retains_baseline_training_loss():
    baseline = ROOT.with_name("ST-WAM-Semantic-Temporal-World-Action-Model") / MODEL_PATH.relative_to(ROOT)
    if not baseline.exists():
        pytest.skip("Original checkout is optional for standalone use")
    old = make_model(model_cls=isolated_class(baseline, "FastWAMVAEDinoMoT"))
    new = make_model(enabled=False)
    new.mot.load_state_dict(old.mot.state_dict())
    batch = sample()
    batch.pop("geometry")
    batch.pop("geometry_valid")
    torch.manual_seed(91)
    old_loss, old_info = old.training_loss(batch)
    torch.manual_seed(91)
    new_loss, new_info = new.training_loss(batch)
    assert torch.equal(old_loss, new_loss)
    assert all(old_info[key] == new_info[key] for key in old_info)


@pytest.mark.parametrize("task", ["libero_geometry_aux", "libero_geometry_mot"])
def test_cache_packing_and_check_cli_with_episode_clamped_frames(tmp_path, monkeypatch, task):
    # The real resolver module is lightweight; importing the parent utils
    # package would also import the optional video I/O stack.
    monkeypatch.setitem(sys.modules, "fastwam.utils.config_resolvers", load_file(
        ROOT / "src/fastwam/utils/config_resolvers.py"))
    module = load_file(ROOT / "scripts/prepare_geometry_cache.py")

    class MetadataOnlyDataset:
        def __init__(self, cfg):
            self.num_frames = cfg.num_frames
            self.video_sample_indices = list(range(0, cfg.num_frames, cfg.action_video_freq_ratio))
            self.geometry_size = cfg.geometry_size
            self.geometry_normalize = cfg.geometry_normalize
            self.geometry_max_depth = cfg.geometry_max_depth
            self.geometry_min_depth = cfg.geometry_min_depth

        def __len__(self):
            return 2

        def _get_video_global_indices(self, idx):
            return [min(1, idx + offset) for offset in self.video_sample_indices]

    bank, output = tmp_path / "frames.npy", tmp_path / "windows"
    np.save(bank, np.arange(1., 3., dtype=np.float32)[:, None, None] * np.ones((2, 4, 8), np.float32))
    monkeypatch.setattr(module, "instantiate", MetadataOnlyDataset)
    argv = ["prepare_geometry_cache.py", "--task", task, "--output", str(output)]
    monkeypatch.setattr(sys, "argv", [*argv, "--frame-depths", str(bank)])
    module.main()
    payload = torch.load(output / "00000000.pt", weights_only=True)
    assert payload["frame_indices"] == [0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert payload["geometry"][:, 0, 0].tolist() == [1., 2., 2., 2., 2., 2., 2., 2., 2.]
    monkeypatch.setattr(sys, "argv", [*argv, "--check-only"])
    module.main()


def test_geometry_cache_errors_do_not_trigger_dataset_random_retry():
    dataset_class = isolated_class(ROOT / "src/fastwam/datasets/lerobot/robot_video_dataset.py", "RobotVideoDataset")
    dataset = dataset_class.__new__(dataset_class)
    dataset.geometry_enabled = True

    def missing(idx):
        raise FileNotFoundError("geometry label missing")

    dataset._get = missing
    with pytest.raises(FileNotFoundError, match="geometry label missing"):
        dataset[4]
