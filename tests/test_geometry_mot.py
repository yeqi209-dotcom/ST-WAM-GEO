"""Small CPU integration checks using the real DiTs, MoT and training loss.

Only model loading/backbones are omitted. The transformer blocks, masks,
flow scheduler, dense patch codec and checkpoint logic execute as shipped.
"""

import ast
import importlib
import logging
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

import test_geometry_auxiliary as auxiliary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_modules():
    # Import only the lightweight model modules, bypassing optional simulator,
    # transformers, rich and video IO imports in package __init__ files.
    base = ROOT / "src/fastwam/models/wan22"
    package = types.ModuleType("_geometry_mot_real")
    package.__path__ = [str(base)]
    helpers = types.ModuleType("_geometry_mot_real.helpers")
    helpers.__path__ = [str(base / "helpers")]
    logger_name = "fastwam.utils.logging_config"
    previous = sys.modules.get(logger_name)
    logger = types.ModuleType(logger_name)
    logger.get_logger = logging.getLogger
    sys.modules[logger_name] = logger
    sys.modules[package.__name__] = package
    sys.modules[helpers.__name__] = helpers
    modules = types.SimpleNamespace()
    try:
        for name in ("wan_video_dit", "dino_video_dit", "action_dit", "geometry_video_dit", "mot"):
            setattr(modules, name, importlib.import_module(f"{package.__name__}.{name}"))
        yield modules
    finally:
        if previous is None:
            sys.modules.pop(logger_name, None)
        else:
            sys.modules[logger_name] = previous
        for name in list(sys.modules):
            if name.startswith(package.__name__):
                del sys.modules[name]


def make_model(modules, enabled=True, checkpointing=False, dtype=torch.float32):
    common = dict(hidden_dim=12, ffn_dim=24, text_dim=8, freq_dim=8,
                  eps=1e-6, num_heads=1, attn_head_dim=12, num_layers=2,
                  use_gradient_checkpointing=checkpointing)
    video = modules.wan_video_dit.WanVideoDiT(
        **common, in_dim=2, out_dim=2, patch_size=(1, 1, 1),
        has_image_input=False, seperated_timestep=True,
        video_attention_mask_mode="first_frame_causal")
    dino = modules.dino_video_dit.DinoVideoDiT(**common, dino_dim=2)
    action = modules.action_dit.ActionDiT(**common, action_dim=2)
    mixtures = dict(video=video, dino=dino, action=action)
    geometry = None
    config = dict(enabled=enabled, mode="expert", hidden_dim=8,
                  ffn_dim=16, freq_dim=8, eps=1e-6, patch_size=[1, 2, 2])
    if enabled:
        geometry = modules.geometry_video_dit.GeometryVideoDiT(
            **{**common, "hidden_dim": 8, "ffn_dim": 16}, patch_size=(1, 2, 2))
        mixtures["geometry"] = geometry
    mot = modules.mot.MoT(mixtures, mot_checkpoint_mixed_attn=checkpointing,
                          inference_optional_experts=("geometry",) if enabled else ())
    model = auxiliary.Model(
        video_expert=video, dino_video_expert=dino, action_expert=action,
        mot=mot, vae=auxiliary.TinyVAE(), dino_encoder=None, text_dim=8,
        geometry_config=config, geometry_video_expert=geometry,
        loss_lambda_geometry=.05 if enabled else 0, torch_dtype=dtype)
    return model.to(dtype=dtype)


def small_sample():
    batch = auxiliary.sample()
    return {key: value[:1] for key, value in batch.items()}


@pytest.mark.parametrize("checkpointing,dtype", [
    (False, torch.float32), (True, torch.float32), (True, torch.bfloat16)])
def test_real_four_expert_loss_and_gradients(real_modules, checkpointing, dtype):
    model = make_model(real_modules, checkpointing=checkpointing, dtype=dtype)
    auxiliary.Trainer._apply_dit_only_train_mode(model)
    batch = small_sample()
    loss, metrics = model.training_loss(batch)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(sum(metrics[k] for k in (
        "loss_video", "loss_dino", "loss_action", "loss_geometry")), rel=1e-4)
    loss.backward()
    assert model.geometry_head is None
    assert model.mot.expert_order == ["video", "dino", "action", "geometry"]
    assert all(p.requires_grad for p in model.geometry_video_expert.parameters())
    assert model.geometry_video_expert.patch_embedding.weight.grad.abs().sum() > 0
    assert model.geometry_video_expert.blocks[0].self_attn.q.weight.grad.abs().sum() > 0
    assert model.geometry_video_expert.head.head.weight.grad.abs().sum() > 0
    assert model.action_expert.action_encoder.weight.grad.abs().sum() > 0
    # Isolate geometry loss to check it also trains the shared RGB/DINO path.
    model.zero_grad()
    model.loss_lambda_video = model.loss_lambda_dino = model.loss_lambda_action = 0
    model.training_loss(batch)[0].backward()
    assert model.video_expert.patch_embedding.weight.grad.abs().sum() > 0
    assert model.dino_video_expert.input_projection.weight.grad.abs().sum() > 0


def test_depth_flow_mask_and_zero_valid_samples():
    pred = torch.tensor([100., 3., 100., 100.]).reshape(1, 1, 4, 1, 1).requires_grad_()
    target = torch.tensor([0., 1., float("nan"), 0.]).reshape_as(pred)
    pad = torch.tensor([[False, False, False, True]])
    loss = auxiliary.Model._compute_geometry_flow_loss_per_sample(pred, target, None, pad)
    assert loss.item() == 4  # MSE velocity loss, not depth L1
    loss.sum().backward()
    assert pred.grad.flatten().tolist() == [0., 4., 0., 0.]
    loss = auxiliary.Model._compute_geometry_flow_loss_per_sample(
        pred, target, torch.zeros_like(pred).bool(), pad)
    assert loss.item() == 0 and torch.isfinite(loss)


def test_invalid_depth_never_enters_mot(real_modules):
    model, batch = make_model(real_modules), small_sample()
    batch["geometry_valid"][:, :, 2] = False
    batch["geometry"][:, :, 2] = 99
    batch["image_is_pad"][:, -1] = True
    built = model.build_inputs(batch)
    assert built["geometry"][:, :, 2].count_nonzero() == 0
    assert built["geometry"][:, :, -1].count_nonzero() == 0
    batch.pop("geometry")
    with pytest.raises(ValueError, match="batch has no"):
        model.training_loss(batch)


def prepare_states(model):
    batch = small_sample()
    context, context_mask = batch["context"], batch["context_mask"]
    t = torch.full((1,), 500.)
    return dict(
        video=model.video_expert.pre_dit(
            x=batch["vae_latents"], timestep=t, context=context,
            context_mask=context_mask, fuse_vae_embedding_in_latents=True),
        dino=model.dino_video_expert.pre_dit(
            x=batch["dino_latents"], timestep=t, context=context,
            context_mask=context_mask, fuse_vae_embedding_in_latents=True),
        action=model.action_expert.pre_dit(
            action_tokens=batch["action"], timestep=t, context=context, context_mask=context_mask),
        geometry=model.geometry_video_expert.pre_dit(
            x=batch["geometry"], timestep=t, context=context, context_mask=context_mask))


def call_mot(model, states, mask):
    return model.mot(
        embeds_all={k: s["tokens"] for k, s in states.items()},
        freqs_all={k: s["freqs"] for k, s in states.items()},
        t_mod_all={k: s["t_mod"] for k, s in states.items()},
        context_all={k: dict(context=s["context"], mask=s["context_mask"]) for k, s in states.items()},
        attention_mask=mask)


def masks(model, states):
    kwargs = dict(vae_seq_len=states["video"]["tokens"].shape[1],
                  dino_seq_len=states["dino"]["tokens"].shape[1],
                  vae_tokens_per_frame=states["video"]["meta"]["tokens_per_frame"],
                  dino_tokens_per_frame=states["dino"]["meta"]["tokens_per_frame"])
    base = model._build_tribranch_attention_mask(
        **kwargs, action_seq_len=states["action"]["tokens"].shape[1], device=torch.device("cpu"))
    return base, model._extend_geometry_attention_mask(
        base, **kwargs, geometry_seq_len=states["geometry"]["tokens"].shape[1])


def test_geometry_cannot_leak_into_action_across_real_layers(real_modules):
    model = make_model(real_modules).eval()
    states = prepare_states(model)
    base, full = masks(model, states)
    with torch.no_grad():
        a = call_mot(model, states, full)
        states["geometry"]["tokens"] += 100 * torch.randn_like(states["geometry"]["tokens"])
        b = call_mot(model, states, full)
        c = call_mot(model, {k: s for k, s in states.items() if k != "geometry"}, base)
    assert torch.allclose(a["action"], b["action"], atol=2e-6, rtol=2e-6)
    assert torch.allclose(a["action"], c["action"], atol=2e-6, rtol=2e-6)
    assert not torch.allclose(a["video"][:, 8:], b["video"][:, 8:])
    assert not torch.allclose(a["dino"][:, 8:], b["dino"][:, 8:])
    # Missing expert must fail during training, instead of silently dropping it.
    with pytest.raises(ValueError, match="Missing expert tokens"):
        call_mot(model, {k: s for k, s in states.items() if k != "geometry"}, base)


def test_original_action_helpers_work_without_depth_and_cache_matches(real_modules):
    model = make_model(real_modules).eval()
    sample = small_sample()
    args = dict(first_frame_vae_latents=sample["vae_latents"][:, :, :1],
                first_frame_dino_latents=sample["dino_latents"][:, :, :1],
                latents_action=sample["action"], timestep_action=torch.tensor([500.]),
                context=sample["context"], context_mask=sample["context_mask"],
                fuse_vae_embedding_in_latents=True)
    uncached = model._predict_action_noise(**args)
    cache = model._prefill_action_static_cache(
        first_frame_vae_latents=args["first_frame_vae_latents"],
        first_frame_dino_latents=args["first_frame_dino_latents"],
        latents_action=args["latents_action"], context_video=args["context"],
        context_video_mask=args["context_mask"], context_dino=args["context"],
        context_dino_mask=args["context_mask"], fuse_vae_embedding_in_latents=True)
    cached = model._predict_action_noise_with_static_cache(
        latents_action=args["latents_action"], timestep_action=args["timestep_action"],
        context_action=args["context"], context_action_mask=args["context_mask"],
        static_kv_cache=cache[0], attention_mask=cache[1], static_seq_len=cache[2])
    assert torch.allclose(uncached, cached, atol=2e-6, rtol=2e-6)


def test_four_expert_checkpoint_roundtrip_and_baseline_init(real_modules, tmp_path):
    model, path = make_model(real_modules), tmp_path / "four.pt"
    model.save_checkpoint(path, step=17, safe_serialization=True)
    restored = make_model(real_modules)
    assert restored.load_checkpoint(path) == 17
    assert all(torch.equal(v, restored.mot.state_dict()[k]) for k, v in model.mot.state_dict().items())
    baseline = make_model(real_modules, enabled=False)
    baseline_path = tmp_path / "three.pt"
    baseline.save_checkpoint(baseline_path)
    before = {k: v.clone() for k, v in restored.geometry_video_expert.state_dict().items()}
    restored.load_checkpoint(baseline_path)
    assert all(torch.equal(v, restored.geometry_video_expert.state_dict()[k]) for k, v in before.items())
    payload = torch.load(path, weights_only=False)
    key = next(k for k in payload["mot"] if k.startswith("mixtures.geometry."))
    del payload["mot"][key]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="incomplete"):
        restored.load_checkpoint(path)


@pytest.mark.parametrize("task,size,tokens", [
    ("libero_geometry_mot", [56, 112], 3528),
    ("robotwin_geometry_mot", [96, 80], 4320)])
def test_new_tasks_resolve_and_preserve_original_components(task, size, tokens):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task}"])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert cfg.model.geometry_config.mode == "expert"
    assert cfg.model.semantic_history_config.enabled
    assert list(cfg.data.train.geometry_size) == size
    assert 9 * (size[0] // 4) * (size[1] // 4) == tokens
    assert cfg.batch_size * cfg.gradient_accumulation_steps == 16
    assert resolved["model"]["geometry_config"]["use_gradient_checkpointing"] is True


def test_geometry_patch_codec_uses_independent_blocks_and_rejects_bad_shapes(real_modules):
    model = make_model(real_modules)
    geo, dino = model.geometry_video_expert, model.dino_video_expert
    states = prepare_states(model)
    out = geo.post_dit(states["geometry"]["tokens"], states["geometry"])
    assert out.shape == (1, 1, 9, 4, 8)
    assert geo.blocks[0] is not dino.blocks[0]
    with pytest.raises(ValueError, match="no clean"):
        geo.pre_dit(torch.rand(1, 1, 9, 4, 8), torch.tensor([500.]), torch.rand(1, 3, 8),
                    fuse_vae_embedding_in_latents=True)
    with pytest.raises(ValueError):
        geo.pre_dit(torch.rand(1, 1, 9, 3, 8), torch.tensor([500.]), torch.rand(1, 3, 8))


def test_runtime_factory_constructs_the_fourth_expert(real_modules, monkeypatch):
    prototype = make_model(real_modules, enabled=False)
    components = types.SimpleNamespace(
        dit=prototype.video_expert, vae=prototype.vae, text_encoder=None,
        tokenizer=None, dit_path=None, vae_path=None, text_encoder_path=None, tokenizer_path=None)
    components.vae.upsampling_factor = 1  # tiny RGB/VAE grids coincide in this fixture

    class CachedDino(torch.nn.Module):
        def __init__(self, input_resolution, patch_size, latent_spatial_pool, **kwargs):
            super().__init__()
            self._loaded = False
            self.input_resolution = input_resolution
            self.patch_size = patch_size
            self.latent_spatial_pool = latent_spatial_pool

    scope = auxiliary.Model.from_config.__func__.__globals__
    for key, value in dict(
        load_wan22_ti2v_5b_components=lambda **kwargs: components,
        DinoVideoEncoder=CachedDino, DinoVideoDiT=real_modules.dino_video_dit.DinoVideoDiT,
        GeometryVideoDiT=real_modules.geometry_video_dit.GeometryVideoDiT,
        ActionDiT=real_modules.action_dit.ActionDiT, MoT=real_modules.mot.MoT).items():
        monkeypatch.setitem(scope, key, value)

    tree = ast.parse((ROOT / "src/fastwam/runtime.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "create_fastwam_vae_dino_mot")
    # The class is already loaded above; omit its relative import only.
    function.body = [n for n in function.body if not isinstance(n, ast.ImportFrom)]
    runtime_scope = dict(torch=torch, DictConfig=DictConfig, OmegaConf=OmegaConf,
                         FastWAMVAEDinoMoT=auxiliary.Model)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "runtime.py", "exec"), runtime_scope)
    common = dict(hidden_dim=12, ffn_dim=24, text_dim=8, freq_dim=8, eps=1e-6,
                  num_heads=1, attn_head_dim=12, num_layers=2, use_gradient_checkpointing=False)
    cfg = OmegaConf.create(dict(
        dino_config=dict(model_name="test", feature_dim=2, load_backbone=False,
                         input_resolution=[2, 4], patch_size=1, latent_spatial_pool=[1, 1]),
        vae_video_dit_config=dict(text_dim=8), dino_video_dit_config=common,
        action_dit_config=dict(**common, action_dim=2),
        geometry_config=dict(enabled=True, mode="expert", hidden_dim=8, ffn_dim=16,
                             freq_dim=8, patch_size=[1, 2, 2], use_gradient_checkpointing=False),
        action_scheduler=dict(train_shift=5, infer_shift=5, num_train_timesteps=1000),
        loss=dict(lambda_geometry=.05)))
    model = runtime_scope[function.name](**cfg, device="cpu", model_dtype=torch.float32,
                                         skip_dit_load_from_pretrain=True)
    assert model.mot.mixtures["geometry"] is model.geometry_video_expert
    assert len(model.geometry_video_expert.blocks) == len(model.video_expert.blocks)
    assert torch.isfinite(model.training_loss(small_sample())[0])


def test_rejects_indirect_future_leakage(real_modules):
    model = make_model(real_modules)
    model.video_expert.video_attention_mask_mode = "bidirectional"
    with pytest.raises(ValueError, match="must not read future"):
        model.training_loss(small_sample())
