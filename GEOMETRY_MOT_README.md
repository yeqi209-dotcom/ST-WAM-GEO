# Geometry as the fourth MoT expert

此目录是可独立安装、训练的完整 ST-WAM 副本。原始目录
`ST-WAM-Semantic-Temporal-World-Action-Model` 和先前的辅助 head 版本
`ST-WAM-Semantic-Temporal-World-Action-Model-geometry` 均保持原样。
基础安装和权重准备见 [README](README.md)；旧辅助 head 实验说明保留在
[GEOMETRY_README](GEOMETRY_README.md)，其 `*_geometry_aux` task 仍可运行。
本版本请选 `libero_geometry_mot` 或 `robotwin_geometry_mot`。

## 模型与训练目标

```text
RGB ── frozen Wan VAE ── noisy VAE latents ── VAE expert ────── VAE flow loss
RGB ── frozen DINOv3 ─── noisy DINO features ─ DINO expert ───── DINO flow loss
actions ──────────────── noisy actions ────── action expert ─── action flow loss
aligned depth ────────── noisy depth maps ─── geometry expert ─ geometry flow loss
                                              ↑ mixed MoT attention ↑
```

`GeometryVideoDiT` 拥有独立的 patch embedding、时间/上下文投影、所有 DiT blocks
和深度 velocity 输出投影。它复用 `DinoVideoDiT` 的通用 pre/post-DiT 和 patch
编解码实现；没有调用 DINO 编码器去编码深度，也没有共享 DINO expert 参数。
geometry 作为第四项注册到 `MoT.mixtures`，顺序是
`video, dino, action, geometry`，没有实例化辅助 `geometry_head`。

默认 geometry hidden width 为 256、FFN width 为 1024；层数、attention heads
和 head dimension 从 VAE expert 继承。正式配置为 30 层、24 heads、head_dim=128，
因此 attention 内部宽度仍为 3072，默认新增 **207,159,056 参数**（约 2.07 亿）。
这不是“每个 geometry token 只有 256 维，所以所有注意力开销也缩小至 256 维”。
geometry expert 随机初始化，基础三 expert 和 CAIR 初始化方式沿用 baseline。

深度输入为 `[B,1,T,H,W]`，默认每窗口 T=9。以 `[1,4,4]` 对深度做空间 patchify，
不合并时间；输出会还原成原尺寸的单通道 velocity，没有额外深度 VAE/teacher
在线前向。LIBERO 的 56×112 深度产生 3528 个 geometry tokens；RoboTwin 的
96×80 深度产生 4320 个 geometry tokens。目标空间尺寸必须可被 patch size 整除。

geometry 与视频分支使用同一个 video timestep 和 scheduler，但采样独立 Gaussian
noise。所有 geometry 帧都加噪，不向模型提供 clean f0 depth 条件。目标为 scheduler
定义的 flow velocity；loss 是有效未来像素的 **MSE**，乘同一 video timestep weight。
它与旧版辅助 head 的 clean depth L1 目标不同。

```text
L = 1.0 L_vae_flow + 0.02 L_dino_flow + 1.0 L_action_flow + 0.05 L_geometry_flow
```

geometry 的 f0、`image_is_pad` 帧和 `geometry_valid=false` 像素不计入 flow loss；
每样本按有效未来像素平均，再按 batch 平均。全无有效像素的样本贡献零 geometry loss。
无效/非有限深度和 padding 对应的 clean 输入先归零，避免错误标签作为输入参与混合注意力。
`loss_geometry_raw` 是包含 timestep weight、未乘 lambda 的值；`loss_geometry` 已乘 lambda。
VAE encoder、DINO encoder、Qwen backbone 的冻结策略和 CAIR 的 action-only 注入均沿用 baseline；
geometry expert 位于 `model.dit` 中，自动加入原 trainer 的优化器和 DeepSpeed 管理。

## 注意力与部署

表中行是 query，列是 key；“未来 world”指 VAE/DINO 的未来 tokens。

| Query | 当前 VAE/DINO | 未来 world | action | geometry |
|---|---|---|---|---|
| 当前 VAE/DINO | 是 | 否 | 否 | 否 |
| 未来 world | 沿用 baseline | 沿用 baseline | 否 | 是 |
| action | 是 | 否 | 是 | 否 |
| geometry | 是 | 是 | 否 | 是 |

当前 VAE/DINO 和 action 构成一个不读取未来 world/geometry 的闭合集合；这也排除了
“geometry → future RGB/DINO → current tokens → action”的间接泄漏。expert 模式会
检查原三分支 mask 是否满足这一约束；若切换到会让当前帧读取未来帧的 bidirectional
world mask，将显式报错。

因此保留原 `infer_action` 接口和三分支 action denoising 计算路径。推理不需要深度
标签、几何 teacher、geometry cache 或 geometry expert 前向。`MoT.forward` 仅在
`torch.no_grad()` 下允许省略注册为可选的 geometry expert；训练缺失分支会报错。
原 static-KV cache 实现继续只预填 VAE/DINO 当前 tokens。

建议用原版 baseline task 加载新 checkpoint 评测，从而不实例化 geometry expert，
连它的参数显存也可省去：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_wan5b_dino_s_aux_mot_short_qwen3vl_hist4_vae_mmap_2cam_224_1e-4 \
  ckpt=/path/to/geometry_mot_checkpoint.pt \
  EVALUATION.num_trials=50 \
  EVALUATION.output_dir=./evaluate_results/libero/geometry_mot \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=2 \
  model.semantic_history_config.vlm_model_name_or_path=/path/to/Qwen3-VL-4B-Instruct
```

baseline loader 会非严格加载 MoT，忽略 `mixtures.geometry.*`，保留原三分支及 CAIR
权重。这里只是推理计算路径/输入接口保持一致；训练后参数和动作效果会改变，收益需要实测。
本实现也不是 GeoSem-WAM 的完整复现。

## 数据和缓存

几何标签接口沿用辅助 head 版本，已有对齐的 geometry window cache 可复用。
缓存必须和所选 dataset/split 的 resolved sample index、episode-clamped frame indices、
相机拼接、方向和裁剪一致。更改数据集顺序或切分后必须重新生成对应的 cache。

LIBERO 顺序是 `image` 在左、`wrist_image` 在右；RoboTwin 为上方主相机、下方左右腕相机。
默认假设 metric depth（米），以 `clip(depth,0,10)/10` 归一化。MuJoCo 深度缓冲需先
转换为 metric depth；普通单目 teacher 的 relative/inverse depth 不能直接当米。
若离线 teacher 输出已统一到 `[0,1]`，设置
`data.train.geometry_normalize=none`，并按输出约定设置 `geometry_min_depth`。
本版本不包含尺度不变损失或深度 teacher 推理程序。对各相机分别估深度再拼接。

推荐每个窗口的缓存内容如下：

```python
torch.save({
    "sample_index": 42,
    "frame_indices": [42, 46, 50, 54, 58, 62, 66, 70, 74],
    "geometry": depth.float(),       # [T,H,W]，T=9
    "geometry_valid": valid.bool(),  # 可选，与深度同形状
}, "00000042.pt")
```

也支持完整 T=33 窗口、单通道维度、`.pth` 和数值 `.npy`；详见
[缓存格式](GEOMETRY_README.md)。当前 window cache 要求 `global_sample_stride=1`。
打包器只消费已生成的逐帧深度，不会自动下载或运行深度模型：

```bash
python scripts/prepare_geometry_cache.py \
  --task libero_geometry_mot \
  --frame-depths /path/to/aligned_depth_frames.npy \
  --output /root/autodl-tmp/dataset/geometry_cache/libero/train

python scripts/prepare_geometry_cache.py \
  --task libero_geometry_mot --check-only \
  --output /root/autodl-tmp/dataset/geometry_cache/libero/train
```

`--frame-depths` 是 `[len(dataset),H,W]`，按所选数据集全局帧索引排序。可以加
`--frame-valid`、`--start/--limit`、`--overwrite` 及 Hydra 数据路径覆盖。
RoboTwin 将 task 改成 `robotwin_geometry_mot`。

LIBERO RGB/动作演示仍使用四套 LeRobot 数据。若放在实例
`/root/autodl-tmp/dataset/libero/`，需覆盖 `data.train.dataset_dirs`，或把本副本默认
`data/libero_mujoco3.3.2/` 下的四个数据目录链接到该路径。打包和训练必须使用同样的目录顺序。

文本/VAE/DINO 缓存继续用 baseline task 预计算，以免预计算阶段尝试读取尚未生成的
geometry cache：

```bash
BASE_TASK=libero_wan5b_dino_s_aux_mot_short_qwen3vl_hist4_vae_mmap_2cam_224_1e-4
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=${BASE_TASK}
torchrun --standalone --nproc_per_node=8 scripts/precompute_vae_latents.py task=${BASE_TASK}
torchrun --standalone --nproc_per_node=8 scripts/precompute_dino_latents.py task=${BASE_TASK}
```

路径覆盖方法同原 README。已有正确的 baseline 缓存可直接复用。

## 训练与 checkpoint

安装本目录 `pip install -e .`，配置好 baseline 权重和数据缓存后：

```bash
bash scripts/train_zero1.sh 8 \
  task=libero_geometry_mot \
  data.train.geometry_cache_dir=/root/autodl-tmp/dataset/geometry_cache/libero/train \
  model.semantic_history_config.vlm_model_name_or_path=/path/to/Qwen3-VL-4B-Instruct \
  output_dir=./runs/libero_geometry_mot \
  wandb.enabled=false
```

RoboTwin 使用 `task=robotwin_geometry_mot` 和自己的 cache。默认每卡 microbatch=1、
梯度累积=16，保持这两份 baseline YAML 的每卡有效 batch=16；8 卡时 global batch=128。
若要 global batch=1024，可在 8 卡、microbatch=1 下设 `gradient_accumulation_steps=128`。
这只是配置起点，不是显存保证；新增 tokens 增加了所有混合注意力层的开销。修改 GPU
数量时同步核算 global batch。先用真实数据做短跑：

```bash
bash scripts/train_zero1.sh 1 \
  task=libero_geometry_mot \
  data.train.geometry_cache_dir=/path/to/libero_geometry_windows \
  model.semantic_history_config.vlm_model_name_or_path=/path/to/Qwen3-VL-4B-Instruct \
  output_dir=./runs/libero_geometry_mot_smoke \
  max_steps=2 batch_size=1 gradient_accumulation_steps=1 \
  num_workers=0 save_every=1 eval_every=1 wandb.enabled=false
```

用 `resume=/path/to/baseline_weights.pt` 初始化时，只允许缺少第四 expert 的完整键集合，
现有三 expert 必须完整匹配；geometry 保留当前初始化。新 checkpoint 把第四 expert
权重一起写入 `mot` 并保存 `geometry_config`。同架构新权重可完整往返；geometry
配置不匹配或第四 expert 权重不完整会显式报错。

baseline `.pt` 是权重初始化，不恢复 step/optimizer。不要用原三 expert 的完整
Accelerate/DeepSpeed state 目录恢复新增结构的 optimizer；仅本版本、同配置产生的
state 目录适用于完整训练恢复。

继承任务没有独立验证集，runtime 会复用训练集作为 val；日志的 `val_loss` 不能作为
held-out 泛化结果。单独设置 val 时应同时启用 geometry 并生成该 split 自己的标签缓存。
建议与 baseline 和辅助 head 版比较动作成功率，同时扫描 lambda=0.01/0.05/0.1。

## 验证范围

```bash
python -m pytest -q tests/test_geometry_auxiliary.py tests/test_geometry_mot.py
```

测试使用 CPU 和小尺寸真实 WanVideoDiT、DinoVideoDiT、ActionDiT、GeometryVideoDiT、MoT，
覆盖 float32/bfloat16 前向与反向、checkpointing、四项 flow loss、有效性/padding 屏蔽、
geometry 及 RGB/DINO 梯度、两层混合注意力的深度隔离、无深度 action 预测、static KV cache
一致性、runtime factory 接线、Hydra 配置、baseline 初始化及 checkpoint 完整性。
另有旧辅助 head 的回归测试。大模型加载由小型 components 替代，测试不下载权重。
本地 Python 3.12 / CPU PyTorch 2.7.1 下，以上 **35 项测试通过**；源码语法检查通过。

尚未执行完整 Wan5B/Qwen 前向、多 GPU/DeepSpeed 训练、真实深度标签生成、GPU 峰值显存测量
或 LIBERO/RoboTwin 成功率评测。本目录不包含数据、外部模型权重或已训练 checkpoint。
