此文件保留 **geometry auxiliary supervision head** 的实验说明。本目录的新主版本是
[geometry 第四个 MoT expert](GEOMETRY_MOT_README.md)，使用 `*_geometry_mot` task；
下面的 `*_geometry_aux` task 仍对应旧辅助 head 路径。安装、预训练权重和基础缓存依赖
见 [README](README.md)。以下旧版命令均在本目录执行。

## 实现范围

保留原来的 VAE + DINO 双空间和 action MoT、CAIR。训练时，从 `tokens_out["dino"]`（MoT 输出、DINO flow 输出层之前）预测每一帧的深度图，作为第三种 future representation 的辅助监督目标。

```text
RGB -> VAE  -> VAE expert  ─┐                 -> VAE flow loss
RGB -> DINO -> DINO expert ─┼─ 原有三分支 MoT -> DINO flow loss
action     -> action expert┘                 -> action flow loss
                             DINO hidden tokens
                                     └-> geometry head -> future depth L1
```

几何 head 是 LayerNorm + Linear + 两层 Conv2d/GroupNorm/GELU + 单通道输出，并对每帧单独做空间上采样。这是轻量空间 decoder，**没有实现 DPT 的多层、多尺度融合，也不是 GeoSem-WAM 的完整复现**。没有加入第四个 diffusion expert、geometry 输入、geometry token 或几何编码器。监督梯度回传到 head 和现有 MoT；VAE、DINO、Qwen backbone 的冻结方式沿用 baseline。

默认损失为：

```text
L = 1.0 L_vae_flow + 0.02 L_dino_flow + 1.0 L_action_flow + 0.05 L_geometry
```

前三项保留原来的 diffusion timestep 权重；geometry 是 clean depth 的直接回归，不额外乘 flow timestep 权重。geometry L1 在每个样本内按有效未来像素取平均，再对 batch 平均。排除当前帧 f0、`image_is_pad` 标记的帧以及无效深度。全无有效未来像素的样本贡献零损失。日志中的 `loss_geometry` 已乘系数，`loss_geometry_raw` 未乘系数。

支持默认 `flat` token 布局，也正确恢复 `view` 布局的 `(T,view,H,W)` 顺序。要求 DINO 时间 patch 大小为 1、`dino_future_mode=predict`、单通道深度。当前 window cache 要求 `global_sample_stride=1`。

## 准备几何监督数据

本仓库的 LeRobot 读取链路没有可直接使用的深度标签；本次修改也不包含训练数据或预生成标签。需要先准备 RGB 对齐的深度，可来自仿真器渲染或离线深度 teacher。若用 teacher，先对各摄像机分别估计深度，再按 RGB 的处理方式组成 mosaic；不要把摄像机拼图当作一台真实相机去估计几何。

优先使用同一单位的 metric depth（默认米）。MuJoCo 的归一化深度缓冲需要先依据相机 near/far 转换成 metric depth，不能直接当米使用。普通 monocular teacher 的相对深度/逆深度也不能直接当 metric depth：应明确输出约定并统一尺度，或将标签预处理到一致的 `[0,1]` 范围后设 `geometry_normalize=none`。本版不包含尺度不变损失；跨帧/摄像机尺度漂移会降低监督质量。

深度必须和最终 RGB 同方向、同一帧、同一裁剪及 camera 顺序：

| 数据 | RGB mosaic | 默认 geometry target |
| --- | --- | --- |
| LIBERO | `image` 在左、`wrist_image` 在右；224×448 | 56×112 |
| RoboTwin | `cam_high` 在上（256×320），左右腕相机在下（各128×160）；384×320 | 96×80 |

读取器会检查宽高比，但无法判断标签是否实际翻转或相机顺序是否正确，需要在生成标签时核对。深度/有效性使用最近邻缩放，避免 NaN 和深度边界混合。

每个窗口一个文件，名称对应 **当前 dataset/split 中 resolved sample index**，例如 `00000042.pt`；不是 episode 内编号，也不是 DataLoader 的 batch 编号。推荐格式：

```python
torch.save({
    "sample_index": 42,
    "frame_indices": [42, 46, 50, 54, 58, 62, 66, 70, 74],
    "geometry": depth.float(),       # [T,H,W] 或 [1,T,H,W] 或 [T,1,H,W]
    "geometry_valid": valid.bool(),  # 可选，形状与 geometry 相同
}, "00000042.pt")
```

默认 `T=9`，即 offsets `[0,4,8,...,32]`，包含 f0。也支持完整 `T=33` 窗口，读取时会同时选取深度、mask 和 frame_indices。跨 episode 边界应重复最后一帧，同时由原始 `image_is_pad` 排除 padding。若保存 `frame_indices`，必须是 episode-clamped 的全局帧编号，读取器会逐项检查。

也接受 `.pth` 或直接 tensor 的 `.pt`，以及纯数值 `.npy`（不使用 pickle）。`sample_index` 和 `frame_indices` 可以省略，但推荐保留以检查错位。非有限值和默认 `depth <= 0` 的像素无效；支持额外有效性 mask。缺失/损坏的 cache 会显式报错，不会静默改用其他样本。

默认以固定尺度 `clip(depth,0,10)/10` 归一化。可修改 `geometry_max_depth`，但同一实验应保持尺度一致；这也会影响 L1 量级和 lambda 的含义。已预归一化且 0 是有效标签时，设置：

```bash
data.train.geometry_normalize=none data.train.geometry_min_depth=null
```

更改 dataset_dirs 的顺序、训练/验证划分或 episode_filter 后，必须按新索引重建 cache。与 baseline 的 latent cache 一样，episode_filter 默认禁止复用旧 cache；只有确认 cache 与过滤后的索引一致时才设置 `allow_latent_cache_with_episode_filter=true`。

## 从逐帧深度打包与检查

`scripts/prepare_geometry_cache.py` 将现有深度打包成窗口，不执行深度 teacher。输入 `depth_frames.npy` 是 `[len(dataset),H,W]` 的数值数组，帧顺序必须与所选任务的 LeRobot 数据索引一致；图像已完成相同的 camera 拼接、裁剪和方向转换。脚本只加载数据元信息和 episode 索引，不加载大模型或 RGB，也不计算归一化统计。

```bash
python scripts/prepare_geometry_cache.py \
  --task libero_geometry_aux \
  --frame-depths /path/to/depth_frames.npy \
  --frame-valid /path/to/optional_valid_frames.npy

python scripts/prepare_geometry_cache.py \
  --task libero_geometry_aux --check-only
```

没有单独 mask 时省略 `--frame-valid`。RoboTwin 使用 `--task robotwin_geometry_aux`。可用 `--output` 指定缓存目录，训练时也需要设置相同的 `data.train.geometry_cache_dir`；脚本支持 `--start/--limit` 分段处理、`--overwrite` 重建和后接 Hydra 数据路径覆盖。已有外部生成的窗口可以直接 `--check-only`，无需逐帧 bank。逐窗口文件较多，正式大规模实验需预留存储和 I/O。

## 训练

在配置好 baseline 的 Linux/CUDA 环境、预训练权重及原有文本/VAE/DINO 数据缓存后：

```bash
bash scripts/train_zero1.sh 8 \
  task=libero_geometry_aux \
  data.train.geometry_cache_dir=/path/to/libero_geometry_windows \
  model.semantic_history_config.vlm_model_name_or_path=/path/to/Qwen3-VL-4B-Instruct \
  output_dir=./runs/libero_geometry_aux \
  wandb.enabled=false
```

RoboTwin 将 task 换成 `robotwin_geometry_aux` 并设置对应的 geometry cache。batch、梯度累积、VAE/DINO 缓存策略等继承原任务，根据可用显存调整。可先用 `max_steps=2 save_every=1 eval_every=1 num_workers=0 batch_size=1` 做真实数据 smoke run。

支持通过 `resume=/path/to/baseline_weights.pt` 从原版权重初始化，新增 head 随机初始化。`.pt` 只加载权重，不恢复 optimizer/step。原版的完整 Accelerate/DeepSpeed state 目录不能用于恢复新增参数后的优化器；新版本训练产生的 state 才能用于同配置继续训练。Checkpoint 会单独保存 `geometry_head` 和完整 decoder 配置，加载时校验结构。

默认 lambda 为 0.05，可通过 `model.loss.lambda_geometry` 调整。建议与无 geometry 的原任务比较，并至少尝试 0.01/0.05/0.1；需要动作成功率实验验证收益。本任务没有启动正式训练。

两份任务都继承 baseline 的无独立验证集设置（RoboTwin 显式 `val:null`，LIBERO 未配置 val）；runtime 会复用训练集作为 val，因此训练日志的 `val_loss` **不是 held-out 泛化结果**。如建立独立验证集，需要同步开启 `data.val.geometry_enabled`，指定该 split 自己的 cache、相同尺度和 target size；不能沿用 train 索引的 cache。

## 推理与评测

`infer_action`、三项 action noise/cache helper、MoT 及注意力掩码均保持 baseline 实现。Geometry head 不在 action inference 调用链中，推理不需要深度、teacher 或 geometry cache。这里指计算路径/接口不变，训练后模型权重和动作结果会发生变化。

可以直接用原版评测 task 加载新增版本的 `.pt` 权重；原版 loader 会读取 MoT/CAIR 等权重并忽略额外的 geometry 字段，从而连 head 也无需实例化。例如：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_wan5b_dino_s_aux_mot_short_qwen3vl_hist4_vae_mmap_2cam_224_1e-4 \
  ckpt=/path/to/geometry_trained_checkpoint.pt \
  EVALUATION.num_trials=50 \
  EVALUATION.output_dir=./evaluate_results/libero/geometry_aux \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=2
```

保持训练时的 backbone、CAIR 和输入分辨率设置一致，其他评测参数参考原版 README。

## 验证与边界

```bash
python -m pytest -q tests/test_geometry_auxiliary.py
```

本地使用 Python 3.12 / CPU PyTorch 2.7.1，21 项测试通过，覆盖 float32/bfloat16 head 前向/反向、f0/无效/padding 屏蔽、完整/稀疏深度窗口及元信息、view token 空间顺序、实际 training_loss 与小模型反向传播、验证样本构造、权重往返、baseline 初始化及旧 loader 读取新权重。还检查了关闭 geometry 时 loss 与原版一致，以及打包/检查 CLI（使用合成深度和小型 dataset 元信息替身）。Hydra 两个新 task 均能完整解析。测试隔离加载实际 model/trainer 类逻辑，以小型 experts 替代大模型；未执行 Wan5B/Qwen 前向、仿真环境评测、真实数据打包或多 GPU/DeepSpeed 训练。

动作推理入口/三个 helper/注意力掩码与原代码做了 AST 等价检查。完整原目录的文件均包含在此副本；仅修改模型、dataset、runtime、trainer 的必要接线，新增 head、target loader、配置、打包脚本和测试。未复制任何外部训练数据、深度模型或预训练权重。
