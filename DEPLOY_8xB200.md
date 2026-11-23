# ST-WAM-GEO：单机 8×B200 部署与下载手册

编写日期：2026-11-19；checkpoint 保存策略更新：2026-11-23。依据本目录的实际源码、Hydra 配置及下载仓库文件元数据核对。

本文交给服务器上的合作者 / agent 执行。**主流程以 LIBERO 四个 suite 的训练为目标**，RoboTwin 的差异在末尾单列。本文中的服务器根目录 `/data/stwam-geo` 是示例，执行前改为容量充足的本地 NVMe 路径；所有命令使用 **Linux Bash**。

本目录的训练任务是 `libero_geometry_mot`：VAE / DINO / Action / Geometry 四个 MoT experts，保留 Qwen3-VL + DINO 历史 CAIR。它不是旁边的 `ST-WAM-GEO（infer_without_dino）` 版本，也不是 `geometry_aux` 辅助 head 版本。从零部署指安装环境、准备基础预训练权重后训练；不是将全部模型随机初始化。

本文没有在 B200 服务器执行过。下载大小中标为“元数据”的值已查询；解压大小、缓存预算和显存范围属于估算。安装版本组合仍须通过下文的真实加载与训练验收。

## 1. 执行清单与完成标准

| 顺序 | 工作 | 完成标准 |
|---|---|---|
| 1 | 传输完整 `ST-WAM-GEO` 源码 | 配置、scripts、src、tests 均在；Python 导入来自这个副本 |
| 2 | B200 驱动、CUDA Toolkit、Python 环境 | 8 卡可见，bf16 计算和 NCCL all-reduce 通过 |
| 3 | 修正部署副本的三项依赖版本 | Qwen3-VL 可导入；TorchCodec 与 Torch 配套；`pip check` 通过 |
| 4 | 下载 LIBERO LeRobot 2.1 四个压缩包并解压 | 四套目录齐全，视频与 Parquet 可读，记录总帧数 N |
| 5 | 下载 Wan DiT、Wan VAE、UMT5、tokenizer、DINO-S、Qwen3-VL | 固定仓库 revision；文件完整；不混用不同格式权重 |
| 6 | 生成 ActionDiT、DinoVideoDiT 初始化文件 | 两个 `.pt` 有效，参数形状匹配 |
| 7 | 生成文本、VAE、DINO 缓存 | VAE 为 window_mmap；DINO 为 frame_mmap，支持历史读取 |
| 8 | 准备对齐深度并打包 geometry cache | 每个训练 sample 都有窗口，索引/相机/尺度正确 |
| 9 | 数据检查、小模型测试、单卡及 8 卡短跑 | 四项 loss 有限，完成反向与 optimizer step，权重 checkpoint 可保存并重新加载训练 |
| 10 | 正式训练 | 记录实际显存、吞吐、配置、数据与权重版本，并监控磁盘 |
| 11 | 可选：安装 LIBERO 仿真器并评测 | 动作成功率评测独立于训练集 loss |

执行 agent 应按阶段记录到 `deployment/STATUS.md`：完成项、命令、退出码、产物路径、实际大小、未解决问题。下载可断点续传；已存在的缓存须验证与当前数据索引一致，不能仅凭目录存在判定完成。不要通过关闭 geometry、CAIR 或更换模型来绕过错误。

## 2. 硬件、磁盘与下载预算

建议 Linux Ubuntu 22.04/24.04 x86_64、8×B200（具体单卡容量以 `nvidia-smi` 为准）、至少 256 GB 主机内存，推荐 512 GB 或更多；32–64 个可用 CPU 核，本地 NVMe。多进程加载基础权重和保存模型权重会产生较高的主机内存峰值。当前默认不保存 DeepSpeed/Accelerate state。

**LIBERO 建议先分配 2 TB 可用 NVMe**。若长期保存全部权重 checkpoint，或运行多个实验，应分配更多；下面的缓存预算不能代替 checkpoint 预算。环境、临时文件、原始下载、缓存和实验输出均需落到大盘。

本节 GB = 10^9 bytes，GiB = 2^30 bytes。下载数据、缓存和模型权重是不同项目，不要重复计算共享文件。

### 2.1 必须下载的内容

| 内容 | 来源 / 文件 | 大小（GB） | 用途 |
|---|---|---:|---|
| LIBERO Spatial | `yuanty/LIBERO-fastwam` / `libero_spatial_no_noops_lerobot.tar.gz` | **0.969**，元数据 | RGB / 动作 / 状态 / 指令 |
| LIBERO Object | 同仓库 / `libero_object_no_noops_lerobot.tar.gz` | **1.353**，元数据 | 同上 |
| LIBERO Goal | 同仓库 / `libero_goal_no_noops_lerobot.tar.gz` | **0.837**，元数据 | 同上 |
| LIBERO Long / 10 | 同仓库 / `libero_10_no_noops_lerobot.tar.gz` | **1.534**，元数据 | 同上 |
| 四包合计 | 上述四个文件 | **4.692** | 保留下载包另外占用这部分空间 |
| 解压后的四套数据 | 从上述包解压 | **按 6–12 预算，未实测** | 以服务器 `du` 为准；不是另一次下载 |
| Wan2.2 TI2V 5B DiT | `Wan-AI/Wan2.2-TI2V-5B` 三个 `diffusion_pytorch_model-*.safetensors` | **19.999**，元数据 | 主 VAE expert、两种 backbone 初始化来源 |
| Wan2.2 VAE | 同仓库 / `Wan2.2_VAE.pth` | **2.819**，元数据 | RGB latent 编码/解码 |
| UMT5-XXL encoder | 同仓库 / `models_t5_umt5-xxl-enc-bf16.pth` | **11.362**，元数据 | 离线文本 embedding；正式训练默认不加载 T5 |
| UMT5 tokenizer | `Wan-AI/Wan2.1-T2V-1.3B` 的 `google/umt5-xxl/*` | **0.0215**，元数据 | 只下载 tokenizer，不下载 Wan2.1 视频模型 |
| DINOv3 ViT-S/16 | `timm/vit_small_patch16_dinov3.lvd1689m` / `model.safetensors` | **0.0864**，元数据 | 384 通道未来特征与历史特征 |
| Qwen3-VL-4B-Instruct | `Qwen/Qwen3-VL-4B-Instruct` 的两个权重分片及 processor/tokenizer/config | **约 8.888**，元数据 | CAIR 当前语义条件；冻结但仍在线前向 |
| Python/CUDA 环境 | PyTorch cu128、CUDA Toolkit、项目依赖 | **安装后约 30–60，预算** | 包下载缓存和安装临时空间另计 |

基础模型文件合计约 **43.18 GB**，加 LIBERO 四包约 **47.87 GB**，不含环境、解压数据、初始化产物及训练缓存。这里 Wan DiT 按实际仓库文件大小计费，不能简单用“5B × 2 bytes”估成 10 GB。

选择离线 depth teacher 时额外下载：`depth-anything/Depth-Anything-V2-Metric-Hypersim-Large` 的 `depth_anything_v2_metric_hypersim_vitl.pth`，**1.341 GB**（元数据），以及 Depth Anything V2 源码。它是本文建议的可选标签来源，**不是 ST-WAM-GEO 已内置的组件，也不代表复现 GeoSem-WAM 的 teacher**。

以下不是从基础权重开始训练的必需下载：已训练 ST-WAM checkpoint、整个 Wan2.1 模型、完整 DiffSynth 模型合集、Mask2Former/SAM、RoboTwin 数据与仿真资产、LIBERO 官方 HDF5 演示。当前训练读取的是上述处理后的 LeRobot 2.1 数据；使用现成仿真深度重放方案时，HDF5/状态等额外输入按该方案单独准备。

### 2.2 本地生成的内容

令 N 为当前四套数据按配置顺序拼接后的帧数 / sample 数（默认 stride=1、无划分）。最终以第 5 节实测为准。下表以 **N=277,713** 举例；该数来自发布仓库对 v3.0 数据的说明，仅作同源数据的容量参考，不能代替对下载的 v2.1 包进行计数。

| 产物 | 估算公式（bytes） | N=277,713 时（GB） |
|---|---|---:|
| ActionDiT / DINO DiT 初始化 | bf16，每个约 1B 量级，按实际 payload 计 | 合计约 4–6 |
| 文本 embedding | 唯一指令数 × 128 × 4096 × 2，另有 mask/序列化 | 通常小于 0.1，预留 1 |
| VAE window_mmap | N × 48 × 3 × 14 × 28 × 2 | **31.35** |
| DINO frame_mmap | N × 384 × 14 × 28 × 2 | **83.61** |
| DINO 中间逐帧 `.pt` | 同上，再加小文件开销 | 额外约 84–90；打包时两份共存 |
| 对齐逐帧深度 bank，float32 | N × 56 × 112 × 4 | **6.97** |
| Geometry 窗口，float32 | N × 9 × 56 × 112 × 4 | **62.71**，另有小文件开销 |
| 可选有效性 mask | bank: N×56×112；windows: N×9×56×112 | 约 **1.74 + 15.68** |
| 模型权重 checkpoint | 四专家 + CAIR adapter / proprio，不含冻结 Qwen 主体 | 约 15–20 / 份，待实测 |
| 完整 DeepSpeed/Accelerate state | 默认 `save_accelerate_state=false`，不生成 | **0（默认不保存）** |

Geometry 每 sample 一个文件，可能产生几十万个小文件；检查 inode。VAE 9 个采样 RGB 帧经时间压缩后是 3 个 latent 帧，不应按 9 帧 latent 估计。DINO 缓存按逐帧保存，未来窗口与 CAIR 历史共享该库，不额外缓存每个 9 帧 DINO 窗口。

默认 LIBERO 为 10 epochs、每 2000 optimizer steps 保存一次权重，训练结束也会保存，代码**没有自动保留最近 K 份 checkpoint 的清理策略**。按示例 N、global batch=128，约 2.17 万 steps，共约 11 份权重；按每份 15–20 GB 估算约 165–220 GB，最终以实测为准。默认不生成 `checkpoints/state/` 或 `trainer_state.json`。旧启动脚本若显式设置 `save_accelerate_state=true`，必须改为 `false`；已运行的训练进程不会因更新源码而自动改变配置。已有 state 不会自动删除。

### 2.3 显存与 batch 起点

先使用 bf16、mixed-attention gradient checkpointing、ZeRO-1、每卡 microbatch=1、梯度累积=16：`8 × 1 × 16 = 128`。B200 的八卡显存并不会自动组成一个可任意使用的大显存池；ZeRO-1 主要分片 optimizer 状态，参数和梯度仍有复制。

这个设置可先按 **单卡约 60–120 GB 的预算区间**规划，但这不是测量结果或上限。Attention kernel 对自定义 mask 的支持、初始化、验证视频生成和权重保存会影响峰值；若回退到数学 attention，可能超过该范围。先实测，再试 `batch_size=2 gradient_accumulation_steps=8` 或 `4/4`，保持 global batch=128。不要因 B200 支持 FP8 就改成 FP8；本代码没有配置相应训练流程。

## 3. 代码、目录和系统准备

将本机整个 `ST-WAM-GEO/` 复制到服务器的 `$STWAM_ROOT/code/ST-WAM-GEO/`。不要用最新上游 FastWAM 覆盖这个副本；上游已新增的数据格式与配置不等于当前版本。

```bash
export STWAM_ROOT=/data/stwam-geo
export STWAM_REPO="$STWAM_ROOT/code/ST-WAM-GEO"
mkdir -p "$STWAM_ROOT"/{dataset,models,runs,logs,cache,tmp}
cd "$STWAM_REPO"
mkdir -p deployment checkpoints
# 新部署副本没有 data 目录时执行；已有目录时先检查其内容和位置。
if [ ! -e data ] && [ ! -L data ]; then
  ln -s "$STWAM_ROOT/dataset" data
fi
readlink -f data
```

预期布局：

```text
/data/stwam-geo/
  code/ST-WAM-GEO/                  # 当前源码，pip editable 安装位置
    data -> /data/stwam-geo/dataset
    checkpoints/                   # 本地生成的 Action/DINO 初始化文件
    deployment/                    # 环境记录、manifest、日志索引
  dataset/
    downloads/libero/              # 四个原始压缩包
    libero_mujoco3.3.2/             # 四套 LeRobot 2.1 目录
    text_embeds_cache/libero/
    vae_latents_cache/...
    dino_latents_cache/...
    geometry_frames/libero/
    geometry_cache/libero/train/
  models/{组织名}/{仓库名}/
  runs/  logs/  cache/  tmp/
```

检查：

```bash
uname -a
uname -m
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi topo -m
df -h "$STWAM_ROOT"
df -i "$STWAM_ROOT"
free -h
```

驱动采用服务器供应商支持 B200 / CUDA 12.8 的版本（R570 或更新的适配分支），由管理员维护驱动及匹配的 Fabric Manager。不要直接在共享主机上覆盖驱动。`nvidia-smi` 显示的 CUDA 版本不是已安装 Toolkit 版本；DeepSpeed JIT 编译还需要 `nvcc` 和 C++ 编译器。容器中需暴露 8 卡并配置充足 `/dev/shm` / memlock。

Ubuntu 系统依赖（由有管理权限的执行者安装）：

```bash
sudo apt-get update
sudo apt-get install -y build-essential git git-lfs curl ca-certificates \
  ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libaio-dev pkg-config libgl1 libglib2.0-0 tmux
git lfs install
```

## 4. Python / CUDA 环境：不要照旧依赖表直接安装

当前 `pyproject.toml` 有三项部署冲突：

1. `transformers==4.49.0` 不含 Qwen3-VL，`trust_remote_code=true` 不解决内置架构缺失。
2. 升级 Transformers 后，原 `huggingface-hub==0.29.2` 不满足版本要求。
3. `torchcodec==0.5` 与固定的 Torch 2.7.1 组合不应直接沿用；官方兼容表列出 TorchCodec 0.3/0.4 对应 Torch 2.7。

建议部署组合：**Python 3.10、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、Transformers 4.57.1、huggingface-hub 0.36.0、TorchCodec 0.4.0、DeepSpeed 0.18.5、Accelerate 1.12.0**。其余直接依赖沿用当前项目；这是待服务器验收的部署基线，不声称已完成 B200 端到端验证。

若服务器没有 conda，先从 [Miniforge 官方发布页](https://github.com/conda-forge/miniforge/releases) 下载匹配 `uname -m` 的 Linux 安装器、核验官方 SHA256，再安装到 `$STWAM_ROOT/miniforge3` 并加载其 `etc/profile.d/conda.sh`。已有 conda 可直接使用。

```bash
conda create -n stwam-geo-b200 python=3.10 -y
conda activate stwam-geo-b200
export TMPDIR="$STWAM_ROOT/tmp"
export PIP_CACHE_DIR="$STWAM_ROOT/cache/pip"
python -m pip install --upgrade pip setuptools wheel

# 若系统已有可用的 CUDA 12.8 Toolkit，则直接设置 CUDA_HOME 指向它，跳过此项。
conda install -y -c nvidia cuda-toolkit=12.8
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version

python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

**只在服务器部署副本中**修正依赖元数据，保留原件；不改网络结构或训练逻辑。本文在编写端只新增了本 MD，没有修改原代码：

```bash
cd "$STWAM_REPO"
python - <<'PY'
from pathlib import Path
p = Path('pyproject.toml')
backup = Path('deployment/pyproject.original.toml')
if not backup.exists():
    backup.write_bytes(p.read_bytes())
s = p.read_text()
for old, new in {
    'transformers==4.49.0': 'transformers==4.57.1',
    'huggingface-hub==0.29.2': 'huggingface-hub==0.36.0',
    'torchcodec==0.5': 'torchcodec==0.4.0',
}.items():
    assert old in s or new in s, (old, 'unexpected dependency metadata')
    s = s.replace('"' + old + '"', '"' + new + '"')
p.write_text(s)
PY
DS_BUILD_OPS=0 python -m pip install -e . \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install ninja sentencepiece ftfy pytest
python -m pip check
python -m pip freeze > deployment/pip-freeze.txt
conda list --explicit > deployment/conda-explicit.txt
ds_report > deployment/deepspeed-report.txt
```

`DS_BUILD_OPS=0` 仅关闭安装时预编译，首次 DeepSpeed 训练仍可能 JIT 编译。当前 MoT 使用 PyTorch `scaled_dot_product_attention`，无需先安装外部 `flash-attn` / xformers；函数名 `flash_attention` 不表示一定使用某个第三方 kernel。本地 DINO 加载器可直接转换 timm 格式权重，也不要求额外安装 timm。

创建以后每个终端都要加载的 Bash 环境文件（环境名已激活）：

```bash
cat > deployment/env.sh <<'SH'
export STWAM_ROOT=/data/stwam-geo
export STWAM_REPO="$STWAM_ROOT/code/ST-WAM-GEO"
export HF_HOME="$STWAM_ROOT/cache/huggingface"
export PIP_CACHE_DIR="$STWAM_ROOT/cache/pip"
export TORCH_EXTENSIONS_DIR="$STWAM_ROOT/cache/torch_extensions"
export TMPDIR="$STWAM_ROOT/tmp"
export DIFFSYNTH_MODEL_BASE_PATH="$STWAM_ROOT/models"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CUDA_HOME="$CONDA_PREFIX"  # 系统 Toolkit 安装时改为实际路径
export PATH="$CUDA_HOME/bin:$PATH"
export STWAM_BASE_TASK=libero_wan5b_dino_s_aux_mot_short_qwen3vl_hist4_vae_mmap_2cam_224_1e-4
STWAM_COMMON=(
  model.redirect_common_files=false
  "model.dino_config.model_path=$STWAM_ROOT/models/timm/vit_small_patch16_dinov3.lvd1689m/model.safetensors"
  "model.semantic_history_config.vlm_model_name_or_path=$STWAM_ROOT/models/Qwen/Qwen3-VL-4B-Instruct"
  ++model.semantic_history_config.attn_implementation=sdpa
)
SH
source deployment/env.sh
```

本文统一选择 **Wan 官方 `.pth` VAE/T5 文件**，并设置 `model.redirect_common_files=false`。源码默认 `true` 会改去下载 DiffSynth 转换版，不能下载一种却仍按另一种路径加载。保留 `model.model_id=Wan-AI/Wan2.2-TI2V-5B`；本地根路径由 `DIFFSYNTH_MODEL_BASE_PATH` 控制，不要把绝对路径误填成 Hub repo ID。

### 4.1 GPU、依赖和 NCCL 验收

```bash
python - <<'PY'
import torch, fastwam, transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from torchcodec.decoders import VideoDecoder
print('fastwam:', fastwam.__file__)
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('transformers:', transformers.__version__, 'arch:', torch.cuda.get_arch_list())
assert torch.cuda.device_count() == 8
for i in range(8):
    with torch.cuda.device(i):
        assert torch.cuda.is_bf16_supported()
        a = torch.randn(256, 256, device=f'cuda:{i}', dtype=torch.bfloat16)
        assert torch.isfinite(a @ a).all()
        torch.cuda.synchronize()
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY

cat > deployment/check_nccl.py <<'PY'
import os
from datetime import timedelta
import torch
import torch.distributed as dist
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
dist.init_process_group('nccl', timeout=timedelta(seconds=120))
x = torch.ones(1, device=f'cuda:{local_rank}')
dist.all_reduce(x)
assert x.item() == dist.get_world_size()
print('NCCL OK rank', dist.get_rank(), flush=True)
dist.destroy_process_group()
PY
torchrun --standalone --nproc_per_node=8 deployment/check_nccl.py
```

B200 的计算能力应为 10.0；用实际 CUDA 计算验收，不能只看 `torch.cuda.is_available()`。若 NCCL 失败，先检查 GPU 互联、驱动/Fabric Manager、容器权限和共享内存，不要直接默认禁用 P2P/NVLink。

## 5. 下载：固定 revision、记录大小、保持目录一致

默认使用 Hugging Face 官方源。如果服务器无法连接，可由合作者选择可信镜像，并在**启动 Python 前**设置 `HF_ENDPOINT`；记录使用的 endpoint，认证令牌只放在环境或 Hugging Face 登录配置中，不写入手册/日志。

下面脚本先生成下载计划和固定 revision 的 manifest，然后按同一个计划下载。可重跑续传，不会下载整个 Wan2.1 或重复的 DINO `.bin`。如需更新版本，另存新的 manifest，不要覆盖用于当前实验的记录。

```bash
cat > deployment/download_assets.py <<'PY'
import fnmatch
import json
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path(os.environ['STWAM_ROOT'])
manifest_path = Path('deployment/download-manifest.json')
jobs = [
    ('model', 'Wan-AI/Wan2.2-TI2V-5B', [
        'diffusion_pytorch_model*.safetensors',
        'diffusion_pytorch_model.safetensors.index.json',
        'Wan2.2_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth',
        'config.json', 'configuration.json', 'README.md'],
        root / 'models/Wan-AI/Wan2.2-TI2V-5B'),
    ('model', 'Wan-AI/Wan2.1-T2V-1.3B', ['google/umt5-xxl/*'],
        root / 'models/Wan-AI/Wan2.1-T2V-1.3B'),
    ('model', 'timm/vit_small_patch16_dinov3.lvd1689m',
        ['model.safetensors', 'config.json', 'README.md', 'LICENSE*'],
        root / 'models/timm/vit_small_patch16_dinov3.lvd1689m'),
    ('model', 'Qwen/Qwen3-VL-4B-Instruct',
        ['*.safetensors', '*.json', '*.txt', '*.md'],
        root / 'models/Qwen/Qwen3-VL-4B-Instruct'),
    ('dataset', 'yuanty/LIBERO-fastwam', [
        'libero_spatial_no_noops_lerobot.tar.gz',
        'libero_object_no_noops_lerobot.tar.gz',
        'libero_goal_no_noops_lerobot.tar.gz',
        'libero_10_no_noops_lerobot.tar.gz'],
        root / 'dataset/downloads/libero'),
]
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
else:
    api = HfApi()
    manifest = {'endpoint': api.endpoint, 'jobs': []}
    for kind, repo, patterns, dest in jobs:
        info = api.repo_info(repo, repo_type=kind, files_metadata=True)
        selected = [f for f in info.siblings
                    if any(fnmatch.fnmatch(f.rfilename, pat) for pat in patterns)]
        assert selected, repo
        files = [{'path': f.rfilename, 'bytes': f.size} for f in selected]
        assert all(f['bytes'] is not None for f in files), repo
        manifest['jobs'].append(dict(repo_type=kind, repo_id=repo,
            revision=info.sha, local_dir=str(dest), files=files))
    manifest_path.write_text(json.dumps(manifest, indent=2))
for job in manifest['jobs']:
    total = sum(f['bytes'] for f in job['files'])
    print(job['repo_id'], job['revision'], f'{total / 1e9:.3f} GB', flush=True)
if '--download' not in sys.argv:
    print('Plan written. Run again with --download to fetch exactly these revisions.')
    raise SystemExit(0)
for job in manifest['jobs']:
    snapshot_download(repo_id=job['repo_id'], repo_type=job['repo_type'],
        revision=job['revision'], local_dir=job['local_dir'],
        allow_patterns=[f['path'] for f in job['files']], max_workers=4)
    for f in job['files']:
        p = Path(job['local_dir']) / f['path']
        assert p.is_file() and p.stat().st_size == f['bytes'], str(p)
print('All selected downloads have the expected sizes.')
PY
python deployment/download_assets.py
python deployment/download_assets.py --download
```

下载后可额外为所选文件生成 SHA256 清单，转移到其他节点后校验；Hub revision 和文件大小清单至少应保留。

解压到固定目录；目标目录已有数据时先检查完整性，避免混合两套版本：

```bash
mkdir -p "$STWAM_ROOT/dataset/libero_mujoco3.3.2"
for suite in spatial object goal 10; do
  archive="$STWAM_ROOT/dataset/downloads/libero/libero_${suite}_no_noops_lerobot.tar.gz"
  gzip -t "$archive" || exit 1
  tar -xzf "$archive" -C "$STWAM_ROOT/dataset/libero_mujoco3.3.2" || exit 1
done

python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['STWAM_ROOT']) / 'dataset/libero_mujoco3.3.2'
N = 0
for suite in ['spatial', 'object', 'goal', '10']:
    d = root / f'libero_{suite}_no_noops_lerobot'
    info = json.loads((d / 'meta/info.json').read_text())
    assert (d / 'data').is_dir() and (d / 'videos').is_dir()
    assert (d / 'meta/episodes.jsonl').is_file()
    assert (d / 'meta/tasks.jsonl').is_file()
    assert str(info['codebase_version']).startswith('v2.'), info['codebase_version']
    print(suite, info['codebase_version'], info['total_frames'], info['total_episodes'])
    N += info['total_frames']
print('N =', N)
for name, bytes_per_sample in {
    'VAE mmap': 48*3*14*28*2,
    'DINO mmap': 384*14*28*2,
    'depth bank fp32': 56*112*4,
    'geometry windows fp32': 9*56*112*4,
    'geometry window masks': 9*56*112,
}.items():
    print(name, f'{N * bytes_per_sample / 1e9:.3f} GB')
PY
du -sh "$STWAM_ROOT/dataset/downloads/libero" "$STWAM_ROOT/dataset/libero_mujoco3.3.2"
```

**顺序固定为 spatial → object → goal → 10**，与 `configs/data/libero_2cam.yaml` 一致。不要下载仓库中的 `lerobot_v30/` 代替这些压缩包；当前副本仍按 LeRobot 2.x 的 episode / video 组织读取。

## 6. 生成两个初始化文件：使用适配后的展开配置

README 示例把 `fastwam_wan5b_dino_s_aux_mot_short_qwen3vl_hist4.yaml` 直接传给预处理脚本，但两份脚本都要求顶层 `video_dit_config`；该模型 YAML 使用的是 `vae_video_dit_config` / `dino_video_dit_config`。下面从真实训练配置展开生成两个仅供预处理的 YAML，避免字段不匹配，也不会改变原模型配置。

```bash
python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base='1.3'):
    cfg = compose(config_name='train', overrides=['task=libero_geometry_mot',
                                                  'model.redirect_common_files=false'])
m = OmegaConf.to_container(cfg.model, resolve=True)
for name, source in [('action', 'vae_video_dit_config'), ('dino', 'dino_video_dit_config')]:
    out = dict(m)
    out['video_dit_config'] = m[source]
    OmegaConf.save(OmegaConf.create(out), f'deployment/preprocess_{name}.yaml')
PY
CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_action_dit_backbone.py \
  --model-config deployment/preprocess_action.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_dino_video_dit_backbone.py \
  --model-config deployment/preprocess_dino.yaml \
  --output checkpoints/DinoVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

Action 脚本当前会一并加载 VAE/T5/tokenizer，因此第 5 节需先下载齐全。两个产物按训练配置应为 hidden_dim=1024、ffn_dim=4096、30 层、24 heads、head_dim=128；DINO 输入通道为 384。无需生成 small Wan VAE-video DiT：本任务保留原生 Wan5B 主 expert。Geometry expert 随机初始化，无需 geometry 初始化权重。

## 7. 生成文本 / VAE / DINO 缓存

所有缓存按 **同一份四套数据、相同目录顺序和 split** 生成。使用 `STWAM_BASE_TASK`，让预计算阶段不要求尚未存在的 geometry 标签。不要为腾空间改 stride 或过滤 episode 后继续使用旧缓存。

### 7.1 文本（T5）

LIBERO 指令数不多，单 GPU 足够。该脚本只构建 T5，不会同时构建四 expert 和 Qwen：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_text_embeds.py \
  "task=$STWAM_BASE_TASK" "${STWAM_COMMON[@]}" overwrite=false
```

正式训练 `load_text_encoder=false` 依赖这些缓存；Qwen 不是 T5 的替代。相同指令的文本缓存可以复用，不能用不同 prompt 模板生成的文件替代。

### 7.2 VAE window_mmap

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_vae_latents.py \
  "task=$STWAM_BASE_TASK" "${STWAM_COMMON[@]}" \
  vae_latent_cache_mode=window_mmap vae_latent_cache_dtype=bf16 \
  vae_precompute_batch_size=2 vae_precompute_num_workers=2 overwrite=false
```

产物目录为 `data/vae_latents_cache/libero_wan22vae38_2cam224_window_mmap`，应有 `.complete`、`metadata.json` 和 `latents.bf16.bin`。每 sample `[48,3,14,28]`。当前脚本对**不完整 mmap 不支持逐样本断点续算**，中断后会要求重建；确认目录只属于本次未完成任务后才用 `overwrite=true`。

### 7.3 DINO：逐帧生成，再打包 mmap

这个脚本只支持输出 `window` 或 `frame`，**不直接输出 frame_mmap**。顶层 `dino_latent_cache_mode` 默认是 window，会覆盖数据配置；必须显式选 frame。数据加载配置中的 VAE/历史缓存依赖也需关掉，确保从 RGB 逐帧编码：

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_dino_latents.py \
  "task=$STWAM_BASE_TASK" "${STWAM_COMMON[@]}" \
  dino_latent_cache_dir=./data/dino_latents_cache/libero_dino_s_2cam224_frame_raw \
  dino_latent_cache_mode=frame dino_latent_cache_dtype=bf16 \
  dino_precompute_batch_size=16 dino_precompute_num_workers=2 \
  data.train.vae_latent_cache_dir=null data.train.vae_latent_cache_required=false \
  data.train.skip_video_load_if_latent_cached=false \
  data.train.load_history_dino_latents=false \
  data.train.history_dino_latent_cache_required=false overwrite=false

python scripts/convert_dino_frame_cache_to_mmap.py \
  --src ./data/dino_latents_cache/libero_dino_s_2cam224_frame_raw \
  --dst ./data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact_mmap \
  --read-workers 8 --read-batch-size 256
```

打包完成后逐帧 shape 应为 `[384,14,28]`，训练窗口 `[384,9,14,28]`，CAIR 历史 `[384,4,14,28]`。检验通过后可按磁盘保留策略归档中间 `.pt`；不能提前删除打包输入。Qwen 仍需读取当前 RGB，不能因为 VAE/DINO 已缓存而删除原 RGB 视频。

## 8. Geometry 深度：这是当前仍需准备的数据环节

**仓库未提供逐帧 depth teacher 推理程序，也未附带可训练深度标签。**`scripts/prepare_geometry_cache.py` 仅将已有深度打包并校验。下载上面的 LIBERO RGB 演示和模型权重，不会自动得到 geometry 监督。

两个可选来源：

| 来源 | 额外准备 | 适用性 / 约束 |
|---|---|---|
| 与演示逐帧对齐的仿真 metric depth | 原始仿真状态、相机设置、匹配 MuJoCo 版本、深度渲染/重放程序 | 已有精确状态时优先；仅有 LeRobot RGB/动作包不能假定能无误恢复相同帧 |
| 离线单目 depth teacher | teacher 源码和权重、逐帧提取适配脚本 | 可从现有 RGB 制作伪标签；metric prediction 也可能有域差异与误差 |

没有精确仿真状态时，建议首轮尝试 **Depth Anything V2 Metric Hypersim Large**，使用官方 `metric_depth` 实现、`vitl` 配置及 teacher `max_depth=20`。teacher 的 20 米模型参数与训练标签 `clip(depth,0,10)/10` 是不同层次，不能为了训练 max_depth=10 而把 teacher 模型构造参数也随意改为 10。

下载示例（选用 teacher 路线才执行；在下载 manifest 中补记 revision 和 SHA256）：

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git \
  "$STWAM_ROOT/code/Depth-Anything-V2"
git -C "$STWAM_ROOT/code/Depth-Anything-V2" rev-parse HEAD \
  > deployment/depth-teacher-git-revision.txt
python - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo = 'depth-anything/Depth-Anything-V2-Metric-Hypersim-Large'
sha = HfApi().model_info(repo).sha
dest = Path(os.environ['STWAM_ROOT']) / 'models' / repo
snapshot_download(repo, revision=sha, local_dir=dest,
    allow_patterns=['depth_anything_v2_metric_hypersim_vitl.pth', 'README.md'])
Path('deployment/depth-teacher-download.json').write_text(
    json.dumps({'repo_id': repo, 'revision': sha, 'local_dir': str(dest)}, indent=2))
PY
```

Teacher 建议使用独立环境，或先审查其 requirements；不要让未锁版本的 OpenCV / NumPy 升级破坏训练环境。仅推理依赖可在独立环境安装；若使用 OpenCV，明确 RGB↔BGR 转换。官方 `run.py --save-numpy` 可处理单图，但不能自动完成本项目的 LeRobot 全局索引映射。

### 8.1 交给执行 agent 的提取脚本规格

若没有现成对齐深度，执行 agent 需要新增一个独立的离线数据准备脚本并保存代码/运行记录。**这一步是待实现的数据工具，不应把本文未提供的脚本当成已有文件执行。**训练网络不需要因提取标签而修改。

要求如下：

1. 按 `configs/data/libero_2cam.yaml` 的四目录顺序及 LeRobot 帧索引遍历；每个全局帧保留 suite、episode、帧号 / timestamp 映射。不要跨 episode 把索引拼错。
2. 解码同一时刻的 `observation.images.image` 和 `observation.images.wrist_image`，**每台相机分别送入 teacher**。不要将两视角拼图当成单相机送入深度模型。
3. 使用与训练 RGB 一致的方向、裁剪和缩放。参考 `scripts/precompute_dino_latents.py` 的 `_FrameVideoDataset._format_frame` 及 `RobotVideoDataset`，不要另加一次翻转。
4. 每相机深度缩放至 `56×56`，左主视角、右腕视角拼成 `56×112`；对应 RGB 为 `224×448`。有无效值时使用 mask-aware resizing，避免无效值污染邻近像素。
5. 生成数值 `.npy`：`depth_frames.npy`，shape `[N,56,112]`，float32，保留米单位；可附 `depth_valid.npy`，同形状 bool。用 `numpy.lib.format.open_memmap` 写 bank，避免把全数据 RGB 或全部深度都放进 RAM。
6. 保存 teacher repo / revision、权重 hash、单位、输入与输出分辨率、数据目录顺序、帧映射、完成标记。可以按 GPU 分片处理，再按确定的全局索引合并；不允许靠文件名词典序猜测帧顺序。
7. 抽查多个 suite / episode 的 RGB-depth 对照图、左右相机接缝、时间连续性、无效比例和深度分位数。当前损失不是尺度不变损失；**不能把普通 relative / inverse depth 输出直接当作米**，也不建议每帧独立 min-max 归一化破坏时间尺度。

若选择已经统一到 `[0,1]` 的非 metric 表示，必须在数据清单中说明含义，并在**打包检查和训练两处**设置 `data.train.geometry_normalize=none`、按约定设置 `geometry_min_depth`（零值有效时需低于 0）。这相当于改变监督目标，需作为实验差异记录，不能混入 metric 标签。

### 8.2 打包与完整性检查

逐帧 bank 确认就绪后执行：

```bash
python scripts/prepare_geometry_cache.py \
  --task libero_geometry_mot \
  --frame-depths "$STWAM_ROOT/dataset/geometry_frames/libero/depth_frames.npy" \
  --frame-valid "$STWAM_ROOT/dataset/geometry_frames/libero/depth_valid.npy" \
  --output "$STWAM_ROOT/dataset/geometry_cache/libero/train"

python scripts/prepare_geometry_cache.py \
  --task libero_geometry_mot --check-only \
  --output "$STWAM_ROOT/dataset/geometry_cache/libero/train"
```

没有单独 mask 文件时删除 `--frame-valid` 参数。该打包器是普通单进程脚本，**不要直接用 torchrun 让 8 个进程写同一批文件**。需要加速时用互不重叠的 `--start / --limit` 范围分段处理，再执行完整检查。

预期每 sample 文件包含：`sample_index`、episode 内截断后的 `frame_indices`、`geometry [9,56,112]`、可选有效 mask。采样偏移为 `[0,4,8,12,16,20,24,28,32]`。第 0 帧和 padding / 无效像素不计入 geometry loss。检查器通过只表示文件格式和有效未来标签基本可用，不证明 RGB-depth 物理对齐正确；可视化抽查仍需做。

## 9. 启动前验证

```bash
python -m pytest -q tests/test_geometry_auxiliary.py tests/test_geometry_mot.py
python scripts/train.py task=libero_geometry_mot "${STWAM_COMMON[@]}" \
  --cfg job --resolve > deployment/libero-resolved.yaml
```

测试使用小模型；若缺少旁边特定旧名称的 baseline 目录，两项“与旧源码比较”的测试可能 skip，要记录原因。测试通过不代表真实 Wan5B/Qwen 或 B200 kernel 已验证。

检查实际数据样本（会计算/读取动作归一化统计）：

```bash
python - <<'PY'
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from fastwam.utils import misc
out = Path('deployment/data-check').resolve()
out.mkdir(parents=True, exist_ok=True)
misc.register_work_dir(str(out))
with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base='1.3'):
    cfg = compose(config_name='train', overrides=['task=libero_geometry_mot'])
ds = instantiate(cfg.data.train)
expected = {'vae_latents': (48,3,14,28), 'dino_latents': (384,9,14,28),
            'history_dino_latents': (384,4,14,28), 'geometry': (1,9,56,112),
            'semantic_image': (3,224,448)}
for i in [0, len(ds)//2, len(ds)-1]:
    x = ds[i]
    for key, shape in expected.items():
        assert tuple(x[key].shape) == shape, (i, key, x[key].shape)
        assert torch.isfinite(x[key]).all(), (i, key)
    assert x['geometry_valid'].shape == x['geometry'].shape
    assert x['context'].shape == (128,4096)
    print('sample', i, 'resolved', x['dataset_idx'], 'shapes OK')
print('dataset length', len(ds))
PY
```

同时检查缓存 metadata 中的 `total_samples`、shape、dtype、video_size 和数据顺序。数据尾部的窗口可能主要是 padding，不要求每个窗口都有非零 geometry loss；但整套数据必须有足够有效的未来监督。

## 10. 单卡与 8 卡短跑、正式训练

所有训练命令均从 `$STWAM_REPO` 运行，激活环境并 `source deployment/env.sh`。以下 `WAIT_FOR_GPUS=0` 假设这些卡已分配给本次任务；脚本默认会等待空闲 GPU，集群调度环境一般使用调度器分配的设备即可。

### 10.1 单卡：验证真实四 expert、Qwen、缓存和反向

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 WAIT_FOR_GPUS=0 bash scripts/train_zero1.sh 1 \
  task=libero_geometry_mot "${STWAM_COMMON[@]}" \
  "output_dir=$STWAM_ROOT/runs/libero_geo_single_smoke" \
  max_steps=2 batch_size=1 gradient_accumulation_steps=1 num_workers=0 \
  log_every=1 save_every=0 eval_every=0 \
  save_accelerate_state=false wandb.enabled=false \
  2>&1 | tee "$STWAM_ROOT/logs/libero_geo_single_smoke.log"
```

单卡 ZeRO-1 不分摊 optimizer 状态，显存可能比 8 卡起点更高；单卡 OOM 不能单独证明 8 卡不能训练。短跑关闭视频验证和完整 state 保存以隔离训练问题，但 trainer 结束时仍会保存权重，需预留磁盘。

### 10.2 8 卡：验证通信、真实 batch 和权重保存

```bash
WAIT_FOR_GPUS=0 bash scripts/train_zero1.sh 8 \
  task=libero_geometry_mot "${STWAM_COMMON[@]}" \
  "output_dir=$STWAM_ROOT/runs/libero_geo_8gpu_smoke" \
  max_steps=3 batch_size=1 gradient_accumulation_steps=16 num_workers=2 \
  log_every=1 save_every=0 eval_every=0 \
  save_accelerate_state=false wandb.enabled=false \
  2>&1 | tee "$STWAM_ROOT/logs/libero_geo_8gpu_smoke.log"
```

至少观察 `loss_video`、`loss_dino`、`loss_action`、`loss_geometry_raw` / `loss_geometry` 为有限值；有效未来深度样本应产生 geometry 梯度。默认权重为 `1.0 / 0.02 / 1.0 / 0.05`。记录 8 个 rank、global batch=128、bf16、ZeRO stage、step 时间和进程显存。

另一个终端可采样 GPU 使用量，注意这会生成持续日志，训练结束后停止监控：

```bash
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1 > "$STWAM_ROOT/logs/libero_geo_gpu.csv"
```

`nvidia-smi` 的采样值可能漏掉瞬时峰值；严谨测量应在训练进程中记录 `torch.cuda.max_memory_allocated()` 和 `max_memory_reserved()`。不要只以冻结模型参数大小推断训练显存。

短跑成功后检查 `runs/libero_geo_8gpu_smoke/checkpoints/weights/step_000003.pt`，新输出目录下不应生成 `checkpoints/state/`。用同样模型配置，在新的输出目录设置 `resume=该权重文件`、`max_steps=1` 再运行，确认权重加载后能完成一个新的训练 step；日志应从 step 0 重新计数，并提示 optimizer / scheduler / step 未恢复。这只验证权重重新启动机制；正式训练请使用新输出目录、`resume=null`。

正式启用视频验证前，另做一个短跑设置 `eval_every=1 eval_num_inference_steps=2` 测试验证路径及额外显存；该验证仍不是仿真动作成功率评测。

### 10.3 正式训练起点

```bash
WAIT_FOR_GPUS=0 bash scripts/train_zero1.sh 8 \
  task=libero_geometry_mot "${STWAM_COMMON[@]}" \
  "output_dir=$STWAM_ROOT/runs/libero_geometry_mot_run01" \
  batch_size=1 gradient_accumulation_steps=16 num_workers=4 \
  num_epochs=10 max_steps=null learning_rate=1e-4 \
  save_every=2000 eval_every=200 save_accelerate_state=false \
  mixed_precision=bf16 model.mot_checkpoint_mixed_attn=true \
  resume=null wandb.enabled=false \
  2>&1 | tee "$STWAM_ROOT/logs/libero_geometry_mot_run01.log"
```

`num_workers=4` 表示每 rank 4 个，共 32 个。先观察数据读取是否成为瓶颈；缓存放在高延迟共享存储时，geometry 小文件尤其容易拖慢。显存不足时先保持 microbatch=1，检查 attention backend 和验证峰值，然后试项目已有 `train_zero2.sh`；更换 ZeRO stage 后重新进行短跑，检查显存和权重加载。

恢复规则：

- `resume=/.../checkpoints/weights/step_xxxxxx.pt`：加载已训练权重继续训练；optimizer、scheduler（含 warmup）、step、epoch 和数据进度重新开始。使用新的 `output_dir` 和日志文件；`max_steps` 是本次新增的步数，例如原计划 20000 步、已完成 6000 步，则设置 `max_steps=14000`。学习率不会自动接上旧日志中的值，应根据训练阶段重新设置。
- 默认 `save_accelerate_state=false`，不保存完整 state；旧的兼容 state 目录仍可用于完整恢复，但不会因此开启后续 state 保存。只有显式设置 `save_accelerate_state=true` 才会保存完整状态。
- 已训练三 expert baseline `.pt` 可作为权重初始化，geometry 保持新初始化；三 expert 的 DeepSpeed state 不能用来恢复四 expert optimizer。
- 原始随机 geometry expert 没有预训练文件；不要为让 loss 正常而关闭它。

当前任务没有独立验证集，runtime 会复用训练集。训练日志 `val_loss` 不表示 held-out 泛化结果；最终比较需要 LIBERO 仿真成功率。若重新划分训练/验证集，所有按 sample index 组织的缓存必须按新 split 生成。

## 11. 评测环境是额外阶段

纯离线训练不需要 LIBERO / RoboTwin simulator。若要评估训练结果，准备 [LIBERO 官方源码](https://github.com/Lifelong-Robot-Learning/LIBERO) 的环境及资产，使用与数据一致的 **MuJoCo 3.3.2**，配置 EGL / NVIDIA 图形库；安装 simulator 依赖时检查是否降级了训练环境包。建议单独评测环境，或在复制的训练环境内安装后再次 `pip check`。

评测需保留本次训练生成的 `dataset_stats.json`、权重 checkpoint、训练展开配置，以及 Qwen/DINO/Wan VAE 资源。本版本推理仍用 DINO 和 Qwen/CAIR；depth teacher 和 geometry cache 仅用于训练。可按 `GEOMETRY_MOT_README.md` 使用原三 expert baseline task 加载 geometry checkpoint，省掉 geometry expert 的实例化；不得误选 VAE-history / infer_without_dino 配置。

## 12. 如果训练 RoboTwin，替换哪些项目

不要在 LIBERO 部署中默认下载 RoboTwin 全部数据。选择 RoboTwin 时使用 `task=robotwin_geometry_mot`，RGB mosaic 为 `[384,320]`、geometry 为 `[96,80]`、action/state 14 维。

| 内容 | 大小 / 处理 |
|---|---|
| `yuanty/robotwin2.0-fastwam` 的 `robotwin2.0.tar.gz.part-00` … `part-07` | **79.086 GB（73.655 GiB）**，文件元数据合计 |
| 同仓库 `dataset_stats.json` | 约 0.000089 GB；配置要求位于 `data/robotwin2.0/dataset_stats.json` |
| 解压数据 | 未实测，初步预留 100–200 GB，下载后实际计量 |
| 模型权重 | 复用第 2 节公共基础权重；按 RoboTwin action 配置生成初始化文件并核对 |
| T5 文本缓存 | 按 RoboTwin 指令重新生成 |
| DINO / VAE latent cache | 当前 RoboTwin task 默认在线编码，不要求全量预缓存；DINO 历史也在线编码 |
| Geometry cache | 必需；每 100 万帧，9×96×80 float32 窗口约 **276.48 GB**，mask 另 **69.12 GB**；全数据可能达到 TB 级 |
| 仿真器 / 资产 | 仅评测或重渲染标签时需要；按 RoboTwin 官方文档另行下载 |

下载分片后，使用确定的文件顺序拼接并流式解压，不需要额外保存一个合并后的大压缩包：

```bash
# 此命令只处理已经下载好的分片，不负责下载。
mkdir -p "$STWAM_ROOT/dataset/robotwin2.0"
set -o pipefail
cat "$STWAM_ROOT"/dataset/downloads/robotwin/robotwin2.0.tar.gz.part-* \
  | tar -xzf - -C "$STWAM_ROOT/dataset/robotwin2.0"
```

数据根目录应为 `data/robotwin2.0/robotwin2.0/{data,meta,videos}`。Geometry bank 顺序跟该数据集索引一致，相机布局为上方主相机、下方左右腕相机；分别估深度后按 RGB 的具体缩放布局拼接。不要复用 LIBERO 的缓存或尺度统计。

RoboTwin 的基础预计算任务是 `robotwin_wan5b_dino_s_aux_mot_short_qwen3vl_hist4_3cam_384x320_1e-4`。本地 `robotwin_geometry_mot` 默认 batch 仍为 `8×1×16=128`；论文/README 的 global batch=1024 不会自动生效。如要严格设置 1024，在 8 卡、microbatch=1 下应使用 accumulation=128，并单独核算学习率、训练时间与实验可比性。

## 13. 常见故障对应检查

| 现象 | 优先检查 |
|---|---|
| Qwen3-VL unknown model / class import failed | Transformers 仍是 4.49，或 editable 环境与实际 Python 不一致 |
| Hub 版本冲突 | 服务器副本 pyproject 三项 pin 是否都更新，后续 install 是否又降级 |
| TorchCodec 动态库 / undefined symbol | Torch/TorchCodec 版本、FFmpeg shared libraries；代码可回退 PyAV，但应记录并测试真实帧读取 |
| `no kernel image` / unsupported `sm_100` | 是否误装 cu121/cu124 或旧扩展；使用 cu128 及匹配 Toolkit，旧扩展缓存不要跨环境混用 |
| 仍下载 DiffSynth 或找不到 VAE/T5 | 是否漏掉 `redirect_common_files=false`，以及 `DIFFSYNTH_MODEL_BASE_PATH` 目录层级 |
| “Cannot detect model type” | 是否拿了 diffusers 重排版、量化版、不全分片或错误 VAE；用本清单原始文件 |
| 预处理要求 `video_dit_config` | 没有使用第 6 节生成的两个展开 YAML |
| DINO 历史缺失或 frame_mmap 不支持 | 先以 frame 生成，再转换；不能将 window 缓存伪装成 frame_mmap |
| Geometry 找不到 sample / frame_indices 不同 | 数据顺序、split、episode filter、stride 改变，或提取脚本没有遵守全局索引 |
| Geometry loss 持续为零 | 检查有效未来深度覆盖率、padding、mask、归一化后的有效范围；不能仅观察总 loss |
| 训练开始很慢 | 首次模型加载、dataset stats 计算、扩展编译、缓存 I/O；先看日志和 CPU/NVMe 占用 |
| GPU 卡住或只有部分 rank 继续 | NCCL、某 rank 数据异常、磁盘满、DeepSpeed 保存同步；保留全部 rank 的错误上下文 |
| 磁盘突然满 | 累积的权重 checkpoint、旧实验 state、启动脚本是否仍设 `save_accelerate_state=true`、重复下载、DINO 中间文件、geometry 小文件/inode |

## 14. 交付给模型负责人的结果

- `deployment/STATUS.md`：各阶段是否完成，geometry 标签来源与提取脚本位置，尚未解决的问题。
- GPU/驱动/Toolkit 记录、Python 路径、`pip-freeze.txt`、`conda-explicit.txt`、`pip check` 和 `ds_report`。
- 源码版本/hash、服务器依赖调整 diff、下载 manifest / revision / checksum。
- 数据集实际帧数、目录顺序、原始与解压大小、缓存 metadata、深度尺度与可视化抽查图。
- 单卡/8 卡短跑日志、四项 loss、NCCL 结果、GPU 显存采样、首份权重 checkpoint 大小、权重重新启动测试结果。
- 正式训练的完整命令、展开配置、global batch、输出目录与 checkpoint 保留安排。

做到“依赖导入通过”只能交付环境已安装；做到“所有缓存齐全”才能交付训练输入已准备；真实短跑与权重重新启动通过后才能交付训练部署已验收。

## 参考与核对依据

- 本地：`configs/task/libero_geometry_mot.yaml`、其 baseline task、`configs/model/fastwam_geometry_mot.yaml`、`configs/model/fastwam_wan5b_dino_s_aux_mot_short_qwen3vl_hist4.yaml`。
- 本地：`src/fastwam/models/wan22/helpers/{loader,io}.py`、`semantic_history.py`、`dino_encoder.py`、`scripts/precompute_*.py`、`scripts/prepare_geometry_cache.py`、`src/fastwam/trainer.py`。
- [LIBERO 数据发布](https://huggingface.co/datasets/yuanty/LIBERO-fastwam) / [RoboTwin 处理后数据](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)。数据来源与文件组织同时对照 [FastWAM README](https://github.com/yuantianyuan01/FastWAM#dataset-download)，但本手册不采用其最新代码替换本地版本。
- [Wan2.2 原始文件](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/main)、[UMT5 tokenizer](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main/google/umt5-xxl)、[DINO timm 权重](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m)、[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)。大小核对使用这些仓库的镜像文件元数据；执行时以固定 revision 的 manifest 为准。
- [Transformers 4.57.1 依赖声明](https://github.com/huggingface/transformers/blob/v4.57.1/setup.py)、[TorchCodec 兼容表](https://github.com/pytorch/torchcodec/tree/v0.5.0#installing-torchcodec)。
- [Depth Anything V2 metric 使用说明](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)、[Hypersim Large 权重](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large)。
