# ST-WAM-GEO：LIBERO 四数据集联合训练指南（单机 8×B200）

本文给合作者在服务器上直接照做。目标：

1. 使用 LIBERO 四个 suite（spatial / object / goal / 10）联合训练。
2. 每 3000 steps 保存一次 checkpoint，并输出 loss 曲线。
3. 尽量吃满 8×B200（180GB）的算力。

环境安装、权重下载、缓存与 geometry 深度标签的准备见 `DEPLOY_8xB200.md`。本文只讲“准备好之后怎么训练”。

## 0. 前提检查

开始前确认以下各项已按 `DEPLOY_8xB200.md` 完成：

- 已激活 conda 环境（如 `stwam-geo-b200`），`nvidia-smi` 可见 8 张 B200，NCCL 验收通过。
- 四套 LeRobot 数据位于 `data/libero_mujoco3.3.2/`，顺序固定为 spatial → object → goal → 10。`configs/data/libero_2cam.yaml` 默认就把这四个目录拼成一个训练集，所以**联合训练不需要额外设置**，不要改目录顺序，也不要改 stride 或过滤 episode。
- 文本（T5）、VAE（window_mmap）、DINO（frame_mmap）缓存已生成，geometry 窗口缓存已打包并通过检查。
- 单卡和 8 卡短跑均已通过，四项 loss 为有限值。

快速自检：

```bash
cd "$STWAM_REPO"
source deployment/env.sh
ls data/libero_mujoco3.3.2/
python scripts/prepare_geometry_cache.py --task libero_geometry_mot --check-only \
  --output "$STWAM_ROOT/dataset/geometry_cache/libero/train"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

## 1. 环境变量与公共参数

每个新终端都先执行下面几行，并建议在 tmux 中运行训练，避免断开连接中断任务：

```bash
cd "$STWAM_REPO"
source deployment/env.sh        # 提供 STWAM_COMMON 等公共 Hydra 参数
export WAIT_FOR_GPUS=0          # 8 卡已分配给本任务，跳过等待空闲 GPU
mkdir -p "$STWAM_ROOT/logs"
tmux new -s train               # 重新连接用 tmux attach -t train
```

`STWAM_COMMON` 已包含本地 Wan、DINO、Qwen 路径以及 `model.redirect_common_files=false`，后续命令直接引用 `"${STWAM_COMMON[@]}"`。


## 2. 本次训练的关键配置

| 目标 | 参数 | 说明 |
|---|---|---|
| 四数据集联合训练 | `task=libero_geometry_mot` | 数据配置继承 `libero_2cam.yaml`，四个 suite 拼成一个训练集，每个 epoch 均匀打乱遍历全部样本 |
| 每 3000 步保存 | `save_every=3000` | 每次只写权重 `checkpoints/weights/step_xxxxxx.pt`，不保存 state；训练结束时还会保存权重 |
| loss 曲线 | `log_every=10` | 每 10 个 optimizer step 把各项 loss 写入日志，训练后用第 5 节脚本画图 |
| 提高算力利用率 | `batch_size` 与 `gradient_accumulation_steps` | 见第 3 节，保持乘积为 16，使 8 卡 global batch = 128 |
| 数据加载 | `num_workers`、`prefetch_factor`、`persistent_workers=true` | 避免 GPU 等数据 |
| 精度与显存 | `mixed_precision=bf16`、`model.mot_checkpoint_mixed_attn=true` | 沿用默认；checkpointing 保持开启，才能放大 microbatch |

说明：

- global batch = `batch_size × gradient_accumulation_steps × 8`。默认 `1 × 16 × 8 = 128`，与 baseline 一致。**调大 microbatch 时必须同步减小累积步数，保持 128**，否则学习率、总步数和实验可比性都会改变。
- 总 optimizer steps = `ceil(N / 128) × num_epochs`。N 为四套数据总帧数（见 `DEPLOY_8xB200.md` 第 5 节的统计输出），默认 `num_epochs=10`。若想直接指定步数，设 `max_steps=<数值>`，此时它优先于 epoch。
- `save_every=3000` 按 optimizer step 计数，不是 micro step。
- 日志中的 `val_loss` 来自与训练集相同的数据（任务没有独立验证集），**不是泛化指标**，画 loss 曲线时以 `[train]` 行为准。
- `eval_every` 会触发视频生成式验证，既占显存又拖慢训练。正式训练建议设为 `eval_every=0` 关闭；需要时再单独打开。

## 3. 探测每卡 batch，吃满显存

默认每卡 microbatch=1 是保守起点，B200 180GB 通常还有余量。**不要凭估计直接定值，用短跑实测。**

依次尝试 `batch_size = 1 → 2 → 4 → 8`，对应 `gradient_accumulation_steps = 16 → 8 → 4 → 2`（乘积始终为 16）。每组只跑几步，记录峰值显存和 step 速度：

```bash
for cfg in "1 16" "2 8" "4 4" "8 2"; do
  set -- $cfg
  bs=$1; ga=$2
  echo "=== batch_size=$bs grad_accum=$ga ==="
  bash scripts/train_zero1.sh 8 \
    task=libero_geometry_mot "${STWAM_COMMON[@]}" \
    "output_dir=$STWAM_ROOT/runs/probe_bs${bs}" \
    batch_size=$bs gradient_accumulation_steps=$ga \
    num_workers=4 persistent_workers=true \
    max_steps=6 log_every=1 save_every=0 eval_every=0 \
    save_accelerate_state=false wandb.enabled=false \
    2>&1 | tee "$STWAM_ROOT/logs/probe_bs${bs}.log"
done
```

另开一个终端，在探测期间采样显存和利用率：

```bash
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv -l 2 > "$STWAM_ROOT/logs/probe_gpu.csv"
```

判断方法：

1. 看 `probe_bs*.log` 里 `[train]` 行的 `speed=... samples/s`，**samples/s 越高越好**。
2. 看 `probe_gpu.csv` 中单卡 `memory.used` 的峰值，建议留 10–15% 余量（约 150–160 GB 以内），因为保存 checkpoint 和偶发长序列会有瞬时峰值。
3. 出现 OOM 就退回上一档。选 samples/s 最高且不 OOM 的一档作为正式配置。
4. microbatch 放大后若 samples/s 提升不明显，说明瓶颈在数据加载或通信，不在显存，转看第 7 节。

探测前几步包含模型加载和预热，速度读数以最后几步为准。探测结束后停止 `nvidia-smi` 采样，并删除 `runs/probe_*` 里不需要的输出，每组短跑结束时会写一份权重。


## 4. 正式训练命令

先按第 3 节的结果填写 `BS` 和 `GA`，**必须满足 `BS × GA = 16`**（8 卡 global batch = 128）。下面以 `BS=4`、`GA=4` 为例，请替换为实测值。

```bash
cd "$STWAM_REPO"
source deployment/env.sh
export WAIT_FOR_GPUS=0

BS=4          # 按第 3 节探测结果填写
GA=4          # 保证 BS * GA = 16
RUN=libero_geometry_mot_4suites_run01

bash scripts/train_zero1.sh 8 \
  task=libero_geometry_mot "${STWAM_COMMON[@]}" \
  "output_dir=$STWAM_ROOT/runs/$RUN" \
  batch_size=$BS gradient_accumulation_steps=$GA \
  num_workers=6 prefetch_factor=4 persistent_workers=true pin_memory=true \
  num_epochs=10 max_steps=null learning_rate=1e-4 \
  log_every=10 save_every=3000 eval_every=0 \
  save_accelerate_state=false \
  mixed_precision=bf16 model.mot_checkpoint_mixed_attn=true \
  resume=null wandb.enabled=false \
  2>&1 | tee "$STWAM_ROOT/logs/$RUN.log"
```

参数说明：

- `save_every=3000`：每 3000 个 optimizer step 保存一次。产物位于 `$STWAM_ROOT/runs/$RUN/checkpoints/`：
  - `weights/step_003000.pt`、`step_006000.pt` ……：仅权重，用于评测或作为初始化；
  - 不生成 `state/` 或 `trainer_state.json`；已有 state 不会自动删除。
- `eval_every=0`：关闭训练内的视频验证，避免额外显存和耗时（日志里的 `val_loss` 本来也不是泛化指标）。
- `num_workers=6`、`prefetch_factor=4`、`persistent_workers=true`：每个 rank 6 个 worker，共 48 个进程。服务器 CPU 核数不足（建议 ≥ 64）时降到 4；若第 7 节发现 GPU 利用率有周期性空档，再适当增大。
- 总步数：`ceil(N/128) × num_epochs`，N 为四套数据总帧数。想缩短训练可设 `num_epochs` 或直接设 `max_steps`。
- 日志文件 `$STWAM_ROOT/logs/$RUN.log` 同时是第 5 节画 loss 曲线的数据源，**不要删除，也不要用 `>` 覆盖**。
- 训练会持续数十小时以上，务必在 tmux 中运行（第 1 节已创建）。

启动后先观察前几十步：确认出现 `[train] epoch=0 step=10/... loss=...` 行，且 `loss_video`、`loss_dino`、`loss_action`、`loss_geometry_raw` 均为有限值。

## 5. loss 曲线

trainer 只把 loss 写入文本日志（以及可选的 wandb），没有本地 csv 或 tensorboard，所以用脚本解析日志画曲线。可以在训练过程中随时运行，用来查看当前进度。

### 5.1 绘图脚本

新建 `scripts/plot_train_loss.py`：

```bash
cat > scripts/plot_train_loss.py <<'PY'
"""Parse '[train]' lines from a training log and plot loss curves."""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(r"\[train\] epoch=(\d+) step=(\d+)/(\d+) loss=([-\d.eE+naif]+) (.*)")
KV = re.compile(r"([A-Za-z_]+)=([-\d.eE+naif]+)")


def parse(log_path):
    rows = {}
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        step = int(m.group(2))
        row = {"step": step, "epoch": int(m.group(1)), "loss": float(m.group(4))}
        for k, v in KV.findall(m.group(5)):
            try:
                row[k] = float(v)
            except ValueError:
                pass
        rows[step] = row  # 续训时同一 step 以最后一次为准
    return [rows[s] for s in sorted(rows)]


def smooth(values, window):
    if window <= 1:
        return values
    out, acc = [], []
    for v in values:
        acc.append(v)
        if len(acc) > window:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", nargs="+", required=True, help="一个或多个训练日志（续训时按顺序传入）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smooth", type=int, default=20, help="滑动平均窗口（按记录点计）")
    ap.add_argument("--save-every", type=int, default=3000)
    args = ap.parse_args()

    merged = {}
    for p in args.log:
        for r in parse(p):
            merged[r["step"]] = r
    rows = [merged[s] for s in sorted(merged)]
    if not rows:
        raise SystemExit("日志中没有找到 [train] 行，检查路径和 log_every 设置")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    keys = sorted({k for r in rows for k in r} - {"step", "epoch", "lr", "speed", "eta"})
    with open(out / "loss.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch"] + keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    steps = [r["step"] for r in rows]
    ckpt_steps = [s for s in steps if s % args.save_every == 0]

    def draw(ax, key, title):
        ys = [r.get(key, float("nan")) for r in rows]
        ax.plot(steps, ys, alpha=0.25, linewidth=0.8)
        ax.plot(steps, smooth(ys, args.smooth), linewidth=1.5)
        for s in ckpt_steps:
            ax.axvline(s, color="gray", linestyle=":", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)

    draw(plt.figure(figsize=(9, 5)).gca(), "loss", "total loss")
    plt.tight_layout()
    plt.savefig(out / "loss_total.png", dpi=150)
    plt.close("all")

    parts = [k for k in keys if k.startswith("loss_")]
    cols = 2
    nrows = (len(parts) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(12, 4 * nrows), squeeze=False)
    for ax, k in zip(axes.flat, parts):
        draw(ax, k, k)
    for ax in list(axes.flat)[len(parts):]:
        ax.axis("off")
    fig.suptitle("per-term loss (thin: raw, thick: moving average; dotted: checkpoint)")
    fig.tight_layout()
    fig.savefig(out / "loss_terms.png", dpi=150)
    print(f"parsed {len(rows)} points, step {steps[0]}..{steps[-1]}")
    print(f"wrote {out}/loss.csv, loss_total.png, loss_terms.png")


if __name__ == "__main__":
    main()
PY
python -c "import matplotlib" 2>/dev/null || python -m pip install matplotlib
```

### 5.2 使用

```bash
python scripts/plot_train_loss.py \
  --log "$STWAM_ROOT/logs/$RUN.log" \
  --out-dir "$STWAM_ROOT/runs/$RUN/loss_curves" \
  --smooth 20 --save-every 3000
```

输出：

- `loss.csv`：每个记录点的 step 及各项 loss；
- `loss_total.png`：总 loss；
- `loss_terms.png`：各分项（`loss_video`、`loss_dino`、`loss_action`、`loss_geometry` 等）；细线为原始值，粗线为滑动平均，灰色虚线标出 checkpoint 位置。

建议每保存一次 checkpoint 后画一次图，检查曲线是否正常下降。若训练中断后续训，把两段日志按时间顺序一起传给 `--log`，同一 step 以后者为准。

### 5.3 可选：wandb

服务器能联网时，可把 `wandb.enabled=false` 改成 `wandb.enabled=true`，trainer 会自动上报 `train/loss`、`train/loss_*`、`train/grad_norm`、`train/lr` 和吞吐。无法联网时使用 `wandb.mode=offline`，之后用 `wandb sync` 上传。API key 只放在环境变量里，不要写进脚本或文档。


## 6. checkpoint 与磁盘

每到 3000 的整数倍 step，trainer 只写模型权重；训练结束时也会保存。默认 `save_accelerate_state=false`，不写 optimizer、scheduler 或训练进度状态。

| 路径 | 内容 | 用途 | 大小（估计，待首次保存实测） |
|---|---|---|---|
| `checkpoints/weights/step_003000.pt` | 四专家及可训练 adapter / proprio 等权重 | 评测、作为新一轮训练的初始化 | 约 15–20 GB / 份 |

代码**没有自动清理旧 checkpoint 的机制**。以 N≈277,713、global batch=128 估算，10 个 epoch 约 2.17 万步，按每 3000 步保存并计入结束时的保存，共约 8 份权重，约 120–160 GB，最终以实测为准。

1. 第一次保存（step 3000）完成后，实测大小：

   ```bash
   du -sh "$STWAM_ROOT/runs/$RUN/checkpoints/weights"/*
   df -h "$STWAM_ROOT"
   ```

2. 按实测大小和剩余空间制定保留或归档策略。已有实验的 state 不会自动删除；删除任何 checkpoint 前先和模型负责人确认。
3. 仅主 rank 写权重，其他 rank 在同步点等待；保存期间训练会暂停，实际耗时取决于磁盘和权重大小。
4. 保存成功后，日志显示 `[ckpt] step=3000 weights=... state=None`。新输出目录下不应生成 `checkpoints/state/`。
5. 更新源码不会改变已经运行的训练进程；下一次启动时确认没有旧的 `save_accelerate_state=true` 参数覆盖新默认值。

## 7. 监控 GPU 利用率

目标是让 8 张卡持续高负载。训练稳定后（过了前几十步预热），另开终端观察：

```bash
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv -l 5 > "$STWAM_ROOT/logs/${RUN}_gpu.csv"
```

也可以直接用 `watch -n 2 nvidia-smi` 看实时状态。训练结束或不再需要时停止采样，避免日志无限增长。

判断与处理：

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 8 卡 `utilization.gpu` 长期在 90% 以上、显存稳定 | 正常 | 无需调整 |
| 利用率周期性掉到很低 | 数据加载跟不上（geometry 小文件、视频解码、Qwen 当前 RGB 读取） | 增大 `num_workers`、`prefetch_factor`；检查 NVMe 的 I/O（`iostat -x 2`）和 CPU（`top`） |
| 某一张卡明显低于其他卡 | 该 rank 数据读取或通信异常 | 查看全部 rank 的日志，确认 NCCL 与磁盘状态 |
| 显存使用不到 60%，利用率高 | microbatch 偏小 | 回到第 3 节，试更大的 `batch_size`（保持 `batch_size × grad_accum = 16`） |
| 显存接近上限或保存 checkpoint 时 OOM | microbatch 偏大 | 退回上一档；`nvidia-smi` 采样可能漏掉瞬时峰值，留足余量 |

日志里的 `speed=... step/s, ... samples/s` 是最直接的吞吐指标，每次调整参数后对比它。

## 8. 中断恢复与常见问题

### 8.1 从权重重新启动训练

用最近一份 `weights/step_xxxxxx.pt` 初始化新训练，保留相同的模型结构、数据与缓存配置。学到的模型权重会恢复，但 optimizer、scheduler（包括 warmup）、step、epoch 和数据读取位置会重新开始。以下示例假定原计划 20000 步、已完成 6000 步，因此本次再训练 14000 步；按实际计划修改。学习率也需根据中断前日志和训练阶段重新设置。

```bash
OLD_RUN=libero_geometry_mot_4suites_run01
RUN=libero_geometry_mot_4suites_restart01  # 使用新的输出目录和日志
REMAINING_STEPS=14000                   # 本次新增的 optimizer steps
RESTART_LR=5e-5                         # 示例值，按旧日志/训练阶段调整

bash scripts/train_zero1.sh 8 \
  task=libero_geometry_mot "${STWAM_COMMON[@]}" \
  "output_dir=$STWAM_ROOT/runs/$RUN" \
  batch_size=$BS gradient_accumulation_steps=$GA \
  num_workers=6 prefetch_factor=4 persistent_workers=true pin_memory=true \
  max_steps=$REMAINING_STEPS learning_rate=$RESTART_LR \
  log_every=10 save_every=3000 eval_every=0 \
  save_accelerate_state=false \
  mixed_precision=bf16 model.mot_checkpoint_mixed_attn=true \
  "resume=$STWAM_ROOT/runs/$OLD_RUN/checkpoints/weights/step_006000.pt" wandb.enabled=false \
  2>&1 | tee "$STWAM_ROOT/logs/$RUN.log"
```

要点：

- 日志应明确显示加载了指定权重，且 optimizer / scheduler / step 未恢复；新训练从 step 0 重新计数。
- 权重文件中记录的旧 step 不会自动计入新任务，`max_steps` 表示本次还要训练的步数。
- 初期 loss 可能因优化器和学习率重启而波动；检查四项 loss 和梯度是否有限。
- 新旧训练日志分别绘图，不要直接用第 5 节脚本合并相同 step 编号，否则会覆盖旧数据点。
- 已有兼容的完整 state 仍可加载；默认关闭的是后续 state 保存，不会删除旧 state。

### 8.2 常见问题

| 现象 | 优先检查 |
|---|---|
| 启动即 OOM | 降低 `batch_size`（同步提高 `gradient_accumulation_steps`）；确认 `model.mot_checkpoint_mixed_attn=true`；确认没有其他进程占用 GPU |
| 训练中途 OOM | 多发生在保存 checkpoint 或长序列时，microbatch 退一档；用 `nvidia-smi` 检查是否有残留进程 |
| 找不到缓存 / 数据索引不匹配 | 数据目录顺序、stride、split 是否与生成缓存时一致；见 `DEPLOY_8xB200.md` 第 7、8 节 |
| `loss_geometry` 恒为 0 | 检查 geometry 缓存的有效像素覆盖率和归一化设置，不能只看总 loss |
| loss 出现 nan / inf | 立即停止；对比最近一份 `weights`，检查数据与学习率，保留日志，不要继续写 checkpoint |
| 8 卡卡住、部分 rank 不动 | NCCL、磁盘写满、某 rank 数据异常；保留所有 rank 的报错信息 |
| 磁盘写满 | 检查累积权重、旧 state，以及旧命令是否仍开启 state 保存；见第 6 节 |
| 日志中没有 `[train]` 行 | 确认 `log_every` 大于 0，且训练已走过至少一个 optimizer step |

### 8.3 交付物

训练完成（或阶段性完成）后，请向模型负责人提供：

- 完整训练日志 `$STWAM_ROOT/logs/$RUN.log`；
- `loss_curves/` 下的 `loss.csv`、`loss_total.png`、`loss_terms.png`；
- 实际使用的 `BS`、`GA`、`num_workers`、global batch，以及第 3 节的探测记录；
- 各 checkpoint 的路径与大小，GPU 显存和利用率采样文件；
- 训练期间遇到的问题及处理方式。

注意：日志中的 `val_loss` 使用的是训练集，不是泛化指标。要评估模型效果，需要另行准备 LIBERO 仿真环境做成功率评测（见 `DEPLOY_8xB200.md` 第 11 节）。
