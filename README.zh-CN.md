# 多主体视频生成评测框架

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库提供一个本地 benchmark 框架，用于评测开源视频生成模型在“无人工主体参考图”条件下生成多主体视频的能力。

这个框架主要面向 GPU 服务器运行。它不会自动下载或直接导入 HunyuanVideo、Wan、CogVideoX、ContentV 等视频生成模型，而是通过 YAML 配置读取你填写的本地模型命令，然后完成统一 prompt 构建、批量生成调度、抽帧、检测、指标计算和 Markdown 报告生成。

## 框架能做什么

- 从多主体任务 YAML 自动生成统一英文 prompt。
- 通过 `configs/ms_eval_models.yaml` 调用已启用的本地模型命令。
- 支持纯 T2V 模式。
- 支持 pseudo-reference 第一帧加 I2V/TI2V 模式。
- 保存生成 manifest，便于复现实验。
- 对生成视频抽取采样帧。
- 使用 YOLO 做自动目标检测。
- 计算 SPA、SCA、CC、TP、SRA、MC、SFR、MS-VGS 等多主体评测指标。
- 生成 CSV 汇总和 Markdown 报告。

## 需要提前准备什么

视频生成模型需要提前在 GPU 服务器上准备好。

对每个模型，建议先独立完成：

```bash
git clone <model-repository>
cd <model-repository>
# 安装该模型自己的依赖
# 下载模型权重
# 跑通官方单条生成命令
```

建议每个视频生成模型使用独立 conda 环境，因为大视频模型之间的 CUDA、PyTorch、diffusers、transformers 等依赖经常冲突。

本 benchmark 环境只需要安装轻量评测依赖：

```bash
pip install -r requirements-ms-eval.txt
```

如果服务器不能自动下载 YOLO 权重，请提前把 YOLO 权重文件放到服务器，然后修改 `configs/ms_eval_settings.yaml`：

```yaml
detector:
  model_path: /path/to/yolo11x.pt
```

## 配置模型

编辑 `configs/ms_eval_models.yaml`。

将要测试的模型设置为 `enabled: true`，并把占位路径替换成服务器上的真实生成命令。

示例：

```yaml
models:
  - id: hunyuanvideo_1_5
    type: t2v
    enabled: true
    command_template: >
      conda run -n hunyuan python /data/HunyuanVideo/generate.py
      --prompt "{prompt}"
      --output "{output_path}"
      --duration {duration_sec}
      --fps {fps}
      --seed {seed}
```

支持的模板变量：

- `{prompt}`
- `{output_path}`
- `{duration_sec}`
- `{fps}`
- `{seed}`
- `{task_id}`
- `{model_id}`
- `{input_image}`

如果是 I2V 或 TI2V 模型，可以使用 `{input_image}`：

```yaml
models:
  - id: wan2_2_ti2v_5b
    type: ti2v
    enabled: true
    command_template: >
      conda run -n wan python /data/Wan/generate.py
      --prompt "{prompt}"
      --image "{input_image}"
      --output "{output_path}"
      --fps {fps}
      --seed {seed}
```

在 `ti2v` 或 `i2v` 模式下，框架会自动生成 pseudo-reference 第一帧，并通过 `{input_image}` 传给模型命令。这个第一帧不是人工主体参考图，只是用于锚定多主体布局的伪参考。

## 配置任务

任务定义在 `configs/ms_eval_tasks.yaml`。

每个任务包含：

- 主体名称
- 期望数量
- 初始位置
- 动作描述
- 场景
- 相机
- 视频时长
- FPS

第一版主要支持两个主体任务，例如 `dog + car`、`ball + car`、`car + flower`、`person + bicycle`。

## Dry Run 检查命令

正式运行前，先检查会执行什么命令：

```bash
python scripts/ms_generate.py --dry-run --limit 2 --seeds 0
```

这会生成：

```text
outputs/ms_eval/metrics/generation_manifest.jsonl
outputs/ms_eval/metrics/prompts.jsonl
```

重点检查 `generation_manifest.jsonl` 中的：

- `command`
- `prompt`
- `output_path`

如果没有任何命令输出，请检查：

- `configs/ms_eval_models.yaml` 里是否至少有一个模型 `enabled: true`
- `--mode` 是否和模型的 `type` 匹配

## 运行视频生成

```bash
python scripts/ms_generate.py --seeds 0 1 2 --skip-existing
```

生成视频保存到：

```text
outputs/ms_eval/videos/{model_id}/{task_id}_seed{seed}.mp4
```

生成记录保存到：

```text
outputs/ms_eval/metrics/generation_manifest.jsonl
```

常见状态：

- `success`：命令执行成功，且输出视频存在
- `failed`：命令失败，或没有找到输出视频
- `skipped_existing`：使用了 `--skip-existing`，且视频已存在
- `dry_run`：只格式化命令，没有真实执行

## 抽帧

```bash
python scripts/ms_extract_frames.py --sample-every 4
```

含义是每 4 帧抽取 1 帧。

抽帧结果保存到：

```text
outputs/ms_eval/frames/{model_id}/{task_id}_seed{seed}/frame_0000.jpg
```

抽帧记录保存到：

```text
outputs/ms_eval/metrics/frame_manifest.jsonl
```

## 自动评估

```bash
python scripts/ms_evaluate.py
```

这一步会对抽帧结果运行 YOLO 检测，并计算多主体指标。

检测结果保存到：

```text
outputs/ms_eval/detections/{model_id}/{task_id}_seed{seed}.json
```

指标结果保存到：

```text
outputs/ms_eval/metrics/video_metrics.csv
outputs/ms_eval/metrics/model_summary.csv
outputs/ms_eval/metrics/task_summary.csv
outputs/ms_eval/metrics/failure_summary.csv
```

## 生成报告

```bash
python scripts/ms_build_report.py
```

报告保存到：

```text
outputs/ms_eval/reports/ms_eval_report.md
```

## 一键运行完整流程

```bash
python scripts/ms_run_benchmark.py \
  --seeds 0 1 2 \
  --sample-every 4 \
  --skip-existing
```

如果视频已经生成，只想重新抽帧、评估和生成报告：

```bash
python scripts/ms_run_benchmark.py --no-generate --sample-every 4
```

## 输出结果代表什么

建议先看：

```text
outputs/ms_eval/metrics/model_summary.csv
```

主要指标含义：

- `SPA`：Subject Presence Accuracy，目标主体是否同时出现，越高越好。
- `SCA`：Subject Count Accuracy，主体数量是否正确，越高越好。
- `CC`：Class Correctness，主体类别是否检测正确，越高越好。
- `TP`：Temporal Persistence，主体是否持续可见，越高越好。
- `SRA`：Spatial Relation Accuracy，初始空间关系是否正确，例如 left/right、upper/lower、center，越高越好。
- `MC`：Motion Compliance，bbox 运动是否大致符合动作描述，越高越好。
- `SFR`：Subject Fusion Rate，疑似主体融合比例，越低越好。
- `MS-VGS`：Multi-Subject Video Generation Score，综合分，越高越好。

MS-VGS 公式：

```text
MS-VGS =
0.20 * SPA
+ 0.15 * SCA
+ 0.15 * CC
+ 0.15 * TP
+ 0.10 * SRA
+ 0.10 * MC
+ 0.10 * (1 - SFR)
+ 0.05 * VQ
```

当前如果没有 VQ，框架默认使用 `VQ = 0.5`，让不同模型结果可以保持可比。

## 如何分析结果

推荐分析流程：

1. 打开 `model_summary.csv`，按 `ms_vgs` 排序，得到整体模型排名。
2. 查看 `spa_all`、`sca`、`cc`，判断模型是否能稳定生成正确主体。
3. 查看 `sra` 和 `mc`，判断布局和动作控制是否可靠。
4. 查看 `sfr`，高值样本需要人工复核是否发生主体融合。
5. 打开 `task_summary.csv`，分析哪些类别组合最难。
6. 打开 `failure_summary.csv`，查看每个模型主要失败类型。
7. 对 `possible_fusion`、`wrong_motion`、`detection_unreliable` 标记样本做人工复核。

一般建议人工复核 10% 到 20% 的视频，尤其是报告中失败类型明显的样本。

## 当前局限

- YOLO 对所有类别并不可靠。`flower` 目前用 `potted plant` 和 `vase` 近似。
- 主体融合检测只是基于 bbox IoU 的启发式近似。
- 动作符合度基于 bbox 位移，不等价于真实语义运动理解。
- 自动指标适合做初筛，不能完全替代人工复核。
- 真实生成质量仍取决于外部模型仓库和各自环境配置。

## 更多说明

详细框架说明见：

```text
docs/multi_subject_video_eval.md
```

Windows GPU 服务器模型下载、安装命令和服务器版 YAML 模板见：

```text
docs/model_server_setup_windows.zh-CN.md
configs/ms_eval_models.server.yaml
```
