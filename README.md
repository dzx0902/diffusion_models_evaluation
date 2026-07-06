# Multi-Subject Video Generation Evaluation

[English](README.md) | [简体中文](README.zh-CN.md)

This repository provides a local benchmark framework for evaluating open-source video generation models on multi-subject video generation without manual subject reference images.

The framework is intended to run on a GPU server. It does not download or import video generation models directly. Instead, it reads model commands from YAML, runs those commands, extracts frames, performs object detection, computes metrics, and builds a Markdown report.

## What This Framework Does

- Builds unified English prompts from multi-subject task YAML.
- Runs enabled local model commands through `configs/ms_eval_models.yaml`.
- Supports pure T2V and pseudo-reference first-frame I2V/TI2V modes.
- Saves generation manifests for reproducibility.
- Extracts sampled video frames.
- Runs YOLO-based automatic detection.
- Computes multi-subject metrics such as SPA, SCA, CC, TP, SRA, MC, SFR, and MS-VGS.
- Generates CSV summaries and a Markdown report.

## What You Need to Prepare

The video generation models must be prepared before using this benchmark.

For each model, install and test it on the GPU server first:

```bash
git clone <model-repository>
cd <model-repository>
# install model-specific dependencies
# download model weights
# run one official sample generation command
```

Recommended practice: keep each model in its own conda environment because large video models often have incompatible dependencies.

The benchmark environment only needs the lightweight evaluation dependencies:

```bash
pip install -r requirements-ms-eval.txt
```

If the server cannot download YOLO weights automatically, put the YOLO weight file on the server and update `configs/ms_eval_settings.yaml`:

```yaml
detector:
  model_path: /path/to/yolo11x.pt
```

## Configure Models

Edit `configs/ms_eval_models.yaml`.

Set `enabled: true` for the model you want to test and replace the placeholder path with the real server command.

Example:

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

Supported template variables:

- `{prompt}`
- `{output_path}`
- `{duration_sec}`
- `{fps}`
- `{seed}`
- `{task_id}`
- `{model_id}`
- `{input_image}`

For I2V/TI2V models, use `{input_image}`:

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

In `ti2v` or `i2v` mode, the benchmark creates a pseudo-reference first frame automatically and passes it through `{input_image}`.

## Configure Tasks

Tasks are defined in `configs/ms_eval_tasks.yaml`.

Each task describes:

- subject names
- expected counts
- initial positions
- motions
- scene
- camera
- duration
- FPS

The first version focuses on two-subject tasks, for example `dog + car`, `ball + car`, `car + flower`, and `person + bicycle`.

## Dry Run

Always inspect commands before launching real generation:

```bash
python scripts/ms_generate.py --dry-run --limit 2 --seeds 0
```

This writes:

```text
outputs/ms_eval/metrics/generation_manifest.jsonl
outputs/ms_eval/metrics/prompts.jsonl
```

Open `generation_manifest.jsonl` and check the `command`, `prompt`, and `output_path` fields.

If no commands are printed, check that at least one model has `enabled: true` and that `--mode` matches the model type.

## Run Generation

```bash
python scripts/ms_generate.py --seeds 0 1 2 --skip-existing
```

Generated videos are saved to:

```text
outputs/ms_eval/videos/{model_id}/{task_id}_seed{seed}.mp4
```

Generation status is recorded in:

```text
outputs/ms_eval/metrics/generation_manifest.jsonl
```

Common statuses:

- `success`: command completed and output video exists
- `failed`: command failed or output video was not found
- `skipped_existing`: video already exists and `--skip-existing` was used
- `dry_run`: command was formatted but not executed

## Extract Frames

```bash
python scripts/ms_extract_frames.py --sample-every 4
```

Frames are saved to:

```text
outputs/ms_eval/frames/{model_id}/{task_id}_seed{seed}/frame_0000.jpg
```

Frame extraction metadata is saved to:

```text
outputs/ms_eval/metrics/frame_manifest.jsonl
```

## Evaluate

```bash
python scripts/ms_evaluate.py
```

This runs YOLO detection on extracted frames and computes metrics.

Detection results:

```text
outputs/ms_eval/detections/{model_id}/{task_id}_seed{seed}.json
```

Metric outputs:

```text
outputs/ms_eval/metrics/video_metrics.csv
outputs/ms_eval/metrics/model_summary.csv
outputs/ms_eval/metrics/task_summary.csv
outputs/ms_eval/metrics/failure_summary.csv
```

## Build Report

```bash
python scripts/ms_build_report.py
```

Report output:

```text
outputs/ms_eval/reports/ms_eval_report.md
```

## One-command Pipeline

Run the full pipeline:

```bash
python scripts/ms_run_benchmark.py \
  --seeds 0 1 2 \
  --sample-every 4 \
  --skip-existing
```

Evaluate existing videos without generation:

```bash
python scripts/ms_run_benchmark.py --no-generate --sample-every 4
```

## Output Meaning

Start analysis from `outputs/ms_eval/metrics/model_summary.csv`.

Important metrics:

- `SPA`: Subject Presence Accuracy. Higher means required subjects appear together more often.
- `SCA`: Subject Count Accuracy. Higher means the generated number of subjects matches the task.
- `CC`: Class Correctness. Higher means required categories are detected reliably.
- `TP`: Temporal Persistence. Higher means subjects remain visible over time.
- `SRA`: Spatial Relation Accuracy. Higher means initial left/right/upper/lower/center layout is better.
- `MC`: Motion Compliance. Higher means bbox motion roughly matches the requested action.
- `SFR`: Subject Fusion Rate. Lower is better. This is a heuristic for possible subject overlap or fusion.
- `MS-VGS`: Multi-Subject Video Generation Score. Higher is better.

MS-VGS formula:

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

When VQ is not available, the framework uses `VQ = 0.5`.

## How to Analyze Results

Recommended workflow:

1. Sort `model_summary.csv` by `ms_vgs` to get the overall model ranking.
2. Check `spa_all`, `sca`, and `cc` to see whether the model can generate the right subjects.
3. Check `sra` and `mc` to see whether layout and motion are controlled well.
4. Check `sfr`; high values need manual review for possible subject fusion.
5. Use `task_summary.csv` to find difficult category combinations.
6. Use `failure_summary.csv` to understand each model's main failure modes.
7. Manually review 10% to 20% of videos, especially cases marked `possible_fusion`, `wrong_motion`, or `detection_unreliable`.

## Known Limitations

- YOLO is not reliable for all classes. `flower` is approximated with `potted plant` and `vase`.
- Fusion detection is only an IoU-based heuristic.
- Motion compliance is based on bounding-box displacement, not semantic motion understanding.
- Automatic metrics should be treated as a screening tool, not a replacement for manual review.
- Model-specific generation quality still depends on each external model repository and its environment.

## More Details

See `docs/multi_subject_video_eval.md` for the detailed framework description.

For GPU server model download commands and a server-oriented YAML template, see:

```text
docs/model_server_setup.zh-CN.md
configs/ms_eval_models.server.yaml
```
