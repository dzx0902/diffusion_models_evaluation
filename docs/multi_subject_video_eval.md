# Multi-Subject Video Evaluation

Language: [English README](../README.md) | [简体中文 README](../README.zh-CN.md)

This framework benchmarks local video generation models on multi-subject prompts without requiring subject reference images.
It is designed to run on a GPU server. The local repository only provides the benchmark orchestration, metrics, and reporting layer.

## Purpose

- Compare how well different open video models handle multiple main subjects in one scene.
- Support two first-stage modes:
  - pure T2V
  - pseudo-reference first-frame plus I2V/TI2V
- Keep the model invocation layer external so each local model can be plugged in through YAML.

## Model Configuration

Edit `configs/ms_eval_models.yaml` and set `enabled: true` for the models you want to run.

Each entry uses a `command_template` and receives:

- `{prompt}`
- `{output_path}`
- `{duration_sec}`
- `{fps}`
- `{seed}`
- `{task_id}`
- `{model_id}`
- `{input_image}`

## Task Configuration

Edit `configs/ms_eval_tasks.yaml` to add more two-subject or later three-subject cases.
The first release focuses on two subjects per task.

## Dry Run

Use dry run to inspect commands before running any generation job:

```bash
python scripts/ms_generate.py --dry-run --limit 2 --seeds 0
```

## Full Benchmark

```bash
python scripts/ms_run_benchmark.py \
  --tasks configs/ms_eval_tasks.yaml \
  --models configs/ms_eval_models.yaml \
  --settings configs/ms_eval_settings.yaml \
  --seeds 0 1 2 \
  --sample-every 4 \
  --skip-existing
```

Pipeline stages:

1. generation
2. frame extraction
3. detection and metrics
4. report build

For server deployment, install the lightweight evaluation dependencies in the benchmark environment:

```bash
pip install -r requirements-ms-eval.txt
```

Model-specific dependencies should stay in each model's own environment or launch script. The benchmark calls those scripts through `command_template`.

## Metrics

- SPA: Subject Presence Accuracy
- SCA: Subject Count Accuracy
- CC: Class Correctness
- TP: Temporal Persistence
- SRA: Spatial Relation Accuracy
- MC: Motion Compliance
- SFR: Subject Fusion Rate heuristic
- VQ: Visual Quality placeholder

## MS-VGS

The score is computed as:

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

If VQ is not available, the framework uses `VQ = 0.5` as a placeholder so the score remains comparable across runs.

## Current Limitations

- YOLO aliases are weak for categories like `flower`, which are approximated with generic object labels.
- Fusion detection is a heuristic based on repeated bounding-box overlap.
- Motion compliance is based on bounding-box displacement rather than semantic tracking.
- A final 10% to 20% manual review is still recommended.

## Future Improvements

- GroundingDINO or OWL-ViT open-vocabulary detection
- SAM2 mask tracking
- Optical flow based motion analysis
- CLIP, SigLIP, or Video-LLM semantic consistency scoring
- Manual annotation interface

## Local Model Integration

Implement the local model command anywhere on the GPU server, then point `command_template` to it.
The benchmark framework never imports model code directly.
