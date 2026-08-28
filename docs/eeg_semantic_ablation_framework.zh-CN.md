# EEG 语义消融实验代码框架

## 1. 统一入口

实验矩阵定义在 `configs/eeg_semantic/ablation_matrix.yaml`。先展开配置，不会训练：

```bash
python scripts/run_eeg_semantic_ablation.py --stage materialize
```

训练、预测和生成必须分阶段执行，避免训练一个方法后立即生成大量视频而阻塞其余方法：

```bash
python scripts/run_eeg_semantic_ablation.py --stage train --device cuda --skip-existing
python scripts/run_eeg_semantic_ablation.py --stage predict --device cuda --skip-existing
python scripts/run_eeg_semantic_ablation.py --stage generate \
  --models-root .ms_video_models --skip-existing
```

可以用 `--variants a_base b_base c2_full` 选择部分实验。若发现 `last.pt`，训练入口会自动使用
精确续训；若发现 `best.pt` 且使用 `--skip-existing`，则跳过已完成训练。

## 2. 已实现方法和 trick

- A：coarse multi-head classification → confidence filtering → deterministic template。
- B：structured slots → confidence filtering → deterministic verbalizer。
- C1：EEG → 完整 Tora `[226,4096]` 状态；默认小 batch + 梯度累积。
- C2：EEG → train-only PCA bottleneck → Tora 状态，支持 128/256/512/1024/2048。
- C3：train-only nonlinear text autoencoder → frozen bottleneck → EEG alignment → decoder。
- hard、multi-positive、weighted multi-positive、soft semantic contrastive。
- same-video cross-session consistency、辅助语义分类、train-only text prototype。
- classifier-weight prototype、弱 EEG augmentation、hierarchical curriculum。
- 离线 deterministic paraphrase、caption semantic centroid。
- TensorBoard 和 JSONL 记录、best semantic、best overall、last checkpoint、精确续训。

PCA 只在对应 fold 的 train videos 上拟合，并在 metadata 中记录 train ID digest 和
90%/95%/99% explained-variance 建议。prototype 也只从 train videos 构建。

## 3. Paraphrase、centroid 和 nonlinear bottleneck

构建三个不调用在线 LLM 的等价表达：

```bash
python scripts/build_semantic_paraphrases.py \
  --labels outputs/semantic_labels/eeg_semantic_labels_v1.jsonl \
  --output-dir outputs/tora/paraphrases --overwrite
```

分别使用 `cache_tora_text_states.py` 缓存三个 JSONL，然后构建 centroid：

```bash
python scripts/build_tora_caption_centroids.py \
  --indices outputs/tora/paraphrases/cache1/index.jsonl \
            outputs/tora/paraphrases/cache2/index.jsonl \
            outputs/tora/paraphrases/cache3/index.jsonl \
  --output-dir outputs/tora/centroid/text_cache --overwrite
```

centroid 后仍须对每个 fold 单独拟合 PCA。nonlinear 版本先训练 autoencoder，再导出 frozen targets：

```bash
python scripts/train_tora_text_autoencoder.py \
  --index outputs/tora/text_cache/index.jsonl \
  --split-plan outputs/eeg_wan/splits/chentianlin_video_6fold_plan.json \
  --fold video_6fold_1 --bottleneck-dim 512 \
  --output-dir outputs/tora/autoencoder/fold1/dim512

python scripts/export_tora_autoencoder_targets.py \
  --index outputs/tora/text_cache/index.jsonl \
  --checkpoint outputs/tora/autoencoder/fold1/dim512/best.pt \
  --output-dir outputs/tora/autoencoder/fold1/dim512 --overwrite
```

实际部署时应将 autoencoder checkpoint 与 latent index 分开目录，避免 `--overwrite` 与训练产物混用。

## 4. 多生成模型路由

`configs/eeg_semantic/generators.server.yaml` 包含：

- Tora native caption 和 Tora hidden-state injection；
- AnimateDiff、ZeroScope；
- CogVideoX-2B、CogVideoX1.5-5B；
- Wan2.2-TI2V-5B；
- ContentV-8B；
- HunyuanVideo-1.5。

A/B caption 可以进入全部 caption generator。C1/C2/C3 是 Tora 特定空间，只进入
`tora_injected`，不能把 Tora latent 当成其他模型的原生 text state。

优先从 GT test videos 离线提取按实体区分的多轨迹。检测缺失帧只在同一视频内插值，
不会读取 validation/test 的语义统计：

```bash
conda run -n ms-video-eval python scripts/extract_fixed_tora_trajectories.py \
  --video-manifest data/manifests/video_manifest.jsonl \
  --semantic-labels outputs/semantic_labels/eeg_semantic_labels_v1.jsonl \
  --split-plan outputs/eeg_wan/splits/chentianlin_video_6fold_plan.json \
  --fold video_6fold_1 --partition test \
  --output-dir outputs/tora/trajectories/fold1_test --skip-existing
```

输出采用 Tora 官方 256×256 `x,y` 文本格式。每个 core entity 各一条轨迹；
`coverage.csv` 必须人工检查，`fallback_only=1` 的实体不能静默视为可靠 GT 轨迹。
若已有人工轨迹，可改用 `build_fixed_trajectory_manifest.py` 收集多个 txt 文件。

每个生成记录保存 trajectory SHA256。生成完成后必须运行：

```bash
python scripts/audit_eeg_ablation_generation.py \
  --manifests outputs/eeg_semantic/generated/*/*/*/*/generation_manifest.jsonl \
  --output outputs/eeg_semantic/reports/generation_protocol_audit.json
```

## 5. 评估和统计

语义分类输出逐 trial 与逐 video 指标；连续方法输出 MSE、token cosine 和 retrieval top-k。
生成视频 evaluator 支持：

- CLIP overall semantic、subject、object、action score；
- temporal consistency、motion energy；
- sharpness、exposure diagnostic；
- fixed trajectory direction adherence。

```bash
python scripts/evaluate_eeg_ablation_videos.py \
  --generation-manifests outputs/eeg_semantic/generated/*/*/*/*/generation_manifest.jsonl \
  --semantic-labels outputs/semantic_labels/eeg_semantic_labels_v1.jsonl \
  --clip-model /path/to/openai-clip-vit-large-patch14 \
  --output outputs/eeg_semantic/reports/video_metrics_long.csv

python scripts/collect_eeg_ablation_metrics.py \
  --jobs outputs/eeg_semantic/experiment_configs/jobs.jsonl \
  --video-metrics outputs/eeg_semantic/reports/video_metrics_long.csv \
  --output outputs/eeg_semantic/reports/all_metrics_long.csv

python scripts/build_eeg_ablation_report.py \
  --metrics outputs/eeg_semantic/reports/all_metrics_long.csv \
  --baseline c2_mse --output-dir outputs/eeg_semantic/reports/c2
```

统计以 subject/fold/training-seed/generation-seed/video 为配对单位，使用双侧 sign-flip
randomization test，同时报告 effect size 和 Holm 多重比较校正。语义解码指标使用
`generation_seed=-1`，视频指标保留真实 generation seed，不先错误平均。

## 6. 代码门禁与实验门禁

代码门禁：

```bash
python -m compileall -q src scripts
python -m pytest -q
python scripts/run_eeg_semantic_ablation.py --stage materialize
python scripts/run_eeg_semantic_ablation.py --stage train \
  --variants a_base b_base c1_mse c2_full c3_autoencoder512 --dry-run
```

代码门禁通过不等于科研数据已经合格。正式训练前仍须完成两项非代码工作：

1. 人工审计 `eeg_semantic_labels_v1.audit.json` 对应的派生标签；
2. 提供 test video 一一对应的固定 trajectory 文件并冻结生成参数。

完成这两项后，才进入正式训练和实际视频生成阶段。
