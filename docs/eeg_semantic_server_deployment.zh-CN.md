# EEG Semantic Ablation：服务器测试与部署

本文档用于把本地实现通过 GitHub 同步到 GPU 服务器，并按低风险顺序验证 Method A、B、
C1、C2 和 Tora condition injection。正式输出与模型权重不提交 GitHub。

## 1. GitHub 同步

服务器首次部署：

```bash
git clone https://github.com/dzx0902/diffusion_models_evaluation.git
cd diffusion_models_evaluation
git checkout main
```

已有工作目录：

```bash
cd /path/to/diffusion_models_evaluation
git status --short
git pull --ff-only origin main
```

如果服务器有未提交修改，先保存或提交它们；不要使用 `git reset --hard`。仓库忽略本地
`data/`，因此 `git pull` 不会同步 EEG、视频、Tora 权重或训练输出。

## 2. 环境与目录

轻量 EEG 环境：

```bash
conda create -n eeg-semantic python=3.10 -y
conda activate eeg-semantic
# 按服务器 CUDA 版本安装 torch，再执行：
pip install -r requirements-eeg-semantic.txt
```

Tora 缓存和生成必须使用服务器现有的官方 Tora 环境。当前 adapter 面向官方 SAT 版，
应存在：

```text
$TORA_ROOT/sat/sample_video.py
$TORA_ROOT/sat/configs/tora/model/cogvideox_5b_tora.yaml
$TORA_ROOT/sat/ckpts/t5-v1_1-xxl/
$TORA_ROOT/sat/ckpts/tora/t2v/
```

设置路径：

```bash
export PROJECT_ROOT=/path/to/diffusion_models_evaluation
export TORA_ROOT=/path/to/Tora
cd "$PROJECT_ROOT"
```

服务器的 `data/` 至少需要：

```text
data/manifests/chentianlin/eeg_trials.csv
data/manifests/captions_simplified.jsonl
data/EEG_and_EYE/<subject>/session*/EEG/eeg_data.npz
```

`data/manifests/eeg_trials.csv` 是多被试汇总表，不得直接用于被试内训练。四个默认
config 都固定使用 `chentianlin/eeg_trials.csv`；更换被试时必须同时更换
trial manifest、split plan 和输出目录。

split 文件若未从旧输出保留，先重新构建：

```bash
python scripts/build_eeg_split_plans.py \
  --video-manifest data/manifests/captions_simplified.jsonl \
  --subject chentianlin \
  --output-dir outputs/eeg_wan/splits \
  --overwrite
```

## 3. 阶段 0：静态测试

这一步不加载 EEG 大文件或 Tora：

```bash
python -m compileall -q src scripts
python -m pytest -q
```

预期：全部测试通过。本地基线为 `60 passed`；PyTorch Transformer 的 nested-tensor warning
是已知提示，不代表测试失败。

只运行本轮新增测试：

```bash
python -m pytest -q \
  tests/test_semantic_schema.py \
  tests/test_semantic_data.py \
  tests/test_eeg_semantic.py \
  tests/test_tora_conditioning.py
```

## 4. 阶段 1：标签和真实 EEG smoke

构建 624 条派生语义标签：

```bash
python scripts/build_eeg_semantic_labels.py \
  --input data/manifests/captions_simplified.jsonl \
  --output outputs/semantic_labels/eeg_semantic_labels_v1.jsonl \
  --overwrite
```

检查 `eeg_semantic_labels_v1.audit.json`：

- `video_count = 624`；
- 8 类各 78 条；
- `missing_relation_count = 0`；
- 状态仍为 `derived_needs_audit`，不能作为人工审核完成的标签发布。

用真实 EEG 各运行一个 train/validation batch：

```bash
python scripts/train_eeg_semantic.py \
  --config configs/eeg_semantic/method_a_template.yaml \
  --device cuda --smoke

python scripts/train_eeg_semantic.py \
  --config configs/eeg_semantic/method_b_structured.yaml \
  --device cuda --smoke
```

smoke 输出固定进入 `outputs/semantic_smoke/`，不得进入论文表格。验证导出：

```bash
python scripts/evaluate_eeg_semantic.py \
  --checkpoint outputs/semantic_smoke/method_a_template_chentianlin_fold1_seed42/best.pt \
  --partition validation --device cuda --smoke
```

## 5. 阶段 2：Tora condition 缓存

切换到 Tora 环境，但仍在项目根目录执行：

```bash
conda activate tora
export TORA_ROOT=/path/to/Tora

python scripts/cache_tora_text_states.py \
  --captions outputs/semantic_labels/eeg_semantic_labels_v1.jsonl \
  --t5-model "$TORA_ROOT/sat/ckpts/t5-v1_1-xxl" \
  --output-dir outputs/tora/text_cache \
  --device cuda --batch-size 1 --save-dtype float32 --overwrite
```

必须确认 `outputs/tora/text_cache/metadata.json` 中：

- `record_count = 624`；
- `max_length = 226`；
- `hidden_dim = 4096`。

这里不能使用 T5 embedder 类的默认 77；Tora-5B 模型 YAML 明确覆盖为 226。

## 6. 阶段 3：PCA bottleneck gate

只用 fold 1 的 416 个 train videos 拟合 streaming PCA：

```bash
python scripts/fit_tora_text_pca.py \
  --index outputs/tora/text_cache/index.jsonl \
  --split-plan outputs/eeg_wan/splits/chentianlin_video_6fold_plan.json \
  --fold video_6fold_1 \
  --output outputs/tora/pca/fold1/tora_text_pca_2048.npz \
  --max-dim 2048 --batch-vectors 4096 --overwrite
```

held-out test round-trip：

```bash
python scripts/evaluate_tora_pca_holdout.py \
  --index outputs/tora/text_cache/index.jsonl \
  --projector outputs/tora/pca/fold1/tora_text_pca_2048.npz \
  --split-plan outputs/eeg_wan/splits/chentianlin_video_6fold_plan.json \
  --fold video_6fold_1 --partition test \
  --dims 128 256 512 1024 2048 \
  --output-dir outputs/tora/pca/fold1/holdout_test
```

根据 holdout 数值选择候选维度，但最终维度还必须通过下一阶段生成控制，不能只按 explained
variance 或 MSE 选择。

导出 PCA-512 targets：

```bash
python scripts/export_tora_pca_targets.py \
  --index outputs/tora/text_cache/index.jsonl \
  --projector outputs/tora/pca/fold1/tora_text_pca_2048.npz \
  --dim 512 \
  --output-dir outputs/tora/pca/fold1/dim512 --overwrite
```

## 7. 阶段 4：Tora exact-condition 生成 gate

先选择 3--5 个 held-out 视频，导出 PCA round-trip condition：

```bash
python scripts/export_tora_pca_roundtrip.py \
  --index outputs/tora/text_cache/index.jsonl \
  --projector outputs/tora/pca/fold1/tora_text_pca_2048.npz \
  --dim 512 --video-ids 01-001 02-040 07-031 \
  --output-dir outputs/tora/controls/pca512_conditions
```

每个样本生成三组：

1. 官方 native caption；
2. cached full T5 state；
3. PCA round-trip state。

injected 组示例：

```bash
conda activate tora
torchrun --standalone --nproc_per_node=1 \
  "$PROJECT_ROOT/scripts/adapters/tora_condition_generate.py" \
  --tora-repo "$TORA_ROOT" \
  --condition "$PROJECT_ROOT/outputs/tora/text_cache/01-001.pt" \
  --adapter-report "$PROJECT_ROOT/outputs/tora/controls/01-001_exact/report.json" \
  -- \
  --base configs/tora/model/cogvideox_5b_tora.yaml configs/tora/inference_sparse.yaml \
  --load ckpts/tora/t2v \
  --output-dir "$PROJECT_ROOT/outputs/tora/controls/01-001_exact" \
  --point_path trajs/fixed_01-001.txt \
  --input-file "$PROJECT_ROOT/outputs/tora/controls/one_prompt.txt"
```

native、full injection、PCA injection 必须使用相同 prompt、point path、seed、采样配置和 GPU
数。若 cached full injection 与 native 明显不一致，停止后续 EEG 生成并先修复 adapter。

## 8. 阶段 5：正式语义训练

先运行一个被试、fold 1、seed 42：

```bash
conda activate eeg-semantic

python scripts/train_eeg_semantic.py \
  --config configs/eeg_semantic/method_a_template.yaml --device cuda

python scripts/train_eeg_semantic.py \
  --config configs/eeg_semantic/method_b_structured.yaml --device cuda

python scripts/train_eeg_tora_alignment.py \
  --config configs/eeg_semantic/method_c_direct_tora.yaml --device cuda

python scripts/train_eeg_tora_alignment.py \
  --config configs/eeg_semantic/method_c_pca_tora.yaml --device cuda
```

C1 `[226,4096]` 显存占用显著高于 C2；先用 `--smoke` 验证 batch，再根据显存将 C1
`batch_size` 从 12 调整为 3/6，并使用梯度累积的后续实现保持有效 batch 一致。

导出 EEG conditions：

```bash
python scripts/predict_eeg_tora_conditions.py \
  --checkpoint outputs/eeg_semantic/method_c_pca512_tora/chentianlin/fold1/seed42/best.pt \
  --partition test \
  --output-dir outputs/tora/eeg_conditions/c_pca512_chentianlin_fold1_seed42 \
  --device cuda
```

## 9. 阶段 6：扩展实验

只有 fold 1 pipeline 全部通过后才扩展：

1. 冻结 ontology、PCA dimension、threshold search 和训练超参；
2. 扩展 3 seeds；
3. 扩展 6 folds；
4. 扩展完整三 session 被试；
5. 加入 prototype、soft semantic contrastive、caption centroid、EEG augmentation；
6. 自动汇总 trial/video/subject/fold/seed 的 CSV 与显著性检验；
7. 最后批量运行 Tora，保持 fixed trajectory 和 generation config。

## 10. 输出保留与 GitHub 边界

以下内容只留服务器或对象存储，不提交 GitHub：

- `data/`；
- `outputs/semantic_labels/`；
- `outputs/semantic_smoke/`；
- `outputs/eeg_semantic/`；
- `outputs/tora/`；
- checkpoints、T5 states、PCA targets 和生成视频。

应提交 GitHub 的只有代码、YAML、测试、轻量报告模板和不含个人路径的汇总表。
