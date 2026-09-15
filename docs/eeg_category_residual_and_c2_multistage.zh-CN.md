# 类别残差语义检索与 C2 多阶段强化

## 1. 研究边界

固定模板只作为 coarse-category 下界。Method D 显式减去仅由 train captions
估计的类别中心，并在同类别候选中区分动作/关系。test caption bank 检索只用于判断
EEG 是否含细粒度语义，属于 closed-set diagnostic；不得描述成真实部署时已知 test caption。

C2 的 train-overfit stage 只检查容量和优化链路。该阶段的
`data_protocol.json` 固定写入 `diagnostic_train_evaluation=true`，禁止混入 held-out 表格。

## 2. Method D：类别残差检索

先用本地 CLIP 生成一次全数据 caption targets：

```bash
export CLIP_ROOT="$PWD/.ms_video_models/CLIP/clip-vit-base-patch32"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

conda run --no-capture-output -n eeg-semantic python -u \
  scripts/build_clip_caption_targets.py \
  --semantic-labels outputs/semantic_labels/eeg_semantic_labels_v1.jsonl \
  --clip-model "$CLIP_ROOT" \
  --output-dir outputs/eeg_category_residual/clip_caption_targets \
  --batch-size 64 --device cuda --overwrite
```

每个 fold 必须单独拟合 train-only 类别中心：

```bash
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/build_category_residual_targets.py \
  --targets outputs/eeg_category_residual/clip_caption_targets/index.jsonl \
  --split-plan outputs/eeg_wan/splits/chentianlin_video_6fold_plan.json \
  --fold video_6fold_1 \
  --output-dir outputs/eeg_category_residual/fold1/targets --overwrite
```

fold1 训练门禁：

```bash
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/train_eeg_pooled_retriever.py \
  --trials data/manifests/chentianlin/eeg_trials.csv \
  --targets outputs/eeg_category_residual/fold1/targets/index.jsonl \
  --split-plan outputs/eeg_wan/splits/chentianlin_video_6fold_plan.json \
  --experiment video_6fold_1 --validation-partition validation \
  --duration-sec 4 --architecture multiscale \
  --hidden-dim 128 --encoder-layers 1 --heads 8 --token-count 75 \
  --group-sessions --batch-size 48 \
  --contrastive-bank train --negative-scope category \
  --selection-metric session_averaged_within_category_mrr \
  --mse-weight 0.05 --cosine-weight 1 --contrastive-weight 2 \
  --variance-weight 0.1 --covariance-weight 0.001 \
  --epochs 60 --min-epochs 15 --early-stop-patience 10 \
  --seed 42 --device cuda --auto-resume \
  --output-dir outputs/eeg_category_residual/fold1/seed42
```

held-out test 使用相同 checkpoint 与 train-only transform：

```bash
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/train_eeg_pooled_retriever.py \
  --trials data/manifests/chentianlin/eeg_trials.csv \
  --targets outputs/eeg_category_residual/fold1/targets/index.jsonl \
  --split-plan outputs/eeg_wan/splits/chentianlin_video_6fold_plan.json \
  --experiment video_6fold_1 --validation-partition test \
  --duration-sec 4 --architecture multiscale \
  --hidden-dim 128 --encoder-layers 1 --heads 8 --token-count 75 \
  --group-sessions --batch-size 48 \
  --contrastive-bank train --negative-scope category \
  --selection-metric session_averaged_within_category_mrr \
  --mse-weight 0.05 --cosine-weight 1 --contrastive-weight 2 \
  --variance-weight 0.1 --covariance-weight 0.001 \
  --checkpoint outputs/eeg_category_residual/fold1/seed42/best.pt \
  --seed 42 --device cuda \
  --output-dir outputs/eeg_category_residual/fold1/seed42/test
```

首要报告 `session_averaged_within_category_recall_at_1/5` 和
`session_averaged_within_category_mrr`，并与同一 JSON 内 chance 比较。只有该门禁超过
chance，才实现 train-bank caption inference 和视频生成。

## 3. C2 多阶段强化

阶段定义位于 `configs/eeg_semantic/c2_multistage.yaml`：

1. `classifier_pretrain`：先让共享 EEG backbone 学会 coarse semantics；
2. `pooled_alignment`：只对齐 Tora token 平均语义，降低优化难度；
3. `token_refinement`：再细化完整 226×512 token condition；
4. `train_overfit_diagnostic`：在 train 上评估容量上限，不作为泛化结果。

检查命令：

```bash
python scripts/run_c2_multistage.py --dry-run
```

正式顺序训练：

```bash
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_c2_multistage.py --device cuda --skip-existing
```

阶段间通过 `--init-checkpoint` 只 warm-start model state；optimizer、scheduler、loss
和学习率均按新阶段重建。正常 resume 仍要求 config 完全一致，两种机制不可同时使用。

## 4. 判定规则

- classifier stage：看 held-out `category_accuracy`，不看随机 condition retrieval；
- pooled stage：看 held-out `token_cosine` 与 retrieval；
- token stage：看 held-out MSE/cosine/retrieval，并最终生成视频；
- overfit stage：若 train MSE 仍降不下去，说明瓶颈在架构或优化；若 train 很好但 test
  很差，说明是泛化/样本量问题，不能将 train 指标作为重建证据。
