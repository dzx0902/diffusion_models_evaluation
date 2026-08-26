# EEG Semantic Caption Ablation Protocol

本文档冻结第一阶段 EEG→semantic decoding 实验口径。Tora 视频生成只有在服务器端
semantic-condition exact injection 控制通过后才进入主实验。

## 主研究变量

主实验只改变 EEG semantic representation：

- Method A：`subject + object + coarse_action` 多头分类，再使用确定性模板；
- Method B：在 A 上增加 `count + fine_action + relation`，优先使用预测 relation 生成句子；
- Method C1：EEG 直接回归 Tora text encoder full hidden state；
- Method C2：EEG 回归仅由 training captions 拟合的 PCA semantic bottleneck。

EEG encoder、video split、训练 epoch、optimizer、trajectory、Tora checkpoint、采样参数和
seed 策略必须保持一致。Method-specific head 参数量单独报告。

## 数据协议

- 主设置为 subject-dependent；每个被试独立训练和评估。
- 一个 session trial 是一个独立样本；禁止使用 `session_average`。
- 同一视频的不同 session 共享语义 target，可在训练 batch 中构成 multi-positive。
- single-trial 指标是主结果；三 session logits aggregation 只能作为独立 ablation。
- train/validation/test 必须使用已有 video-level 6-fold，同一个 `video_id` 不得跨 partition。
- 当前 01--06 原始 trial 为 800 点，07--08 为 1200 点。主实验统一取刺激开始后的前
  800 点，避免 duration 成为类别捷径。完整 6 秒输入属于后续 time-window ablation。
- 每个 trial/channel 独立 z-score；不从 validation/test 拟合归一化统计。

## Semantic labels

`captions_simplified.jsonl` 的 `caption_entities` 和 `caption_relations` 保留为原始审计字段。
核心实体由数据集 8 个预定义类别确定；自动产生的 coarse/fine action 标注统一带
`derived_needs_audit`，在人工审查完成前不得称为人工真值。

Method B 的 fine-action/relation vocabulary 只使用对应 fold 的 training videos 构建；低频
及 validation/test 未见值映射到 `__unknown__`。场景和属性已被 structured_v2 caption
主动删除，因此第一阶段不把 scene/attribute 列为 Method B 必选 slot。

## 阈值、选择和测试集

- confidence threshold 候选默认为 `0.3, 0.4, ..., 0.8`；只能在 validation 上选择。
- checkpoint 以完整 validation 的 aggregate slot macro F1 选择。
- 除 macro/micro F1 外必须报告 exact match、每 slot coverage 及预测 slot 数；避免通过
  全部输出或全部省略获得误导性分数。
- test partition 不参与 vocabulary、threshold、checkpoint 或超参数选择。
- PCA、prototype、semantic centroid 和标准化统计必须记录 fitted partition，且只能 fit train。

## Tora 控制实验

服务器接入后，每个生成样本保留以下三层：

1. GT caption → Tora native text path；
2. GT caption → cached exact hidden state injection；
3. EEG → predicted semantic condition injection。

每层使用相同 trajectory、seed、sampler、steps、CFG、分辨率、帧数和 style suffix。
如果第 2 层不能近似保持第 1 层生成语义，停止 EEG 视频生成并先修正 condition adapter。

官方开源 Tora-5B 配置将 `FrozenT5Embedder.max_length` 和 DiT `text_length` 都设为 226，
hidden dimension 为 4096。不能使用 embedder 类的默认 77，也不能沿用 Wan 的 128 slots。

服务器端先缓存 target：

```bash
export TORA_ROOT=/path/to/Tora
python scripts/cache_tora_text_states.py \
  --captions outputs/semantic_labels/eeg_semantic_labels_v1.jsonl \
  --output-dir outputs/tora/text_cache
```

对一个 caption/trajectory 做 exact-condition control 时，native 组运行官方命令；injected 组
保持所有官方参数不变，改用以下 adapter：

```bash
torchrun --standalone --nproc_per_node=1 \
  scripts/adapters/tora_condition_generate.py \
  --tora-repo "$TORA_ROOT" \
  --condition outputs/tora/text_cache/01-001.pt \
  --adapter-report outputs/tora/controls/01-001_exact/report.json \
  -- \
  --base configs/tora/model/cogvideox_5b_tora.yaml configs/tora/inference_sparse.yaml \
  --load ckpts/tora/t2v \
  --output-dir outputs/tora/controls/01-001_exact \
  --point_path trajs/example.txt \
  --input-file one_prompt.txt
```

adapter 只覆盖 conditional `crossattn`，不替换 unconditional condition，也不处理 trajectory；
trajectory 仍由官方 `process_traj` 路径生成。

## 当前命令

```powershell
python scripts/build_eeg_semantic_labels.py `
  --output outputs/semantic_labels/eeg_semantic_labels_v1.jsonl `
  --overwrite

python scripts/train_eeg_semantic.py `
  --config configs/eeg_semantic/method_a_template.yaml

python scripts/train_eeg_semantic.py `
  --config configs/eeg_semantic/method_b_structured.yaml

# 在 Tora 服务器缓存官方 T5 [226,4096] target 后运行：
python scripts/train_eeg_tora_alignment.py `
  --config configs/eeg_semantic/method_c_direct_tora.yaml

python scripts/evaluate_eeg_semantic.py `
  --checkpoint outputs/eeg_semantic/method_a_template/chentianlin/fold1/seed42/best.pt `
  --partition test
```

`--smoke` 只运行一个 train/validation batch，输出到 `outputs/semantic_smoke`。其指标不得进入
实验表格。
