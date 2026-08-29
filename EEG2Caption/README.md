# EEG2Caption：双主体 EEG 分类与 Caption 导出

这个目录是可以单独汇总和交付的 EEG-only 流程，只包含当前采用的 Compact
分类模型，不依赖原来的 `Classification`、`HRT_training` 或视频生成目录。

完整数据流为：

```text
同一视频的 session1 / session2 / session3 EEG
                  ↓ 共享 Compact encoder（逐 session 编码）
          三组 6 维 object logits
                  ↓ logits 平均（不平均原始 EEG）
          fused Top-2 主体预测
                  ↓
          EEG caption / generation prompt
```

这里的 `EEG caption` 是**主体级Caption**：只描述EEG分类器预测出的两个主体。
当前分类器没有action head，因此不会从真实caption复制动作词，也不能声称由EEG预测了
`kick/run/fly` 等动作。后续训练独立action head后，可以再把动作短语追加到这里导出的主体Caption。

## 任务与数据划分

- 单被试训练，每个被试单独保存模型。
- 只使用 `01_001` 到 `06_078`，共468个双主体视频。
- 每段 EEG 使用前800个采样点，即200 Hz下的4秒。
- 类别顺序固定为 `[person, dog, car, ball, flower, bird]`。
- 三个session使用相同的视频划分，避免同一视频跨训练集和测试集泄漏。
- 默认划分：训练324、验证48、测试96（每个pair类别54/8/16）。
- 最优权重只按照验证集 fused macro AP 选择，训练阶段不查看测试指标。

六类视频分别为：

| Prefix | 两个主体 |
|---|---|
| 01 | person + ball |
| 02 | dog + ball |
| 03 | dog + car |
| 04 | car + flower |
| 05 | flower + bird |
| 06 | person + bird |

## 环境

脚本直接使用当前终端的 `python`，不会自动激活或指定conda环境。需要：

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn

数据默认位于：

```text
/userhome2/zhoutianyi/Dataset/Multi-Object
```

## 一条命令完成训练、测试与Caption生成

```bash
cd /userhome2/zhoutianyi/BrainDecoding/Multi-object/EEG2Caption

SUBJECT=zhoutianyi DEVICE=cuda:0 bash scripts/run_all.sh
```

常用参数可以直接覆盖：

```bash
SUBJECT=wangxinlin \
DEVICE=cuda:1 \
EPOCHS=100 \
BATCH_SIZE=32 \
SEED=42 \
bash scripts/run_all.sh
```

`Subject1` 到 `Subject20` 也可以使用，程序会通过
`Dataset/Multi-Object/Subjects/subject.txt` 映射到实际文件夹名称。

## 分步运行

只训练：

```bash
SUBJECT=zhoutianyi DEVICE=cuda:0 bash scripts/train.sh
```

只对已有 `best.pt` 测试并生成Caption：

```bash
SUBJECT=zhoutianyi DEVICE=cuda:0 bash scripts/infer_and_caption.sh
```

指定任意兼容的checkpoint：

```bash
SUBJECT=zhoutianyi \
DEVICE=cuda:0 \
CHECKPOINT=/path/to/best.pt \
RESULT_DIR=/path/to/new_results \
bash scripts/infer_and_caption.sh
```

## 输出文件

每次实验保存到：

```text
outputs/<experiment>/<subject>/
├── best.pt                     # 验证集mAP最优权重
├── last.pt                     # 最后一个epoch权重
├── splits.json                 # 324/48/96视频划分及ID
├── history.json                # 每个epoch训练和验证指标
├── training_summary.json
├── test_metrics.json           # 三个session及融合测试结果
├── test_predictions.pt         # 对齐的测试logits、label、video ID
└── captions/
    ├── test_captions.json      # 完整概率、逐session和融合结果
    ├── test_captions.csv       # 便于组内查看和统计
    └── generation_prompts.txt  # video_id + 可直接用于生成模型的prompt
```

`test_captions.*` 中：

- `eeg_caption` 只由 fused EEG Top-2 预测生成；
- `generation_prompt` 是 `eeg_caption` 加固定质量描述；
- `audit_ground_truth_*` 仅用于离线评测，明确不参与 EEG caption 生成。

## 已有结果

当前 `zhoutianyi` 的已完成结果复制在：

```text
released_results/compact_eeg_3session_fusion_test20/zhoutianyi/
```

该结果使用100 epochs、seed 42和324/48/96划分。融合测试结果：

- macro AP：0.5095
- Top-2 exact set accuracy：0.2396（23/96）
- Top-2 recall：0.4583
- pair-head accuracy：0.2083

随机Top-2集合命中率为 `1/C(6,2)=0.0667`；六个已知pair类别的随机准确率为
`1/6=0.1667`。论文和汇报中需要区分这两个chance定义。

## 文件职责

- `src/prepare_labels.py`：生成468条有序6维多标签。
- `src/common.py`：EEG读取、Compact模型、三session融合与指标。
- `src/train.py`：训练和验证，只用验证mAP选择 `best.pt`。
- `src/infer.py`：在checkpoint记录的测试视频上推理。
- `src/generate_captions.py`：将测试集 fused Top-2 转为caption与prompt。
