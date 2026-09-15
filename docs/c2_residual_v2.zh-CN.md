# C2-v2：整个视频 condition 的低维残差

## 实验口径

- chentianlin、video_6fold_1、seed42，01–06 类，4 秒，三 session 特征平均。
- 数据应为 train 312、validation 78、test 78 个视频。不是原八类 C2 的 104-video test，旧结果不可直接配对比较。
- 使用现有 fold-specific 226×512 Tora token-PCA 缓存作为输入；再对整个视频的展平残差拟合 PCA。新 PCA、标准化、EEG 归一化只拟合 first-six train。
- 上游 token-PCA 的 train-only 来源仍需要保持；prepared.protocol.json 记录输入和 projector 文件哈希。
- 新 PCA 默认为 64 维。标准差下限为最大标准差的 0.1 倍；最终能解码到原始 226×4096 Tora condition，保留全部 token 位置。
- prepared 文件含归一化 EEG、残差目标、投影器及分割，约数百 MB，只在服务器生成，不提交 Git。

## 预先固定的四组对比

| variant | 训练目标 |
|---|---|
| mean | 不训练；零残差，即训练集平均 condition |
| regression | 标准化视频残差 MSE |
| contrastive | MSE + 全训练 caption bank 配对对比损失 |
| variance | 上述目标 + 0.1×方差下限 + 0.001×非对角协方差惩罚 |

三种训练共用初始化、数据划分、100 epochs 上限、学习率 0.0002、batch 24、dropout 0.25。
满 20 epochs 后，validation oracle within-category MRR 连续 20 epochs 无提升则停止。
best.pt 按上述 validation MRR 选择；test 不参与模型/维数/超参选择。
每个 epoch 都保存 last.pt；--resume 校验 prepared 哈希和训练设置，恢复优化器、调度器、PyTorch/CUDA 及数据加载随机状态。

训练 bank 每个 caption 保留一个候选；同 caption 为多正例。评估使用视频候选、同 caption 多正例，重复 caption 可能对应多条正确视频。
对并列相似度使用均匀打破并列时的期望 R@1/R@5/MRR，因此常数预测不会从视频排列获得优势。
mean_baseline 中检索值即当前候选及多正例口径的随机基线，不能套用旧的 1/104。
within-category 指标限制候选时使用真实类别，仅用于类别内检索诊断，生成端不使用真实类别。

## 服务器运行

在代码同步后，先运行 CPU 单元和合成数据训练测试：

```bash
cd ~/workspace/diffusion_models_evaluation
conda run --no-capture-output -n eeg-semantic python -m pytest -q tests/test_c2_residual.py
bash scripts/run_c2_residual_wave1.sh --dry-run
```

后台顺序完成数据准备、三组训练、validation 和四组 test：

```bash
cd ~/workspace/diffusion_models_evaluation
mkdir -p outputs/eeg_semantic/logs
tmux new-session -d -s c2-residual-v2 bash -lc '
cd "$HOME/workspace/diffusion_models_evaluation"
set -eo pipefail
bash scripts/run_c2_residual_wave1.sh 2>&1 |
tee -a outputs/eeg_semantic/logs/c2_residual_v2_wave1.log
'
```

重复运行会复用 prepared、跳过匹配设置的已完成训练、恢复中断训练，并重算评估。
不要同时启动两份 wave1 写同一目录；如修改训练设置，应使用新的 --output-root。

```bash
tmux capture-pane -pt c2-residual-v2 -S -50
tail -f outputs/eeg_semantic/logs/c2_residual_v2_wave1.log
```

## 读取对照结果

```bash
conda run --no-capture-output -n eeg-semantic python - <<'PY'
import json
from pathlib import Path
root = Path('outputs/eeg_semantic/c2_residual_v2/dim64')
for variant in ('mean', 'regression', 'contrastive', 'variance'):
    path = root / variant / 'seed42/test/report.json'
    if not path.exists():
        print(variant, 'MISSING')
        continue
    r = json.loads(path.read_text())
    m = r['metrics']
    print(variant, 'n=', r['video_count'], 'epoch=', r['checkpoint_epoch'])
    for key in ('global_r1', 'global_r5', 'within_category_r1',
                'within_category_r5', 'within_category_mrr', 'residual_cosine',
                'between_video_rms', 'target_between_video_rms', 'token_pca_space_mse'):
        print(' ', key, round(m[key], 6))
    print(' shuffle MRR:', r['shuffle']['mean_mrr'])
PY
```

token_pca_space_mse 在原 226×512 空间计算，包含新视频 PCA 丢失的能量；pca_reconstruction_floor_mse 是新压缩器在该空间的重建下限。
报告中的 RMS 在标准化残差空间，不能与旧模型完整 condition RMS 直接比较。
shuffle 使用 100 次同类别内无自配对重排；每个视频的三 session 一起移动。因为模型逐视频推断，重排完整视频预测等价于重排完整 EEG 视频输入。
这是配对依赖诊断，输出重排 MRR 分布和均值，不把它宣称为校准后的显著性 p 值。

验收需同时观察：超过均值基线、残差有合理变化、正确配对优于类别内重排。
仅方差变大、生成视频更丰富不能证明恢复了 EEG 语义。

## 导出 Tora 条件（通过门禁后）

以下以 variance 为例；这是显式操作，wave1 不自动导出或生成视频：

```bash
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_c2_residual.py --stage evaluate --partition test \
  --prepared outputs/eeg_semantic/c2_residual_v2/prepared_fold1_dim64.pt \
  --variant variance --device cuda --export
```

产物为 `outputs/eeg_semantic/c2_residual_v2/dim64/variance/seed42/test/video_index.jsonl`，
可直接接现有 generation runner 的 `--condition-kind tora_state`。

## 维数消融

初次固定 dim64。若扩展 32/128，分别使用新 prepared 文件和同一命令的 --dim 32 / --dim 128，
输出自动按 dim 分目录。维数只按 validation 决定；不根据已读过的 test 持续调参并将最优 test 当无偏结果。
