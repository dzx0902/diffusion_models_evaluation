# C2-v4：预测类别中心加类别内残差

## 预处理源代码检索记录

2026-09-16 检查了当前仓库的 tracked/untracked/ignored Python、MATLAB、notebook 文件（排除依赖和模型仓库）、EEG2Caption.zip、data.rar 的源文件列表，以及上级 scripts 和 dataset_video_pipeline。
找到 `scripts/build_eeg_video_manifest.py`（读取已经切好的 NPZ 并构建映射）和 `EEG2Caption/src/common.py`（加载已切好的 EEG），没有找到生成 chentianlin 的 onset_sample/annotation_event_sample 和切片 NPZ 的源代码。
部分其他被试的 `*_preprocess_summary.json` 描述了 ICA 等处理，但当前 chentianlin EEG 目录无此摘要；不能把其他被试的配置当成 chentianlin 的配置。
data.rar 的源码列表仅发现眼动 check.py，EEG2Caption.zip 只有分类/推断管线源文件。当前副本没有提供独立刺激事件或 EEG 原始文件来验证固定时间原点差值。
结论仍为内部映射 PASS、独立刺激对齐 NOT_VERIFIED。此次不平移、不改写 EEG。

## 模型与对照

复用 first-six、fold1、dim64 prepared，以及已经完成的 `session_category_control/{protocol}/ridge/seed42/best.pt`。
每套协议（all_sessions、holdout_s3）进行三组对照：

1. mean：全局训练均值，即标准化 residual 空间的零向量。
2. predicted_centers：Ridge 预测六类分数，softmax 得到类别概率，加权训练集的类别目标中心。
3. centers_plus_residual：上述输出加 EEG 预测的类别内残差。

计算都在 C2-v2 的64维标准化空间内，最终可经两级逆变换恢复226×4096条件。
类别中心只用train目标，残差训练目标为 `z - center[train_category]`；训练允许使用标签。
推断接口仅接收 EEG 与 checkpoint，不接收真实类别或 caption。
残差预测器为固定25窗通道特征的岭回归，先用可解释的低容量对照判断是否有稳定增益。

## 选择与泄漏边界

- 类别 Ridge 已由此前 validation 准确率选择，不重新使用 test 选择分类器。
- softmax temperature 从0.01/0.03/0.1/0.3/1/3中按validation分类NLL选择；它使分数转换成可用概率，但不保证完美校准。
- 残差 Ridge alpha 为0.001/0.01/0.1/1/10/100，残差权重为0/0.25/0.5/1；联合按validation类内MRR选择。
- 同分优先保留先尝试的零残差；若最终权重为0，明确表示验证集未支持添加残差，不当作细粒度语义成功。
- 共享归一化只拟合train视频及协议允许的session；holdout_s3不使用session3选择参数。
- test 类别只用于计算指标和划定oracle类内检索候选，绝不参与生成condition。
- 类别内shuffle只打乱预测残差，保持各视频的类别中心预测不变，以区分粗类别收益和残差收益；仅作诊断，不提供显著性p值。
- 使用与C2-v2相同的多正例caption和并列得分口径。不能直接与旧八类104视频表格比较。

## 服务器执行

```bash
cd ~/workspace/diffusion_models_evaluation
conda run --no-capture-output -n eeg-semantic python -m pytest -q tests/test_c2_hierarchical.py
conda run --no-capture-output -n eeg-semantic python scripts/run_c2_hierarchical.py --dry-run
```

该轮使用CPU，不需要重训深度模型，也不需要重新计算T5缓存：

```bash
mkdir -p outputs/eeg_semantic/logs
tmux new-session -d -s c2-hierarchical-v4 bash -lc '
set -eo pipefail
cd "$HOME/workspace/diffusion_models_evaluation"
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_c2_hierarchical.py --protocol both --threads 4 2>&1 |
tee -a outputs/eeg_semantic/logs/c2_hierarchical_v4.log
'
```

```bash
tail -f outputs/eeg_semantic/logs/c2_hierarchical_v4.log
```

完成标记 `[c2-hierarchical] COMPLETE`。会输出 train/validation/test 的训练session平均和session3结果。
报告目录：`outputs/eeg_semantic/c2_hierarchical_v4/{protocol}/seed42/report.json`。
再次运行会校验并复用best.pt，重算评估。改prepared或分类器时使用新输出目录，避免混用。

```bash
conda run --no-capture-output -n eeg-semantic python - <<'PY'
import json
from pathlib import Path
root=Path('outputs/eeg_semantic/c2_hierarchical_v4')
for protocol in ('all_sessions','holdout_s3'):
    r=json.loads((root/protocol/'seed42/report.json').read_text())
    print('\n',protocol,'temperature=',r['temperature'],'alpha=',r['alpha'],'residual_weight=',r['selected_residual_weight'])
    for group in ('training_sessions_mean','session3'):
        for variant in ('mean','predicted_centers','centers_plus_residual'):
            row=r['reports'][f'test/{group}/{variant}']
            m=row['metrics']
            print(group,variant,f"MSE={m['token_pca_space_mse']:.6f}",
                  f"R1={m['global_r1']:.4f}",f"within_MRR={m['within_category_mrr']:.4f}",
                  f"cos={m['residual_cosine']:.4f}",'residual_shuffle=',row['residual_shuffle_mean_mrr'])
PY
```

先看 predicted_centers 相对 mean 是否改善，再看 centers_plus_residual 相对 predicted_centers 的增量，不能将类别中心收益描述为动作/关系恢复。
需要生成视频时显式加 `--export`，将导出三个方法在test训练session平均下的condition及video_index.jsonl，不自动运行视频生成。
