# C2-v3：单 session 预测与跨 session 一致性

复用 C2-v2 的 first-six、fold1、dim64 prepared 文件。目标 PCA 不重新拟合。
输出位于 `outputs/eeg_semantic/c2_sessions_v3/{protocol}/{variant}/seed42`。

## 六组预先固定的实验

| protocol | 训练和模型选择 | 最终报告 |
|---|---|---|
| all_sessions | train 视频三 session 训练；validation 视频三 session 平均 MRR 选模型 | 每个 session 和三 session 平均的 train/validation/test |
| holdout_s3 | train 视频 session1/2 训练；validation 视频 session1/2 平均 MRR 选模型 | 每个 session、session1/2 平均的 train/validation/test |

每个协议运行 ridge、single_session、consistent 三组。

- ridge：每通道自适应平均到 25 个时间窗，展平后做带截距岭回归。kernel 除以特征维数；alpha 在 0.001/0.01/0.1/1/10/100 中仅按 validation 类内 MRR 选择。它是固定低时间分辨率特征的线性基线，不代表所有线性特征方案。
- single_session：共享 Compact encoder，每个 session 的输出都接受 MSE + caption bank 配对对比监督；推断平均 session 的预测。
- consistent：同上，另加 0.1×同视频各 session 预测与其均值之间的 MSE。
- 两个深度模型使用相同初始化、batch24（按视频分组）、dropout0.25、lr0.0002，100 epoch 上限，最少20 epoch、20 epoch patience。仍以 validation 类内 MRR 选 best.pt。

归一化先反解 prepared 内旧的 session 归一化，再使用当前协议允许的 train 视频和 session 拟合共享通道均值/标准差。
holdout_s3 的统计拟合、梯度更新和模型选择均不使用 session3。
train/session3 是“见过视频、没见过该 session”的重复性诊断；test/session3 同时未见视频和 session。
目标 caption 在 train/session3 诊断中已用于训练，不能称作未见语义泛化。

所有检索复用 C2-v2 的视频候选、多正例 caption、并列分数期望和均值基线。
shuffle 为同类别内完整视频配对重排20次，仅用于诊断；不同分区候选数不同，应分别对照自己的均值/随机基线。
本轮不自动生成视频。对比旧 C2-v2 时，共享归一化及逐 session 监督也有变化，不能将差异单独归因于一致性项；single_session vs consistent 才是该项的匹配对照。

## 服务器命令

```bash
cd ~/workspace/diffusion_models_evaluation
conda run --no-capture-output -n eeg-semantic python -m pytest -q tests/test_c2_sessions.py tests/test_c2_residual.py
conda run --no-capture-output -n eeg-semantic python scripts/run_c2_sessions.py --stage wave --dry-run
```

```bash
mkdir -p outputs/eeg_semantic/logs
tmux new-session -d -s c2-sessions-v3 bash -lc '
set -eo pipefail
cd "$HOME/workspace/diffusion_models_evaluation"
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_c2_sessions.py --stage wave --device cuda 2>&1 |
tee -a outputs/eeg_semantic/logs/c2_sessions_v3.log
'
```

```bash
tmux capture-pane -pt c2-sessions-v3 -S -50
tail -f outputs/eeg_semantic/logs/c2_sessions_v3.log
```

正常结束标记为 `[c2-sessions] wave COMPLETE`。wave 支持恢复深度训练，跳过设置匹配的已完成训练；评估会重新运行。
prepared 文件必须存在。更换数据、目标维数或超参时使用新的 --output-root，避免覆盖已有结果。

## 汇总结果

```bash
conda run --no-capture-output -n eeg-semantic python - <<'PY'
import json
from pathlib import Path
root = Path('outputs/eeg_semantic/c2_sessions_v3')
for protocol in ('all_sessions', 'holdout_s3'):
    for variant in ('ridge', 'single_session', 'consistent'):
        path = root / protocol / variant / 'seed42/report.json'
        if not path.exists():
            print(protocol, variant, 'MISSING')
            continue
        data = json.loads(path.read_text())
        print('\n', protocol, variant, 'epoch=', data['checkpoint_epoch'], 'alpha=', data['selected_alpha'])
        for key in ('train/training_sessions_mean', 'train/session3', 'test/training_sessions_mean', 'test/session3'):
            row = data['reports'][key]
            m, b = row['metrics'], row['mean_baseline']
            print(key, 'n=', row['video_count'],
                  f"R1={m['global_r1']:.4f}", f"MRR={m['within_category_mrr']:.4f}",
                  f"baseline={b['within_category_mrr']:.4f}",
                  f"cos={m['residual_cosine']:.4f}", f"MSE={m['token_pca_space_mse']:.6f}",
                  'shuffle=', row['shuffle_mean_mrr'])
PY
```
