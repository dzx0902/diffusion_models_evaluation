# 跨 session 配对审计与六分类对照

## 本地初查（2026-09-16）

本地 chentianlin first-six 清单含 468×3=1404 条 trial。NPZ `order_index` 是从1开始的数组排序，metadata `order_index` 是播放顺序。
审计对已识别的顺序数组使用 metadata `sorted_index` 比较，并独立比较 `playback_order_index`；不把字段名称相同当作含义相同。
本地内部检查 PASS：1404条全部完成检查，无非有限值、平坦通道或字节级重复trial；三个session均为200Hz、62通道且通道顺序一致。
trial RMS中位数（原数组单位）分别为8.168e-6、8.299e-6、6.825e-6。相对alpha功率分别为0.1425、0.1866、0.1897；这些描述性差异不构成采集错误的证据。
metadata 中 `onset_sample - annotation_event_sample` 为 session1=0、session2=-4274、session3=-15159（200Hz）。
这些固定差值可能来自 first_samp/crop 的时间原点转换，尚未查证，不能直接认定截取错了21.37或75.795秒。
本仓库未发现生成这些事件字段的预处理脚本，原始刺激日志未提供，因此独立刺激对齐标记为 NOT_VERIFIED。

## 1. 服务器审计（只读原数据）

```bash
cd ~/workspace/diffusion_models_evaluation
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/audit_eeg_sessions.py \
  --trials data/manifests/chentianlin/eeg_trials.csv \
  --output outputs/eeg_semantic/session_audit/audit.json
```

覆盖 manifest/NPZ filename/metadata video_id、trial_index、播放顺序、长度、mask、采样率、通道顺序、非有限值、平坦通道、完全重复信号，
并记录通道统计、trial RMS、1–45Hz相对频带能量与 onset 差值分布。频带统计是描述性诊断，不把 session 总体差异自动判成错误。
输入只选择 first-six；它检查全部468个视频而非只看某个fold。错误写入报告并返回非零退出码。
audit.json 的 PASS 仅表示内部检查通过，不保证 EEG 波形对应的刺激标签正确。
单位、参考方式和滤波配置若无来源文件则保持 unknown，不从振幅大小猜测。

如有独立原始刺激日志，可额外传 `--events PATH.csv`，列为 `session,onset_sample,video_id`。
onset_sample 必须先转换为与 metadata onset 相同的200Hz坐标；不自动估计偏移，不应从当前 metadata 复制一份作为独立验证。
仍需核对预处理脚本中的 first_samp、裁剪起点、事件采样率转换及显示延迟。

## 2. 六分类对照

复用 C2-v2 prepared EEG。分类目标仅为01–06标签，不使用caption残差。
ridge 使用每通道25个时间窗均值，回归到六类one-hot，alpha仅按validation准确率选择；compact 使用共享单session编码器和交叉熵。
all_sessions 用train三session训练；holdout_s3仅用train session1/2训练。归一化及验证选模型只使用协议允许的session。
默认100epoch、至少20epoch、patience20、lr0.0003、dropout0.25。支持自动恢复中断训练。
train/session3（holdout_s3）是已见视频未见session诊断；test/session3才同时未见视频和session。

```bash
conda run --no-capture-output -n eeg-semantic python -m pytest -q tests/test_session_audit_category.py
conda run --no-capture-output -n eeg-semantic python scripts/run_eeg_session_category.py --wave --dry-run
```

审计如有错误先查清；内部检查通过后运行：

```bash
mkdir -p outputs/eeg_semantic/logs
tmux new-session -d -s eeg-session-category bash -lc '
set -eo pipefail
cd "$HOME/workspace/diffusion_models_evaluation"
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_eeg_session_category.py --wave --device cuda 2>&1 |
tee -a outputs/eeg_semantic/logs/session_category.log
'
```

```bash
tail -f outputs/eeg_semantic/logs/session_category.log
```

完成标记 `[category] wave COMPLETE`。粗分类结果不作为连续语义重建结果。

```bash
conda run --no-capture-output -n eeg-semantic python - <<'PY'
import json
from pathlib import Path
root=Path('outputs/eeg_semantic/session_category_control')
for protocol in ('all_sessions','holdout_s3'):
    for model in ('ridge','compact'):
        r=json.loads((root/protocol/model/'seed42/report.json').read_text())
        print('\n',protocol,model,'epoch=',r['epoch'],'alpha=',r['alpha'])
        for key in ('train/training_sessions_mean','train/session3','test/training_sessions_mean','test/session3'):
            m=r['reports'][key]
            print(key,'n=',m['video_count'],f"accuracy={m['accuracy']:.4f}",f"balanced={m['balanced_accuracy']:.4f}",'chance=0.1667')
PY
```
