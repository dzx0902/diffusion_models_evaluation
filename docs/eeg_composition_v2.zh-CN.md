# 三主体组合泛化：第二阶段开发与诊断

本阶段保留 `outputs/eeg_composition/`，只读取其中的prepared缓存及原始checkpoint。
所有新增结果写入 `outputs/eeg_composition_v2/`。不修改源EEG、标签或原始测试结果。
07/08已有结果已被查看，后续结果属于探索性研究，不能包装为全新盲测。

## 实验内容

1. audit-baseline：原checkpoint，不训练。比较前4秒、完整6秒、三个4秒滑窗
   （0–4、1–5、2–6秒）平均logits。跨session/三session平均口径不变。
2. object-only：四协议重训，移除pair CE，其余训练设置不变；使用原420/48/156划分。
3. development：仅01–06内部留出一种组合，剩余五类各70个训练，共350；
   留出类别全部78个用于开发验证，不使用另外五类的验证视频，不使用07/08。
   六种留出都保持六个主体在训练中有正样本。开发验证是单一组合，标签恒定，AP无定义；
   因此按允许训练session上的Top-2 exact选择checkpoint（并列保留较早epoch）。
   单一组合分数不能证明组合内EEG区分能力；应比较六个留出方向的结果，不能只选最好的一类。
   Cross-session开发报告另测留出的session；Session-average开发报告与选择集重合，
   是开发成绩，不是独立测试。开发阈值也仅是诊断，不能直接迁移为正式阈值。

逐主体报告同时包含Top-k recall/FPR和原阈值recall/FPR、正负分数均值。
阈值曲线只用于审计，保持原始.10到.90、步长.05的选择规则不变。
额外加入全部报阳性的F1基线。

正式测试的打乱对照将整个视频输出向量全局置换200次，窗口和session一起移动；
不逐主体打乱，不在07或08类内打乱（同类标签相同，后者无效）。
模型是逐视频独立推理，因此这等价于打乱视频EEG与标签的配对，而不必重复跑模型。
该对照只检验标签配对信息，不排除刺激顺序等混杂，不替代独立刺激时序核验。
分位数为打乱零假设参考区间，不是独立样本置信区间，不自动宣称显著。

本轮不实现GroupNorm/损失重标定/新注意力结构，避免一次混合过多改动。

## 服务器执行

```bash
cd ~/workspace/diffusion_models_evaluation
git -c http.version=HTTP/1.1 pull --ff-only origin main && git log -1 --oneline
conda run --no-capture-output -n eeg-semantic python -m pytest -q
```

三个完整dry-run（不读缓存、不训练、不写文件）：

```bash
conda run --no-capture-output -n eeg-semantic python scripts/run_eeg_composition_v2.py --stage audit-baseline --device cuda --dry-run
conda run --no-capture-output -n eeg-semantic python scripts/run_eeg_composition_v2.py --stage object-only --device cuda --resume --dry-run
conda run --no-capture-output -n eeg-semantic python scripts/run_eeg_composition_v2.py --stage development --protocols session_average --variants original object_only --device cuda --resume --dry-run
```

先顺序运行原checkpoint审计和4个object-only训练：

```bash
mkdir -p outputs/eeg_semantic/logs
tmux new-session -d -s composition-v2 bash -lc '
set -eo pipefail
cd "$HOME/workspace/diffusion_models_evaluation"
conda run --no-capture-output -n eeg-semantic python -u scripts/run_eeg_composition_v2.py --stage audit-baseline --device cuda 2>&1 | tee -a outputs/eeg_semantic/logs/composition_v2_audit.log
conda run --no-capture-output -n eeg-semantic python -u scripts/run_eeg_composition_v2.py --stage object-only --device cuda --resume 2>&1 | tee -a outputs/eeg_semantic/logs/composition_v2_object_only.log
'
```

开发验证先做Session-average：6种留出×2模型=12组，每组100epochs，独立保存。
建议前一批结束后再执行，不同时占用GPU：

```bash
tmux new-session -d -s composition-dev bash -lc '
set -eo pipefail
cd "$HOME/workspace/diffusion_models_evaluation"
conda run --no-capture-output -n eeg-semantic python -u scripts/run_eeg_composition_v2.py --stage development --protocols session_average --variants original object_only --device cuda --resume 2>&1 | tee -a outputs/eeg_semantic/logs/composition_v2_development.log
'
```

后续需要跨session开发时，将`--protocols session_average`改为`--protocols cs_s1 cs_s2 cs_s3`（36组）。
本轮不要求立即运行36组。

日志与结果：

```bash
tail -n 30 -F outputs/eeg_semantic/logs/composition_v2_object_only.log
tail -n 30 -F outputs/eeg_semantic/logs/composition_v2_development.log
column -s, -t outputs/eeg_composition_v2/summary.csv
```

目录分别为：

```text
outputs/eeg_composition_v2/
  audit-baseline/chentianlin/<protocol>/original/seed42/audit.json
  object-only/chentianlin/<protocol>/object_only/seed42/audit.json
  development/held_01/chentianlin/<protocol>/<variant>/seed42/audit.json
  summary.csv
```

每个audit目录另有predictions/*.pt（video_ids、logits、labels、object_names）。
新训练目录含best.pt、last.pt、history.json、completed.json，可通过--resume恢复。
summary包含开发和测试，但有development_only字段，必须分开解释。
更改训练预算请用新的--output-root；不要覆盖旧实验。原始checkpoint不会被本脚本写入。

## 下一步只读检查

不要重跑已经完成的audit-baseline/object-only。继续上面的12组development即可；
--resume可恢复中断训练，已有同配置completed.json的训练会跳过。

新增报告入口只读取audit.json，不需要GPU，不重新推理，也不修改实验结果：

```bash
python scripts/report_eeg_composition_v2.py --section test --protocols session_average
python scripts/report_eeg_composition_v2.py --section test --protocols cs_s1 cs_s2 cs_s3
python scripts/report_eeg_composition_v2.py --section development --protocols session_average
```

test报告展示所有三种输入方式、07/08分项、六主体Top-k/阈值指标和打乱零假设分位数。
development报告列出六种留出组合的original/object_only配对差值；仅在12份报告齐全时
输出六方向宏平均和胜/平/负数量，不将部分完成当成完整结果，不自动选择赢家。
这些方向共享大量训练数据，不能作为六个独立重复实验直接做显著性声明。
