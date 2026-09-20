# 多被试三主体组合泛化

默认固定5名：chentianlin、duzhuoxuan、fangzikai、luotianming、niezhiheng。
本机caiyuxiang/chengjiajie只有Session1，未纳入；名单在新实验运行前按三Session可用性确定。
这是运行前固定的可用性队列，不按测试表现选人；服务器若缺任一人的数据会报错，不能静默跳过。
源文件路径通过dry-run检查；完整624视频、62通道、200Hz、800/1200有效点由prepare检查。

每名被试独立训练Original/Object-only四个协议，共40组，seed42，100epochs，batch32，
lr.001。所有人共享split seed42及同一视频ID划分（420双主体训练、48双主体验证、156三主体测试）。
不是跨被试训练/迁移，不读取留出session的归一化统计。4s训练、完整6s测试，固定六选三。
无阈值、无滑窗，也不跑留组合开发或额外seed。原先单被试实验不覆盖、不混入新目录。
为避免新旧配置混合，当前队列内chentianlin也在新目录按相同预算重跑。

Session-Average先对同视频三个session的logits平均，再算7指标。
Cross-Session先分别算三个留出方向的7指标，再在被试内部平均。
最后对5名被试的指标计算算术均值和样本std（ddof=1），不把15个session或780视频当作独立被试。
Macro AP/AUC仍只含4个有正负标签的主体；其他指标含全部6主体。
不能写成20被试结果，也不把被试间标准差当成置信区间或显著性检验。

```bash
cd ~/workspace/diffusion_models_evaluation
git -c http.version=HTTP/1.1 pull --ff-only origin main
conda run --no-capture-output -n eeg-semantic python -m pytest -q

conda run --no-capture-output -n eeg-semantic python \
  scripts/run_three_entity_multisubject.py \
  --subjects chentianlin duzhuoxuan fangzikai luotianming niezhiheng \
  --device cuda --resume --dry-run

mkdir -p outputs/eeg_semantic/logs
tmux new-session -d -s triple-multisubject bash -lc '
set -eo pipefail
cd "$HOME/workspace/diffusion_models_evaluation"
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_three_entity_multisubject.py \
  --subjects chentianlin duzhuoxuan fangzikai luotianming niezhiheng \
  --device cuda --resume \
  2>&1 | tee -a outputs/eeg_semantic/logs/triple_multisubject.log
'

tail -n 30 -F outputs/eeg_semantic/logs/triple_multisubject.log
```

所有任务串行，每个训练/测试子进程结束释放内存；失败立即停止，可用相同--resume命令续跑。
首次保存plan.json固定名单和预算；改变名单/预算必须用新--output-root。
已完成同配置训练会跳过，测试可重算。输出全在outputs/eeg_composition_multisubject/：
prepared/<subject>.pt、runs/<subject>/<protocol>/<variant>/seed42/、report/。
8方向结果缺任一项不会产出不完整的均值/std。

```bash
conda run --no-capture-output -n eeg-semantic python \
  scripts/run_three_entity_multisubject.py --stage summarize \
  --subjects chentianlin duzhuoxuan fangzikai luotianming niezhiheng

cat outputs/eeg_composition_multisubject/report/table.tex
cat outputs/eeg_composition_multisubject/report/metrics.json
```

表包含原论文7个指标mean±std和chance行，不拆逐session，不加dagger，不强制表头换行。
mean±std较宽，论文双栏模板可放table*；需在最终模板中编译检查，不伪造被试数。
数据独立时序核验仍NOT_VERIFIED；既有被试被探索过，应如实标为探索性扩展。
