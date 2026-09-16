# C2-v5：固定训练预算、视频监督、多被试预训练

本轮目标是检验充分训练与额外监督能否改善 **未见视频** 的细粒度连续条件预测。
不再把 ridge 拟合、缓存生成或短诊断称为充分深度训练。代码实现不代表已完成服务器实验，也不保证泛化提升。

## 固定口径

沿用 `prepared_fold1_dim64.pt`：chentianlin、fold1、seed42、前六类、4秒，
312/78/78 个 train/validation/test 视频，每视频三个 session。**不是 holdout_s3 实验**。
所有统计量仅由训练视频拟合；继续使用旧缓存中的训练集 PCA，保持目标空间一致。
原始刺激时序的独立核验仍为 `NOT_VERIFIED`，本轮不自动修正 onset，不改变标签。

| 变体 | 模型与监督 | 参数更新预算 |
|---|---|---|
| long | 深层 EEG 编码器 → Tora PCA 残差；回归、余弦、多正样本对比、防塌缩、session 一致性 | 20,000 次目标被试更新 |
| joint | 相同编码器与主损失，增加有序视频帧及 CLIP 文本辅助监督 | 20,000 次目标被试更新 |
| multisubject | 与 joint 相同；共享编码器＋被试通道仿射适配器 | 15,000 次多被试预训练＋5,000 次目标被试微调 |

每次更新随机无放回取 24 个视频，包含每视频全部三个 session；跨更新重新采样。
默认每组实际执行 20,000 次 `optimizer.step()`，即 480,000 video exposures / 1,440,000 session exposures，
不是独立数据量增加。多被试阶段均匀抽取被试，**包含目标被试**；目标微调阶段只抽取目标被试。
三组匹配总更新数和 batch 大小，但多被试组目标被试曝光量更小，因此不是等目标曝光量的对比。
不会把多被试预训练额外藏在 20,000 次之外。

模型：时域卷积降采样 → 带位置编码的 50 token Transformer（4层、宽256、8头）
→ 64维残差。三个 session 的预测求平均。所有组保留相同辅助头，long 不启用其损失；
共享编码器初始化在相同 seed 下相同。新增的神经网络结构不同于旧 Compact/ridge，
因此不能将 long 对旧结果的差异单独归因于训练时长。

AdamW、lr=2e-4、weight_decay=.01、前1,000次 warmup、余弦衰减到峰值的5%，梯度裁剪1。
全程无早停，每250次更新验证并保存 `last.pt`；按验证集 within-category MRR 保存 `best.pt`。
多被试组只从微调阶段选择最终 best；三个组均不使用测试结果选 checkpoint。
预算结束时的 `last.pt` 与验证选择的 `best.pt` 可能不同，报告同时记录预算和选中 update。

## 视频监督与泄漏边界

使用已下载的 `.ms_video_models/CLIP/clip-vit-base-patch32`，**严格离线**。
每个原视频前4秒均匀取8个有序帧，冻结 CLIP，监督每帧特征及相邻帧特征差分；
同时监督 caption 的 CLIP 文本特征。它保留时间顺序，但不是专门的视频动作编码器，
不能把该辅助任务的通过直接解释为动作恢复成功。

允许缓存全部分区的冻结特征，但训练时只索引 TRAIN 行。
预测函数只接收 EEG 和被试编号；真实视频、真实 caption、真实类别不参与生成预测。
within-category 检索使用真值类别缩小候选，是**诊断指标**，不是端到端检索性能。

多被试缓存逐 session 核对 NPZ 与 metadata 的 video_id 映射、200Hz/800点及通道集合，
按目标被试顺序重排通道；只写入 **全局312个训练视频**。
其他被试的验证/测试视频同样禁止参与归一化和训练。
自动发现模式下，缺 session 的被试会进入 `donors/audit.json` 的 excluded；
数据存在但映射或通道异常时直接报错，不静默跳过。可用 `--subjects 姓名1 姓名2` 固定 donor 列表。
donor 文件使用 CPU memory mapping，按被试采 batch，不把所有数据一次性搬到 GPU。

## 服务器完整运行命令

以下均在 WSL 的仓库执行，不需要访问本机 F: 盘。不覆盖旧实验。

### 1. 同步、测试、dry-run

```bash
cd ~/workspace/diffusion_models_evaluation
git -c http.version=HTTP/1.1 pull --ff-only origin main && git log -1 --oneline
conda run --no-capture-output -n eeg-semantic python -m pytest -q

conda run --no-capture-output -n eeg-semantic python -u scripts/run_c2_deep.py \
  --stage wave --device cuda --updates 20000 --pretrain-updates 15000 \
  --warmup 1000 --batch-size 24 --seed 42 --dry-run
```

dry-run 只显示计划，不检查数据、GPU或模型。若 Git 同步失败，先不要继续运行新入口。
先核对旧 prepared 文件实际存在：

```bash
test -f outputs/eeg_semantic/c2_residual_v2/prepared_fold1_dim64.pt \
  && echo "Prepared cache exists"
```

### 2. 准备离线视频特征与多被试训练缓存

```bash
cd ~/workspace/diffusion_models_evaluation
mkdir -p outputs/eeg_semantic/logs
tmux new-session -d -s c2-v5-prepare bash -lc '
set -eo pipefail
cd "$HOME/workspace/diffusion_models_evaluation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
conda run --no-capture-output -n eeg-semantic python -u scripts/run_c2_deep.py \
  --stage prepare-visual --device cuda 2>&1 | tee outputs/eeg_semantic/logs/c2_v5_visual.log
conda run --no-capture-output -n eeg-semantic python -u scripts/run_c2_deep.py \
  --stage prepare-donors --device cpu 2>&1 | tee outputs/eeg_semantic/logs/c2_v5_donors.log
'
```

检查准备结束（必须两个文件都有，且日志无 traceback）：

```bash
ls -lh outputs/eeg_semantic/c2_deep_v5/visual.pt
cat outputs/eeg_semantic/c2_deep_v5/donors/audit.json
tail -10 outputs/eeg_semantic/logs/c2_v5_visual.log
tail -10 outputs/eeg_semantic/logs/c2_v5_donors.log
```

第一次读取压缩 NPZ 和计算源文件哈希可能较慢，这不是训练阶段。
若失败，按报错检查对应被试；重跑准备会校验并复用同源 donor 文件，但仍需读取源数据进行审计。

### 3. 三组正式训练＋测试（顺序运行，避免显存争抢）

```bash
cd ~/workspace/diffusion_models_evaluation
mkdir -p outputs/eeg_semantic/logs
tmux new-session -d -s c2-v5-train bash -lc '
set -eo pipefail
cd "$HOME/workspace/diffusion_models_evaluation"
export TOKENIZERS_PARALLELISM=false
conda run --no-capture-output -n eeg-semantic python -u scripts/run_c2_deep.py \
  --stage wave --device cuda --updates 20000 --pretrain-updates 15000 \
  --warmup 1000 --batch-size 24 --seed 42 --resume \
  2>&1 | tee -a outputs/eeg_semantic/logs/c2_v5_train.log
'
```

`wave` 预检全部缓存后，依次执行 long、joint、multisubject，各组训练完自动在 test 评估，
最后写 `summary_test_seed42.csv`。不自动开始视频生成，也不基于测试集自动调整训练超参数。
如多被试数据准备遇到问题，可以先单独运行 long：

```bash
conda run --no-capture-output -n eeg-semantic python -u scripts/run_c2_deep.py \
  --stage train --variant long --device cuda --updates 20000 --warmup 1000 \
  --batch-size 24 --seed 42 --resume
```

中断后，确认旧进程已经退出，再重跑同一个训练命令，最多重算最近250次更新。
完成的同配置训练会跳过；改变预算、学习率、数据哈希等会拒绝复用输出，需另设 `--output-root`。
不要同时启动两个命令写同一个输出目录。

### 4. 观察进度和检查完成

```bash
pgrep -af '[r]un_c2_deep.py'
tmux capture-pane -pt c2-v5-train -S -30
tail -30 outputs/eeg_semantic/logs/c2_v5_train.log
nvidia-smi

for variant in long joint multisubject; do
  path="outputs/eeg_semantic/c2_deep_v5/$variant/seed42/completed.json"
  if test -f "$path"; then
    echo "=== $variant ==="
    cat "$path"
  else
    echo "$variant NOT COMPLETE"
  fi
done
```

日志每25次更新输出进度、loss、lr、梯度范数和 updates/sec；验证阶段另外输出 MRR。
`completed.json` 的 `optimizer_updates=20000` 才说明固定预算训练完成，`best.pt` 存在不代表结束。

### 5. 汇总、训练集对照与 Tora 条件导出

```bash
conda run --no-capture-output -n eeg-semantic python -u scripts/run_c2_deep.py \
  --stage summarize --partition test --seed 42

for variant in long joint multisubject; do
  conda run --no-capture-output -n eeg-semantic python -u scripts/run_c2_deep.py \
    --stage evaluate --variant "$variant" --partition train --device cuda --seed 42 || break
done
```

需要进行生成实验时，重新评估并导出 continuous hidden states，无需重训：

```bash
for variant in long joint multisubject; do
  conda run --no-capture-output -n eeg-semantic python -u scripts/run_c2_deep.py \
    --stage evaluate --variant "$variant" --partition test --device cuda --seed 42 --export || break
done
```

每组输出 `test/video_index.jsonl` 和 `test/video_aggregated/*.pt`，保持226×4096 Tora条件契约。
是否真正提升仍需看：held-out residual cosine、MSE相对均值、全局检索、类内匹配对打乱，
以及独立视频生成结果。不能把预测方差增加或训练集高检索率当作已恢复细粒度语义。
本轮 fold1/seed42 仍是探索结果；当前测试集已多次查看，后续需新 folds/seeds 做确认性评估。
