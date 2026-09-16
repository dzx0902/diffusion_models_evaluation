# C2 训练样本重建诊断

本实验检验：已经拟合的训练 EEG 条件，能否驱动 Tora 生成对应的语义内容。
**不是未见视频泛化实验，不应与测试集重建指标混为正式主结果。**

使用 C2-v5 long 的 validation-selected best.pt。每类在训练分区按 video_id 升序选两条
caption 不同的视频，共12条；同类两条互换 EEG 条件。选择不依赖预测质量或生成效果，
结果不佳也不替换样本。

四组均使用已跑通的 `tora_injected` 后端：

| 组 | 条件 |
|---|---|
| text_full | 原始 caption 对应的完整 T5 缓存 |
| text_pca | 原始 caption 经 token PCA 与 video residual PCA 后恢复的条件 |
| eeg_matched | 当前视频 EEG 预测的条件 |
| eeg_swapped | 同类另一条选中视频 EEG 预测的条件 |

所有组共享同一条合成静止轨迹（49个 `128,128` 点），**不使用真实视频提取的轨迹**。
这是为了保持现有 Tora 接口和控制变量，不等于完全无轨迹，也可能限制运动表现。
不计算该静止轨迹的方向一致性指标。视频49帧、12fps，首末帧间隔4秒，容器时长约4.08秒。
默认 generation seed index=0，实际采样随机种子沿用管线规则 training_seed42+0=42。
共48个视频，不自动扩展至全部训练样本或多个生成器。

## WSL 完整命令

同步成功后再执行后续命令：

```bash
cd ~/workspace/diffusion_models_evaluation
git -c http.version=HTTP/1.1 pull --ff-only origin main && git log -1 --oneline
conda run --no-capture-output -n eeg-semantic python -m pytest -q

conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_c2_reconstruction_diagnostic.py --stage prepare --device cuda

conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_c2_reconstruction_diagnostic.py --stage generate --seeds 0 --dry-run

cat outputs/eeg_semantic/c2_train_reconstruction/protocol.json
```

prepare 不重训，仅对固定12条训练 EEG 推理，并导出48个条件文件。
需要已有 `outputs/tora/text_cache/index.jsonl` 及其引用的原始完整T5缓存。
dry-run 写独立 `dry_run_manifest.jsonl`，不会覆盖正式生成记录，也不验证GPU环境能否成功生成。

后台生成并评估：

```bash
cd ~/workspace/diffusion_models_evaluation
mkdir -p outputs/eeg_semantic/logs
tmux new-session -d -s c2-train-recon bash -lc '
set -eo pipefail
cd "$HOME/workspace/diffusion_models_evaluation"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_c2_reconstruction_diagnostic.py --stage generate --seeds 0 \
  2>&1 | tee -a outputs/eeg_semantic/logs/c2_train_recon_generate.log
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_c2_reconstruction_diagnostic.py --stage evaluate --device cuda \
  2>&1 | tee outputs/eeg_semantic/logs/c2_train_recon_evaluate.log
'
```

生成使用 tora 环境，EEG 推理和视频指标使用 eeg-semantic 环境。CLIP 权重使用本地目录。
中断后确认旧进程退出，再重跑：相同配置的非空已完成mp4会复用；改变条件或生成参数会拒绝混用。
新视频采用 `.partial.mp4` → 成功后重命名。不要同时向同一个输出目录启动两个任务。

观察和检查：

```bash
tail -n 40 -F outputs/eeg_semantic/logs/c2_train_recon_generate.log
# 或者查看生成/评估当前阶段：
tmux capture-pane -pt c2-train-recon -S -40

find outputs/eeg_semantic/c2_train_reconstruction/generated \
  -type f -name '*_seed0.mp4' ! -name '*.partial.mp4' | wc -l

ls -lh outputs/eeg_semantic/c2_train_reconstruction/evaluation/video_metrics.csv
```

预期48个视频。指标文件使用现有 long-form schema；缺帧/解码失败不会伪装为成功指标。
人工查看按 protocol.json 的 caption 和 swapped_caption 并排比较四组，重点记录主体、对象、
动作、关系与背景，不只看画面美观或 CLIP 分数。

解释顺序：先看 text_full 是否表达目标；再看 text_pca 的压缩损失；再看 eeg_matched
能否接近 text_pca；最后看 eeg_swapped 是否随来源 caption 改变。
即使 matched 显著好于 swapped，也只支持训练样本条件区分有效，不排除样本记忆。
独立刺激时序核验仍未完成。跨 session 泛化需另开只用 session1/2 训练的实验。
