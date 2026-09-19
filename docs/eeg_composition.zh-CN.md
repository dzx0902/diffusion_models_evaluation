# EEG2Caption：01–06训练，07–08三主体组合泛化

任务不是训练八分类器，也不是把六分类输出重新命名成07/08。
复用 `EEG2Caption/src/common.py` 的 CompactEEGClassifier，用六维 object head
识别 person/dog/car/ball/flower/bird。07=person+dog+ball，08=person+bird+flower。

## 冻结口径

- 单被试，默认 chentianlin；每类78视频，共624。三个session按video_id对齐。
- 01–06每类70训练、8验证，共420/48；07–08全部156视频测试。split seed固定42。
- 训练seed42/43/44只改变初始化/训练随机性，不改变视频划分。
- cs_s1/cs_s2/cs_s3：其余两个session训练和验证，被留出的一个session测试。
- session_average：三个session训练；测试逐session预测，然后平均 **object logits**。
  报告同时保存三个单session结果；不读取原始 `session_average` 数据。
- 归一化采用训练视频、允许训练session、前800点拟合的共享逐通道均值/标准差。
  这是为跨session留出适配的明确改动，不从被留出的session拟合其专属归一化。
- 主测试取前4秒；补充测试取完整6秒，使用同一个checkpoint及阈值，不重训、不重采样。
  6秒结果同时包含输入长度分布变化；前4秒未必覆盖所有主体最明显的出现时段。
- 原始切片与刺激时序独立核验仍为 NOT_VERIFIED；本实验不擅自移动onset。

## 模型与训练

`original`：原Compact编码器、6维object head、6类pair辅助head；逐session及融合BCE、
pair CE权重.5、归一化特征跨session一致性权重.05。只用object head测试三主体。
`object_only`：仅去掉pair CE，保持其他配置一致（仍有融合BCE和一致性项）。
保留pair模块但不参与该消融的loss，确保编码器初始化一致。

默认100 epochs，无早停；batch32，AdamW lr1e-3、weight_decay1e-4、cosine到5%峰值，
dropout.35、标准化后噪声.02、20点时间mask、梯度裁剪5。
默认FP32，不宣称与原始AMP实现逐位一致；本实现按batch共享一次随机mask位置。
按01–06验证集允许session的平均logits macro-AP选择best，不看07/08。
日志记录实际optimizer updates；last保存优化器、scheduler、随机状态和历史以支持续训。

## 解码与统计

- Top-3：在全部 `C(6,3)=20` 种集合中解码；只使用“有三个主体”的任务先验，
  不限制只能输出07或08，不声称预测了主体数量。
- 阈值：在01–06验证集上从.10到.90、步长.05，选择global micro-F1最优阈值；
  并列时取更接近.5的阈值。所有测试session/长度复用该阈值，允许0到6个预测主体。
- 主指标：Top-3 exact/recall，阈值exact/micro-F1，07/08分项，完整六维概率。
- 测试person全正、car全负：这两项AP标为null。informative_macro_ap只含
  dog/ball/flower/bird四个有正负样本的标签；单列person recall与car false-positive rate。
  macro_f1_six_labels按六标签固定平均（零分母置0），不要与四标签macro-AP混淆。
- 均匀随机Top-3 exact=5%、recall=50%；另报固定预测07、固定预测08的exact=50%基线。
  不能仅超过5%就宣称学会组合泛化。
- 所有记录以视频为单位。同视频的session/重复seed不是独立视频，汇总保留方向与seed，
  不把468次session预测误报为468个独立测试视频，不自动做显著性宣称。

## 服务器命令

```bash
cd ~/workspace/diffusion_models_evaluation
git -c http.version=HTTP/1.1 pull --ff-only origin main && git log -1 --oneline
conda run --no-capture-output -n eeg-semantic python -m pytest -q

conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_eeg_composition.py --stage prepare --subject chentianlin

conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_eeg_composition.py --stage wave --subject chentianlin \
  --variants original --seeds 42 --device cuda --resume --dry-run
```

prepare检查NPZ/metadata映射、通道名顺序、200Hz、800/1200有效点、无padding/非有限数据，
保存源文件哈希。输出 `outputs/eeg_composition/chentianlin/prepared.pt` 和 `prepared.json`。
缓存内4秒视频末尾400点为存储补零，但训练/验证明确只读前800点；6秒只用于07/08测试。
旧缓存存在时不会覆盖，不需重复prepare。dry-run只打印命令，不检查CUDA或执行训练。

首轮四个协议、seed42，自动训练与测试：

```bash
mkdir -p outputs/eeg_semantic/logs
tmux new-session -d -s eeg-composition bash -lc '
set -eo pipefail
cd "$HOME/workspace/diffusion_models_evaluation"
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_eeg_composition.py --stage wave --subject chentianlin \
  --variants original --seeds 42 --device cuda --resume \
  2>&1 | tee -a outputs/eeg_semantic/logs/eeg_composition_pilot.log
'
```

检查日志和结果：

```bash
tail -n 30 -F outputs/eeg_semantic/logs/eeg_composition_pilot.log
# 或
tmux capture-pane -pt eeg-composition -S -40

conda run --no-capture-output -n eeg-semantic python \
  scripts/run_eeg_composition.py --stage summarize --subject chentianlin \
  --variants original --seeds 42
```

正式24组训练（4协议×2损失×3seeds；已有同配置original/42会跳过训练）：

```bash
tmux new-session -d -s eeg-composition-full bash -lc '
set -eo pipefail
cd "$HOME/workspace/diffusion_models_evaluation"
conda run --no-capture-output -n eeg-semantic python -u \
  scripts/run_eeg_composition.py --stage wave --subject chentianlin \
  --variants original object_only --seeds 42 43 44 --device cuda --resume \
  2>&1 | tee -a outputs/eeg_semantic/logs/eeg_composition_full.log
'
```

不要与pilot同时向相同目录运行。改变预算/配置时用新output-root，并通过--prepared复用缓存。
每个run目录：`<subject>/<protocol>/<variant>/seedN/`，含best/last、history、completed、
report.json及predictions/*.json/*.pt。顶层summary.csv逐协议/seed/session/时长报告。
每次summarize只汇总指定variants/seeds，覆盖顶层summary.csv；完整原始报告保留。
本轮不生成caption/视频，不使用C2/PCA/T5缓存，不修改历史实验结果。
