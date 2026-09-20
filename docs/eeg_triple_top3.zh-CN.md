# 固定三个主体的6秒EEG泛化评测

训练仍为01–06双主体4秒EEG，420训练/48验证；测试07/08全部156条三主体6秒EEG。
复用original/object_only、四协议、seed42的已保存预测，不重训，不重新选择checkpoint，
不需要GPU，不读取或优化任何阈值。既有模型按双主体验证AP选checkpoint的规则不变。
这次是改变报告口径后的重新评测，不是新独立实验；结果不应宣称首次盲测。

主分析：完整6秒直接推理的分数。补充：0–4/1–5/2–6秒窗口平均logits。
模型及Session平均均沿用已有预测，不平均原EEG。不挑选测试表现最好的输入方式替代主分析。
从全部六主体中选前三，不限制07/08合法组合，包含person，可能误报car。

主指标为三个主体全部正确率，同时列主体召回率、07/08分项、六主体召回/误报和预测组合。
以视频为单位，全局打乱完整预测向量5000次，报告正确配对减打乱均值及零假设分位数。
同类内打乱因标签相同无效。不将Session/窗口视为新增独立样本，不做自动显著性结论。
随机六选三为5%；固定07/08集合为50%，后者利用测试分布先验，仅作参考。
正确率和打乱差值共同衡量当前设置下的组合泛化，不制造一个加权“泛化总分”。

脚本预检查16份预测文件，正式读取后核验156个ID、标签、主体顺序、有限分数及协议指纹。
旧目录只读，新输出目录存在则拒绝覆盖。JSON记录输入文件哈希、checkpoint哈希和逐视频结果。

```bash
cd ~/workspace/diffusion_models_evaluation
git -c http.version=HTTP/1.1 pull --ff-only origin main
conda run --no-capture-output -n eeg-semantic python -m pytest -q

conda run --no-capture-output -n eeg-semantic python \
  scripts/evaluate_eeg_triple_top3.py \
  --root outputs/eeg_composition_v2 \
  --output-dir outputs/eeg_composition_top3_6s/seed42 \
  --shuffle-repeats 5000 --dry-run

conda run --no-capture-output -n eeg-semantic python -u \
  scripts/evaluate_eeg_triple_top3.py \
  --root outputs/eeg_composition_v2 \
  --output-dir outputs/eeg_composition_top3_6s/seed42 \
  --shuffle-repeats 5000

cat outputs/eeg_composition_top3_6s/seed42/report.zh-CN.md
```

本脚本只使用可信本地torch预测文件，weights_only加载，不读取服务器外部任意checkpoint。
当前限制：单被试、单seed、刺激独立时序核验NOT_VERIFIED、07/08已用于探索。

## 与论文前文一致的紧凑表格

```bash
conda run --no-capture-output -n eeg-semantic python \
  scripts/build_three_entity_paper_table.py \
  --root outputs/eeg_composition_v2 \
  --output-dir outputs/eeg_composition_paper/seed42 --dry-run

conda run --no-capture-output -n eeg-semantic python \
  scripts/build_three_entity_paper_table.py \
  --root outputs/eeg_composition_v2 \
  --output-dir outputs/eeg_composition_paper/seed42

cat outputs/eeg_composition_paper/seed42/section.tex
```

无需训练或GPU，只读取8份完整6秒预测。表格只有4行数据：Session-Average和
Cross-Session各两种训练目标，不列逐session、滑窗或打乱对照。7个指标为Macro AP、
Micro AP、Macro AUC、Micro AUC、Top-3 F1、Set Accuracy、Jaccard。
Macro AP/AUC仅含dog、ball、flower、bird：person全正、car全负，无正负排序意义。
Micro AP/AUC展开全部视频×六标签计算，仍包含person/car，可能受标签先验影响；
所有集合指标也包含六标签，Jaccard为逐视频交并比的平均，不是先平均F1再换算。
跨session先分别计算指标后平均三方向，不合并概率，也不生成误导性的被试间标准差。
单被试、单seed，不与前文20被试结果等同。表注明确logit平均不同于前文信号平均。
输出metrics.json（含8方向原始指标及输入哈希）、table.tex、section.tex。
表格使用booktabs、两层表头、紧凑列距，不使用resizebox强制放大；需在用户论文模板中编译检查最终宽度。
旧输出存在则拒绝覆盖。此处不提供未经验证的chance AP/AUC或Jaccard占位值。
