"""Create the formal Chinese EEG ablation report, optionally with video metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.ablation_statistics import aggregate_long_metrics, paired_comparisons


VIDEO_GROUPS = {
    "full8_caption": {
        "baseline": "a_base",
        "variants": {"a_base", "a_enhanced", "b_base", "b_enhanced"},
    },
    "first6_temporal": {
        "baseline": "a_4s_first6",
        "variants": {"a_4s_first6", "a_2s2_first6", "a_1s4_first6"},
    },
    "full8_tora": {
        "baseline": "c2_mse",
        "variants": {"c2_mse", "c2_full"},
    },
}
VIDEO_METRIC_NOTES = {
    "semantic_clip_score": "整体语义一致性（越高越好）",
    "subject_clip_score": "主体一致性（越高越好）",
    "object_clip_score": "客体一致性（越高越好）",
    "action_clip_score": "动作一致性（越高越好；依赖派生动作标签）",
    "temporal_consistency": "相邻帧一致性（需与 motion energy 联合解释）",
    "motion_energy": "运动幅度（不是单调越高越好）",
    "sharpness_score": "清晰度诊断（仅在同一生成器内比较）",
    "exposure_valid_rate": "曝光有效率（越高越好）",
    "trajectory_direction_score": "固定轨迹方向符合度（越高越好）",
}
REQUIRED_VIDEO = {
    "variant", "generator", "subject", "fold", "seed",
    "generation_seed", "video_id", "metric", "value",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-results", type=Path, required=True)
    parser.add_argument("--video-metrics", type=Path, default=None)
    parser.add_argument("--generation-audits", type=Path, nargs="*", default=())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> str:
    return "—" if value is None or value == "" else f"{float(value):.4f}"


def phase1_sections(payload: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    rows = payload["rows"]
    primary = [row for row in rows if bool(row.get("primary_analysis", True))]
    lines = [
        "## 1. EEG 条件解码实验", "",
        "### 1.1 实验协议", "",
        f"- 被试：`{payload['subject']}`；fold：`{payload['fold']}`；训练 seed：`{payload['training_seed']}`。",
        "- `full8_624` 使用类别 01--08，fold-1 test 为 104 个视频。",
        "- `first6_468` 仅使用类别 01--06，fold-1 test 为 78 个视频。",
        "- 所有 checkpoint、hybrid alpha 和 temporal decoder 只由 validation 选择。",
        "- 两种数据范围以及离散分类与连续 Tora retrieval 不进行跨口径排名。", "",
        "### 1.2 Primary results", "",
        "| 口径 | 方法组 | 变体 | 解码器 | Test n | 主指标 | 结果 | 随机基线 | Object exact |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in primary:
        lines.append(
            f"| {row['scope']} | {row['comparable_group']} | {row['variant']} | "
            f"{row['decoder']} | {row['test_videos']} | {row['primary_metric']} | "
            f"{number(row['primary_value'])} | {number(row['chance'])} | "
            f"{number(row['object_exact'])} |"
        )
    lines.extend(["", "### 1.3 Exploratory decoder audit", ""])
    exploratory = [row for row in rows if not bool(row.get("primary_analysis", True))]
    if exploratory:
        lines.extend([
            "以下结果用于解释解码机制，不用于根据 test 重新选择方法。", "",
            "| 变体 | 解码器 | Category accuracy | Object exact | Invalid-pair rate |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for row in exploratory:
            lines.append(
                f"| {row['variant']} | {row['decoder']} | {number(row['primary_value'])} | "
                f"{number(row['object_exact'])} | {number(row['invalid_pair_rate'])} |"
            )
    else:
        lines.append("无 exploratory decoder 结果。")
    lines.extend(["", "### 1.4 分组结论", ""])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in primary:
        grouped.setdefault(row["comparable_group"], []).append(row)
    for group, values in grouped.items():
        best = max(values, key=lambda row: float(row["primary_value"]))
        delta = float(best["primary_value"]) - float(best["chance"])
        lines.append(
            f"- `{group}`：primary 最优为 `{best['variant']} / {best['decoder']}`，"
            f"{best['primary_metric']}={number(best['primary_value'])}，"
            f"相对基线差值 {delta:+.4f}。"
        )
    lines.extend([
        "", "时间分段结果显示，完整 4 秒上下文优于 2 秒和 1 秒窗口。无约束 Top-2 "
        "会产生非法实体组合；valid-pair 与 hybrid 虽能约束输出，但未稳定超过完整时长的 "
        "mean-logit category decoder。", "",
    ])
    return lines, rows


def phase2_sections(
    video_path: Path | None,
    audits: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    lines = ["## 2. 生成视频层面实验", ""]
    audit_status = "PENDING"
    if audits:
        audit_status = "PASS" if all(value.get("status") == "PASS" for value in audits) else "FAIL"
        lines.append(f"Generation protocol audit：**{audit_status}**。")
        for index, value in enumerate(audits, 1):
            lines.append(
                f"- audit {index}: status={value.get('status')}，jobs={value.get('jobs')}，"
                f"matched_cells={value.get('matched_cells')}。"
            )
        lines.append("")
    if video_path is None:
        lines.extend([
            "状态：**PENDING（视频尚未完成统一评估）**。", "",
            "视频生成完成后必须按以下五个维度报告：", "",
            "1. 语义：overall / subject / object / action CLIP score；",
            "2. 时序：temporal consistency，并与 motion energy 联合解释；",
            "3. 画质：sharpness 与 exposure valid rate；",
            "4. 控制：Tora fixed-trajectory direction score；",
            "5. 稳健性：每个视频三个 generation seeds，按视频和 seed 配对。", "",
            "所有模型分辨率、帧数和 fps 不完全相同，因此画质指标只能在同一 generator 内"
            "比较；跨 generator 只作描述性展示。", "",
        ])
        return lines, {"status": "PENDING", "audit_status": audit_status}
    rows = read_csv(video_path)
    if not rows or not REQUIRED_VIDEO.issubset(rows[0]):
        raise ValueError(f"{video_path} lacks required long-form video fields")
    summary = aggregate_long_metrics(rows)
    comparisons: list[dict[str, Any]] = []
    present = {str(row["variant"]) for row in rows}
    for group, config in VIDEO_GROUPS.items():
        variants = present & config["variants"]
        if config["baseline"] not in variants or len(variants) < 2:
            continue
        selected = [row for row in rows if str(row["variant"]) in variants]
        for value in paired_comparisons(selected, config["baseline"]):
            comparisons.append({"comparison_group": group, **value})
    write_csv(output_dir / "video_summary.csv", summary)
    write_csv(output_dir / "video_paired_tests.csv", comparisons)
    metric_counts = Counter(str(row["metric"]) for row in rows)
    lines.extend([
        f"状态：**COMPLETE**；long-form observations={len(rows)}。", "",
        "### 2.1 指标定义", "",
    ])
    for metric in sorted(metric_counts):
        lines.append(
            f"- `{metric}`：{VIDEO_METRIC_NOTES.get(metric, '补充诊断指标')}；"
            f"n={metric_counts[metric]}。"
        )
    lines.extend([
        "", "### 2.2 汇总结果", "",
        "| Variant | Generator | Metric | n | Mean ± SD |",
        "| --- | --- | --- | ---: | ---: |",
    ])
    for row in summary:
        lines.append(
            f"| {row['variant']} | {row['generator']} | {row['metric']} | {row['n']} | "
            f"{float(row['mean']):.4f} ± {float(row['std']):.4f} |"
        )
    lines.extend([
        "", "### 2.3 配对检验", "",
        "配对单位为 subject/fold/training-seed/generation-seed/video；使用双侧 sign-flip "
        "randomization test，并报告 Holm 校正。", "",
        "| Group | Baseline | Variant | Generator | Metric | n | Δ mean | dz | p | p-Holm |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in comparisons:
        lines.append(
            f"| {row['comparison_group']} | {row['baseline']} | {row['variant']} | "
            f"{row['generator']} | {row['metric']} | {row['n']} | "
            f"{float(row['mean_difference']):.4f} | {float(row['cohen_dz']):.4f} | "
            f"{float(row['p_value']):.4f} | {float(row['p_holm']):.4f} |"
        )
    lines.append("")
    return lines, {
        "status": "COMPLETE", "audit_status": audit_status,
        "observations": len(rows), "summary_rows": len(summary),
        "paired_comparisons": len(comparisons),
    }


def main() -> None:
    args = parse_args()
    semantic = json.loads(args.semantic_results.read_text(encoding="utf-8"))
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in args.generation_audits]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase1, phase1_rows = phase1_sections(semantic)
    phase2, phase2_status = phase2_sections(args.video_metrics, audits, args.output_dir)
    lines = [
        "# 多方法 EEG 视觉重建消融实验正式报告", "",
        "- 报告版本：fold-1 / seed-42 provisional report",
        "- 当前范围：EEG 条件解码已完成；视频章节根据输入指标自动标记 PENDING 或 COMPLETE。",
        "- 可复现产物：机器可读 JSON/CSV、逐视频指标、生成 manifest 与协议审计。", "",
        "## 摘要", "",
        "本实验比较模板 caption、结构化 caption、EEG→Tora 连续条件以及不同时间窗口"
        "的 EEG2Caption 分类。当前结论仅覆盖单被试、单 fold、单训练 seed，不能替代"
        "六 folds、多 seeds 的正式统计。", "",
        *phase1,
        *phase2,
        "## 3. 有效性威胁与报告边界", "",
        "- 624 条语义标签仍标记为 derived-needs-audit；动作和关系结论为 provisional。",
        "- first6 与 full8 的样本范围不同，不进行跨范围优劣宣称。",
        "- 单 fold / 单训练 seed 结果仅用于工程选择；论文结论需扩展六 folds 和多 seeds。",
        "- 视频评价中的 CLIP 属代理指标，应配合盲法人工偏好评估。",
        "- temporal consistency 可能偏好静态视频，必须与 motion energy 和动作语义共同解释。",
        "- 不同生成器输出规格不同，禁止直接用 sharpness 做跨生成器排名。", "",
        "## 4. 后续冻结方案", "",
        "1. 冻结 condition/caption 文件 SHA256、训练 checkpoint、生成参数与 generation seeds；",
        "2. 每个比较组使用完全一致的视频集合、generator 和 generation seed；",
        "3. 生成后先执行 generation protocol audit，FAIL 时禁止统计；",
        "4. 完成视频 long-form metrics 后重新运行本脚本，形成 COMPLETE 版报告；",
        "5. 最终扩展六 folds / 多训练 seeds，并增加盲法人工评价。", "",
    ]
    report_path = args.output_dir / "formal_report.zh-CN.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    metadata = {
        "schema_version": 1, "semantic_results": str(args.semantic_results),
        "phase1_rows": len(phase1_rows), "phase2": phase2_status,
        "generation_audits": [str(path) for path in args.generation_audits],
        "report": str(report_path),
    }
    (args.output_dir / "formal_report.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
