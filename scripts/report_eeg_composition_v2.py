"""Read-only composition diagnostics; does not load EEG, weights, or run inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


OBJECTS = ("person", "dog", "car", "ball", "flower", "bird")
PROTOCOLS = ("cs_s1", "cs_s2", "cs_s3", "session_average")


def number(value):
    return "NA" if value is None else f"{value:.4f}"


def read_report(path, protocol, variant, seed, development):
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    signature = report["signature"]
    if (signature["protocol"], signature["variant"], signature["seed"], report["development_only"]) != (
            protocol, variant, seed, development):
        raise ValueError(f"Report identity mismatch: {path}")
    return report


def test_details(root, subject, seed, protocols):
    found = 0
    print("=== Exploratory triple test: checkpoint-selected on pair validation ===")
    for protocol in protocols:
        group = "session_average" if protocol == "session_average" else "session" + protocol[-1]
        for stage, variant in (("audit-baseline", "original"), ("object-only", "object_only")):
            path = root / stage / subject / protocol / variant / f"seed{seed}" / "audit.json"
            report = read_report(path, protocol, variant, seed, False)
            if report is None:
                print("MISSING:", path)
                continue
            found += 1
            print(f"\n{protocol}/{variant}/seed{seed} epoch={report['checkpoint_epoch']} threshold={report['threshold']}")
            for mode in ("4s", "6s", "6s_windows"):
                r = report["reports"][f"{mode}/{group}"]
                print(f"  {mode}: n={r['video_count']} exact={number(r['topk_metrics']['exact_set_accuracy'])} "
                      f"recall={number(r['topk_metrics']['micro_recall'])} AP={number(r['informative_macro_ap'])}")
                for category in ("07", "08"):
                    c = r["by_category"][category]
                    print(f"    class {category}: n={c['video_count']} exact={number(c['topk_metrics']['exact_set_accuracy'])}")
                s = r["shuffle"]
                print(f"    shuffle: mean={number(s['shuffled_exact_mean'])} "
                      f"null_q025={number(s['shuffled_exact_q025'])} null_q975={number(s['shuffled_exact_q975'])} "
                      f"matched_delta={number(s['matched_minus_shuffled'])}")
                print("    object    topk_recall topk_FPR threshold_recall threshold_FPR pos_score neg_score")
                for obj in OBJECTS:
                    item = r["per_object"][obj]
                    fields = ("topk_recall", "topk_false_positive_rate", "recall", "false_positive_rate",
                              "positive_mean_score", "negative_mean_score")
                    print(f"    {obj:9s} " + " ".join(number(item[k]) for k in fields))
    print(f"\nTriple audit files found: {found}/{len(protocols)*2}")
    print("Shuffle quantiles are null-reference quantiles, not confidence intervals or corrected significance tests.")


def development_details(root, subject, seed, protocols):
    print("\n=== Development only: held-pair validation, NOT an independent triple test ===")
    for protocol in protocols:
        group = "session_average" if protocol == "session_average" else "session" + protocol[-1]
        paired = []
        found = 0
        for category in ("01", "02", "03", "04", "05", "06"):
            values = {}
            for variant in ("original", "object_only"):
                path = root / "development" / f"held_{category}" / subject / protocol / variant / f"seed{seed}" / "audit.json"
                report = read_report(path, protocol, variant, seed, True)
                if report is None:
                    print(f"MISSING: {protocol}/held_{category}/{variant}/seed{seed}")
                    continue
                found += 1
                r = report["reports"][f"4s/{group}"]
                values[variant] = r["topk_metrics"]["exact_set_accuracy"]
                print(f"{protocol} held={category} {variant:11s} n={r['video_count']} "
                      f"epoch={report['checkpoint_epoch']} top2_exact={number(values[variant])} "
                      f"top2_recall={number(r['topk_metrics']['micro_recall'])}")
            if len(values) == 2:
                paired.append((values["original"], values["object_only"]))
                print(f"  object_only_minus_original={values['object_only']-values['original']:+.4f}")
        print(f"{protocol}: reports={found}/12, paired_held_categories={len(paired)}/6")
        if len(paired) == 6:
            print(f"  six-category macro exact: original={mean(a for a,b in paired):.4f} "
                  f"object_only={mean(b for a,b in paired):.4f}; "
                  f"wins/ties/losses={sum(b>a for a,b in paired)}/{sum(b==a for a,b in paired)}/{sum(b<a for a,b in paired)}")
        else:
            print("  INCOMPLETE: no six-category aggregate emitted.")
    print("Single held-category AP is undefined. Session-average development scores reuse checkpoint-selection data.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/eeg_composition_v2"))
    parser.add_argument("--subject", default="chentianlin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--section", choices=("test", "development", "all"), default="all")
    parser.add_argument("--protocols", choices=PROTOCOLS, nargs="+", default=None)
    args = parser.parse_args()
    if args.section in ("test", "all"):
        test_details(args.root, args.subject, args.seed, args.protocols or PROTOCOLS)
    if args.section in ("development", "all"):
        development_details(args.root, args.subject, args.seed, args.protocols or ["session_average"])


if __name__ == "__main__":
    main()
