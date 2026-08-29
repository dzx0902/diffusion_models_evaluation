from __future__ import annotations

import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.ablation_matrix import assert_matched_protocol, deep_merge, materialize_jobs


def test_deep_merge_preserves_unmodified_nested_values() -> None:
    merged = deep_merge({"loss": {"mse": 1.0, "cosine": 0.2}}, {"loss": {"cosine": 0.0}})
    assert merged == {"loss": {"mse": 1.0, "cosine": 0.0}}


def test_materialized_matrix_is_matched_and_resolves_fold_paths(tmp_path: Path) -> None:
    _, jobs = materialize_jobs(
        ROOT / "configs/eeg_semantic/ablation_matrix.yaml", ROOT, tmp_path
    )
    assert_matched_protocol(jobs)
    assert {job.variant for job in jobs} >= {"a_base", "b_base", "c1_mse", "c2_full"}
    c2 = next(job for job in jobs if job.variant == "c2_full")
    config = yaml.safe_load(c2.config_path.read_text(encoding="utf-8"))
    assert config["data"]["fold"] == "video_6fold_1"
    assert config["data"]["tora_target_index"].endswith("pca/fold1/dim512/index.jsonl")
    assert config["experiment"]["output_dir"].endswith("c2_full/chentianlin/video_6fold_1/seed42")
    a_base = next(job for job in jobs if job.variant == "a_base")
    a_config = yaml.safe_load(a_base.config_path.read_text(encoding="utf-8"))
    assert a_config["model"]["implementation"] == "eeg2caption_compact"
    assert "runs_eeg2caption/a_base" in a_config["experiment"]["output_dir"]
