from __future__ import annotations

import sys
import subprocess
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
    matrix, jobs = materialize_jobs(
        ROOT / "configs/eeg_semantic/ablation_matrix.yaml", ROOT, tmp_path
    )
    protocol = matrix["protocol"]
    assert len(protocol["caption_generators"]) == 7
    assert "hunyuanvideo_1_5" not in protocol["caption_generators"]
    assert_matched_protocol(jobs)
    assert {job.variant for job in jobs} >= {"a_base", "b_base", "c1_mse", "c2_full"}
    c2 = next(job for job in jobs if job.variant == "c2_full")
    config = yaml.safe_load(c2.config_path.read_text(encoding="utf-8"))
    assert config["data"]["fold"] == "video_6fold_1"
    assert config["data"]["tora_target_index"].endswith("pca/fold1/dim512/index.jsonl")
    assert config["experiment"]["output_dir"].endswith("c2_full/chentianlin/video_6fold_1/seed42")
    assert config["model"]["implementation"] == "eeg2caption_compact"
    assert "runs_compact/c2_full" in config["experiment"]["output_dir"]
    assert config["augmentation"] == {"noise_std": 0.025, "time_mask_samples": 30}
    a_base = next(job for job in jobs if job.variant == "a_base")
    a_config = yaml.safe_load(a_base.config_path.read_text(encoding="utf-8"))
    assert a_config["model"]["implementation"] == "eeg2caption_compact"
    assert "runs_eeg2caption/a_base" in a_config["experiment"]["output_dir"]
    temporal = {
        job.variant: yaml.safe_load(job.config_path.read_text(encoding="utf-8"))
        for job in jobs if job.variant in {"a_4s_first6", "a_2s2_first6", "a_1s4_first6"}
    }
    assert {key: value["model"]["segment_samples"] for key, value in temporal.items()} == {
        "a_4s_first6": 800, "a_2s2_first6": 400, "a_1s4_first6": 200,
    }
    assert all(value["experiment"]["method"] == "temporal_category" for value in temporal.values())
    assert all(value["data"]["allowed_categories"] == ["01", "02", "03", "04", "05", "06"] for value in temporal.values())


def test_temporal_generation_uses_validation_selected_predictions() -> None:
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/run_eeg_semantic_ablation.py"),
            "--stage", "generate", "--variants", "a_1s4_first6", "--dry-run",
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert "temporal_decoding" in completed.stdout
    assert "selected_predictions.json" in completed.stdout
