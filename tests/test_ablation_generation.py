from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.ablation_generation import (
    AblationGenerator,
    load_generators,
    read_conditions,
    run_generation_matrix,
)


def test_server_generator_config_has_caption_and_tora_routes() -> None:
    models = load_generators(ROOT / "configs/eeg_semantic/generators.server.yaml")
    assert "caption" in models["tora"].condition_kinds
    assert models["tora"].requires_trajectory
    assert models["tora_injected"].condition_kinds == ("tora_state",)
    assert len(models) >= 8
    hunyuan = models["hunyuanvideo_1_5"].command
    assert hunyuan[:7] == (
        "conda", "run", "-n", "hunyuanvideo15", "python", "-m", "torch.distributed.run"
    )
    assert any(path.endswith("ckpts/vision_encoder/siglip") for path in models["hunyuanvideo_1_5"].required_paths)
    caption_models = [key for key, value in models.items() if "caption" in value.condition_kinds]
    records = run_generation_matrix(
        [{"video_id": "01-001", "prompt": "A person holds a ball."}],
        "caption", models, caption_models, [0], ROOT / "unused",
        {"repo_root": str(ROOT), "models_root": "/models", "home": "/home/test"},
        {"01-001": "/trajectories/01-001.txt"}, dry_run=True,
    )
    assert len(records) == len(caption_models)
    latent = run_generation_matrix(
        [{"video_id": "01-001", "condition": "/conditions/01-001.pt"}],
        "tora_state", models, ["tora_injected"], [0], ROOT / "unused",
        {"repo_root": str(ROOT), "models_root": "/models", "home": "/home/test"},
        {"01-001": "/trajectories/01-001.txt"}, dry_run=True,
    )
    assert latent[0]["status"] == "dry_run"


def test_read_caption_predictions_and_build_dry_run(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {"video_aggregation_records": [{"video_id": "01-001", "caption": "A ball moves."}]}
        ),
        encoding="utf-8",
    )
    rows = read_conditions(predictions, "caption")
    spec = AblationGenerator(
        id="demo",
        condition_kinds=("caption",),
        command=(
            "python", "generate.py", "--prompt", "{prompt}", "--output", "{output}",
            "--seed", "{seed}",
        ),
    )
    result = run_generation_matrix(
        rows,
        "caption",
        {"demo": spec},
        ["demo"],
        [0, 1],
        tmp_path / "videos",
        {"training_seed": 42},
        dry_run=True,
    )
    assert len(result) == 2
    assert result[0]["status"] == "dry_run"
    assert result[0]["seed"] == 42
    assert result[0]["generation_seed"] == 0
    assert result[1]["seed"] == 42
    assert result[1]["generation_seed"] == 1
    assert "A ball moves." in result[0]["command"]
    assert result[0]["command"][-1] == "42"
    assert result[1]["command"][-1] == "43"


def test_generation_preflight_checks_all_generators_before_launch(
    tmp_path: Path, monkeypatch
) -> None:
    marker = tmp_path / "launched"
    present = tmp_path / "present"
    present.touch()
    missing = tmp_path / "missing"
    generators = {
        "first": AblationGenerator(
            id="first",
            condition_kinds=("caption",),
            command=("python", "generate.py", "--output", "{output}"),
            required_paths=(str(present),),
        ),
        "second": AblationGenerator(
            id="second",
            condition_kinds=("caption",),
            command=("python", "generate.py", "--output", "{output}"),
            required_paths=(str(missing),),
        ),
    }

    def unexpected_run(*args, **kwargs):
        marker.touch()
        raise AssertionError("subprocess must not start before preflight passes")

    monkeypatch.setattr("ms_video_eval.ablation_generation.subprocess.run", unexpected_run)
    try:
        run_generation_matrix(
            [{"video_id": "01-001", "prompt": "A ball moves."}],
            "caption",
            generators,
            ["first", "second"],
            [0],
            tmp_path / "videos",
            {"training_seed": 42},
        )
    except FileNotFoundError as error:
        assert "second" in str(error)
        assert str(missing) in str(error)
    else:
        raise AssertionError("missing generator input should fail preflight")
    assert not marker.exists()
