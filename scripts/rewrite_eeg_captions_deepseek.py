"""Rewrite EEG video captions with a DeepSeek-compatible chat-completions API.

This produces a new manifest and never overwrites the source annotations.  Each
request is constrained to preserve category entities, entity counts, primary
actions, and the relations needed by EEG-to-Wan semantic reconstruction.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENTITY_PATTERNS = {
    "01": (r"\bperson(?:s)?\b", r"\bball(?:s)?\b"),
    "02": (r"\bdog(?:s)?\b", r"\bball(?:s)?\b"),
    "03": (r"\bdog(?:s)?\b", r"\bcar(?:s)?\b"),
    "04": (r"\bcar(?:s)?\b", r"\bflowers?\b"),
    "05": (r"\bbird(?:s)?\b", r"\bflowers?\b"),
    "06": (r"\bperson(?:s)?\b", r"\bbird(?:s)?\b"),
    "07": (r"\bperson(?:s)?\b", r"\bdog(?:s)?\b", r"\bball(?:s)?\b"),
    "08": (r"\bperson(?:s)?\b", r"\bbird(?:s)?\b", r"\bflowers?\b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite EEG captions through a DeepSeek-compatible API.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "manifests" / "video_manifest.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manifests" / "structured_v2_video_manifest.jsonl",
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many rows; 0 means all rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print one request without calling the API.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def system_prompt() -> str:
    return """You rewrite video captions for an EEG-to-video semantic reconstruction dataset.
Return one JSON object only, with exactly these keys:
{
  "caption": "concise English caption",
  "entities": ["generic entity names with counts when plural"],
  "relations": ["short subject-action-object relations retained from the source"]
}

Rules:
1. Preserve every central entity, its count, the main action, and the direction of interaction.
2. Do not invent entities, actions, or relations absent from the source.
3. Remove names, race, age, clothing, colours, weather, lighting, camera language, and incidental background detail.
4. Use generic nouns: person, dog, car, bird, ball, flower(s).
5. Use one or two short sentences, normally 8 to 28 English words.
6. Categories 07 and 08 have three central entities. Caption 07 must explicitly mention person, dog, and ball; caption 08 must explicitly mention person, bird, and flower(s).
7. Keep distinct interactions explicit. For example: "A person throws a ball. A dog runs after and catches the ball." 
8. JSON must be valid and contain no markdown."""


def user_prompt(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task": "Rewrite this source caption as constrained JSON.",
            "video_id": row["video_id"],
            "category_id": row["category_id"],
            "source_caption": row["caption"],
        },
        ensure_ascii=False,
    )


def endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def request_rewrite(args: argparse.Namespace, api_key: str, row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt(row)},
        ],
    }
    request = urllib.request.Request(
        endpoint(args.base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(args.retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                api_response = json.loads(response.read().decode("utf-8"))
            content = api_response["choices"][0]["message"]["content"]
            if not content:
                raise ValueError("API returned empty JSON content")
            rewritten = json.loads(content)
            if not isinstance(rewritten, dict):
                raise ValueError("API response content is not a JSON object")
            validate_rewrite(str(row["category_id"]), rewritten)
            return {"rewrite": rewritten, "api_model": api_response.get("model", args.model)}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 == args.retries:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"{row['video_id']}: API rewrite failed after {args.retries} attempts: {last_error}")


def validate_rewrite(category: str, rewrite: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    required_keys = {"caption", "entities", "relations"}
    if set(rewrite) != required_keys:
        raise ValueError(f"Expected JSON keys {sorted(required_keys)}, got {sorted(rewrite)}")
    caption = str(rewrite["caption"]).strip()
    if not 5 <= len(caption.split()) <= 32:
        raise ValueError(f"Caption must contain 5..32 words, got {len(caption.split())}: {caption!r}")
    if not caption.endswith((".", "!", "?")):
        raise ValueError(f"Caption must end with sentence punctuation: {caption!r}")
    for pattern in REQUIRED_ENTITY_PATTERNS[category]:
        if not re.search(pattern, caption, flags=re.IGNORECASE):
            raise ValueError(f"Caption misses required entity pattern {pattern!r}: {caption!r}")
    entities = rewrite["entities"]
    relations = rewrite["relations"]
    if not isinstance(entities, list) or not all(isinstance(item, str) and item.strip() for item in entities):
        raise ValueError("entities must be a non-empty list of strings")
    if not isinstance(relations, list) or not all(isinstance(item, str) and item.strip() for item in relations):
        raise ValueError("relations must be a non-empty list of strings")
    return caption, entities, relations


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    if len(records) != len({str(row["video_id"]) for row in records}):
        raise ValueError("Input manifest contains duplicate video_id values")
    for row in records:
        if str(row.get("category_id")) not in REQUIRED_ENTITY_PATTERNS:
            raise ValueError(f"Unsupported category: {row.get('category_id')!r}")
    if args.limit > 0:
        records = records[: args.limit]
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace {args.output}")
    if args.dry_run:
        print(json.dumps({"endpoint": endpoint(args.base_url), "system": system_prompt(), "user": user_prompt(records[0])}, ensure_ascii=False, indent=2))
        return
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} in the environment; do not put API keys in source files or command arguments.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(records, 1):
            try:
                response = request_rewrite(args, api_key, row)
                caption, entities, relations = validate_rewrite(str(row["category_id"]), response["rewrite"])
                output = dict(row)
                output["source_caption"] = str(row["caption"])
                output["caption"] = caption
                output["caption_scheme"] = "deepseek_structured_v2"
                output["caption_entities"] = entities
                output["caption_relations"] = relations
                output["caption_rewrite_model"] = response["api_model"]
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                print(f"[deepseek-captions] {index}/{len(records)} {row['video_id']}: {caption}", flush=True)
            except RuntimeError as error:
                failures.append({"video_id": str(row["video_id"]), "error": str(error)})
                print(f"[deepseek-captions] FAILED {error}", flush=True)
            if args.sleep_seconds > 0 and index < len(records):
                time.sleep(args.sleep_seconds)
    if failures:
        failure_path = args.output.with_suffix(args.output.suffix + ".failures.jsonl")
        failure_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} rewrites failed; see {failure_path}")
    print(f"[deepseek-captions] wrote {args.output} ({len(records)} videos)")


if __name__ == "__main__":
    main()
