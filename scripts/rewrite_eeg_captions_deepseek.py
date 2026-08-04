"""Rewrite EEG video captions with a DeepSeek-compatible chat-completions API.

This produces a new manifest and never overwrites the source annotations.  Each
request is constrained to preserve category entities, entity counts, primary
actions, and the relations needed by EEG-to-Wan semantic reconstruction.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENTITY_PATTERNS = {
    "01": (r"\b(?:person|people)\b", r"\b(?:ball|puck)s?\b"),
    "02": (r"\bdog(?:s)?\b", r"\bball(?:s)?\b"),
    "03": (r"\bdog(?:s)?\b", r"\b(?:car|truck|van|vehicle|suv|hatchback|pickup)s?\b"),
    "04": (r"\b(?:car|truck|van|vehicle|suv|hatchback|pickup)s?\b", r"\b(?:flower|blossom|bloom)s?\b"),
    "05": (r"\bbird(?:s)?\b", r"\b(?:flower|blossom|bloom)s?\b"),
    "06": (r"\b(?:person|people)\b", r"\bbird(?:s)?\b"),
    "07": (r"\b(?:person|people)\b", r"\bdog(?:s)?\b", r"\bball(?:s)?\b"),
    "08": (r"\b(?:person|people)\b", r"\bbird(?:s)?\b", r"\b(?:flower|blossom|bloom)s?\b"),
}
CONTEXT_TERM_PATTERN = re.compile(
    r"\b(?:park|garden|yard|field|beach|street|road|sidewalk|court|gym|room|floor|"
    r"forest|river|lake|sea|pier|plaza|balcony|greenhouse|indoors?|outdoors?|daytime|"
    r"sunny|cloudy|overcast|dusk|sunset|snowy)\b",
    flags=re.IGNORECASE,
)


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
    parser.add_argument(
        "--thinking",
        choices=["disabled", "enabled"],
        default="disabled",
        help="DeepSeek V4 thinking mode; disable it for concise structured extraction.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Captions per API call; 8 is the recommended starting point.")
    parser.add_argument("--max-tokens", type=int, default=120, help="Maximum completion tokens per caption in a batch.")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.4, help="Delay between API batches.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many rows; 0 means all rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append only missing video_ids to an existing partial output.")
    parser.add_argument("--dry-run", action="store_true", help="Print one request without calling the API.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def system_prompt() -> str:
    return """You rewrite video captions for an EEG-to-video semantic reconstruction dataset.
Return one JSON object only, with exactly these keys:
{
  "items": [
    {
      "video_id": "input video id",
      "caption": "concise English caption",
      "entities": ["generic entity names with counts when plural"],
      "relations": ["short subject-action-object relations retained from the source"]
    }
  ]
}

Rules:
1. Preserve every central entity, its count, the main action, and the direction of interaction.
2. Do not invent entities, actions, or relations absent from the source.
3. Remove names, race, age, clothing, colours, weather, lighting, camera language, and incidental background detail.
4. Use only generic central nouns: person, dog, car, bird, ball, flower(s). Do not retain setting nouns such as park, road, beach, garden, court, or indoor/outdoor context.
5. Use one or two short sentences, normally 8 to 28 English words.
6. Categories 07 and 08 have three central entities. Caption 07 must explicitly mention person, dog, and ball; caption 08 must explicitly mention person, bird, and flower(s).
7. Keep distinct interactions explicit. Categories 07 and 08 must retain every central entity in the relation list. Preserve independent actions when they occur in the source; never invent a direct interaction just to connect entities. Supporting entities such as a branch may link the central entities when needed. For example: "A person throws a ball. A dog runs after and catches the ball."
8. Return exactly one item for every input video_id. Do not omit, duplicate, reorder, or invent video_id values.
9. JSON must be valid and contain no markdown."""


def user_prompt(rows: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "task": "Rewrite every source caption as constrained JSON items.",
            "items": [
                {
                    "video_id": row["video_id"],
                    "category_id": row["category_id"],
                    "source_caption": row["caption"],
                }
                for row in rows
            ],
        },
        ensure_ascii=False,
    )


def endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def request_rewrite(args: argparse.Namespace, api_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pending = {str(row["video_id"]): row for row in rows}
    rewrites: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    api_model = args.model
    last_content = ""
    for attempt in range(args.retries):
        if not pending:
            break
        retry_after = 0.0
        try:
            retry_instruction = ""
            if attempt:
                feedback = json.dumps(errors, ensure_ascii=False)
                retry_instruction = (
                    "\nThis is a retry for only the supplied items. Return a complete valid JSON object only. "
                    f"Correct these validation errors: {feedback}"
                )
            payload = {
                "model": args.model,
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "thinking": {"type": args.thinking},
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": user_prompt(list(pending.values())) + retry_instruction},
                ],
            }
            payload["max_tokens"] = args.max_tokens * len(pending)
            request = urllib.request.Request(
                endpoint(args.base_url),
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                api_response = json.loads(response.read().decode("utf-8"))
            content = api_response["choices"][0]["message"]["content"]
            last_content = str(content)
            if not content:
                raise ValueError("API returned empty JSON content")
            rewritten = json.loads(content)
            if not isinstance(rewritten, dict) or set(rewritten) != {"items"}:
                raise ValueError("API response content is not a JSON object")
            items = rewritten["items"]
            if not isinstance(items, list):
                raise ValueError("API response items is not a list")
            returned_by_id: dict[str, dict[str, Any]] = {}
            for item in items:
                if not isinstance(item, dict) or set(item) != {"video_id", "caption", "entities", "relations"}:
                    raise ValueError("Every response item must contain exactly video_id, caption, entities, and relations")
                video_id = str(item["video_id"])
                if video_id in returned_by_id:
                    raise ValueError(f"API response duplicates video_id {video_id}")
                returned_by_id[video_id] = item
            if set(returned_by_id) != set(pending):
                missing = sorted(set(pending) - set(returned_by_id))
                unexpected = sorted(set(returned_by_id) - set(pending))
                raise ValueError(f"API response video_id mismatch; missing={missing}, unexpected={unexpected}")
            for video_id, item in returned_by_id.items():
                try:
                    validate_rewrite(str(pending[video_id]["category_id"]), item)
                except ValueError as error:
                    errors[video_id] = str(error)
                else:
                    rewrites[video_id] = item
                    del pending[video_id]
                    errors.pop(video_id, None)
            api_model = api_response.get("model", args.model)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as error:
            for video_id in pending:
                errors[video_id] = str(error)
            if isinstance(error, urllib.error.HTTPError):
                try:
                    retry_after = float(error.headers.get("Retry-After", "0"))
                except ValueError:
                    pass
        if pending and attempt + 1 < args.retries:
            time.sleep(max(retry_after, 2**attempt) + random.uniform(0, 0.5))
    if pending and last_content:
        for video_id in pending:
            errors[video_id] = f"{errors.get(video_id, 'validation failed')}. Last content: {last_content[:400]!r}"
    return {"rewrites": rewrites, "errors": errors, "api_model": api_model}


def validate_rewrite(category: str, rewrite: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    required_keys = {"caption", "entities", "relations"}
    allowed_keys = (required_keys, required_keys | {"video_id"})
    if set(rewrite) not in allowed_keys:
        raise ValueError(f"Expected JSON keys {sorted(required_keys)}, got {sorted(rewrite)}")
    caption = str(rewrite["caption"]).strip()
    if not 4 <= len(caption.split()) <= 32:
        raise ValueError(f"Caption must contain 4..32 words, got {len(caption.split())}: {caption!r}")
    if not caption.endswith((".", "!", "?")):
        raise ValueError(f"Caption must end with sentence punctuation: {caption!r}")
    context_match = CONTEXT_TERM_PATTERN.search(caption)
    if context_match:
        raise ValueError(f"Caption retains non-central context {context_match.group()!r}: {caption!r}")
    for pattern in REQUIRED_ENTITY_PATTERNS[category]:
        if not re.search(pattern, caption, flags=re.IGNORECASE):
            raise ValueError(f"Caption misses required entity pattern {pattern!r}: {caption!r}")
    entities = rewrite["entities"]
    relations = rewrite["relations"]
    if not isinstance(entities, list) or not all(isinstance(item, str) and item.strip() for item in entities):
        raise ValueError("entities must be a non-empty list of strings")
    if not isinstance(relations, list) or not all(isinstance(item, str) and item.strip() for item in relations):
        raise ValueError("relations must be a non-empty list of strings")
    for value in [*entities, *relations]:
        context_match = CONTEXT_TERM_PATTERN.search(value)
        if context_match:
            raise ValueError(f"entities/relations retain non-central context {context_match.group()!r}: {value!r}")
    for pattern in REQUIRED_ENTITY_PATTERNS[category]:
        if not any(re.search(pattern, item, flags=re.IGNORECASE) for item in entities):
            raise ValueError(f"entities misses required entity pattern {pattern!r}: {entities!r}")
    if category == "07" and not has_relation_coverage(REQUIRED_ENTITY_PATTERNS[category], relations):
        raise ValueError(
            f"Category 07 requires every central entity in at least one relation: {relations!r}"
        )
    if category == "08" and not has_connected_relation_graph(
        REQUIRED_ENTITY_PATTERNS[category], entities, relations
    ):
        raise ValueError(
            f"Category 08 requires a relation graph connecting person, bird, and flower: {relations!r}"
        )
    return caption, entities, relations


def has_relation_coverage(entity_patterns: tuple[str, ...], relations: list[str]) -> bool:
    return all(
        any(re.search(pattern, relation, flags=re.IGNORECASE) for relation in relations)
        for pattern in entity_patterns
    )


def has_connected_relation_graph(
    central_patterns: tuple[str, ...], entities: list[str], relations: list[str]
) -> bool:
    """Allow supporting entities to form paths between the required central entities."""
    parents = list(range(len(entities)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for relation in relations:
        mentioned = [
            index
            for index, entity in enumerate(entities)
            if re.search(rf"\b{re.escape(entity.strip())}\b", relation, flags=re.IGNORECASE)
        ]
        for index in mentioned[1:]:
            union(mentioned[0], index)
    central_indices = [
        next(
            (index for index, entity in enumerate(entities) if re.search(pattern, entity, flags=re.IGNORECASE)),
            None,
        )
        for pattern in central_patterns
    ]
    return all(index is not None for index in central_indices) and len(
        {find(index) for index in central_indices if index is not None}
    ) == 1


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
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    if args.output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"Output exists; pass --overwrite to replace {args.output}")
    if args.dry_run:
        preview = records[: min(args.batch_size, len(records))]
        print(json.dumps({"endpoint": endpoint(args.base_url), "system": system_prompt(), "user": user_prompt(preview)}, ensure_ascii=False, indent=2))
        return
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} in the environment; do not put API keys in source files or command arguments.")

    existing_ids: set[str] = set()
    if args.resume and args.output.exists():
        existing = read_jsonl(args.output)
        existing_ids = {str(row["video_id"]) for row in existing}
        if len(existing_ids) != len(existing):
            raise ValueError(f"Existing output contains duplicate video_id values: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failure_path = args.output.with_suffix(args.output.suffix + ".failures.jsonl")
    if not args.resume and failure_path.exists():
        failure_path.unlink()
    failures: list[dict[str, str]] = []
    mode = "a" if args.resume else "w"
    pending = [row for row in records if str(row["video_id"]) not in existing_ids]
    print(f"[deepseek-captions] pending={len(pending)} skipped={len(records) - len(pending)} batch_size={args.batch_size}")
    with args.output.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            batch_number = start // args.batch_size + 1
            batch_count = (len(pending) + args.batch_size - 1) // args.batch_size
            response = request_rewrite(args, api_key, batch)
            batch_failures: list[dict[str, str]] = []
            written = 0
            for row in batch:
                video_id = str(row["video_id"])
                rewrite = response["rewrites"].get(video_id)
                if rewrite is None:
                    batch_failures.append(
                        {"video_id": video_id, "error": response["errors"].get(video_id, "API did not return a rewrite")}
                    )
                    continue
                try:
                    caption, entities, relations = validate_rewrite(str(row["category_id"]), rewrite)
                    output = dict(row)
                    output["source_caption"] = str(row["caption"])
                    output["caption"] = caption
                    output["caption_scheme"] = "deepseek_structured_v2"
                    output["caption_entities"] = entities
                    output["caption_relations"] = relations
                    output["caption_rewrite_model"] = response["api_model"]
                    handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                    written += 1
                except ValueError as error:
                    batch_failures.append({"video_id": video_id, "error": str(error)})
            handle.flush()
            if batch_failures:
                failures.extend(batch_failures)
                with failure_path.open("a", encoding="utf-8") as failure_handle:
                    for failure in batch_failures:
                        failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(
                f"[deepseek-captions] batch {batch_number}/{batch_count} wrote {written} captions"
                + (f", failed {len(batch_failures)}" if batch_failures else ""),
                flush=True,
            )
            if args.sleep_seconds > 0 and start + len(batch) < len(pending):
                time.sleep(args.sleep_seconds)
    if failures:
        raise RuntimeError(f"{len(failures)} rewrites failed; see {failure_path}")
    print(f"[deepseek-captions] wrote {args.output} ({len(records)} videos)")


if __name__ == "__main__":
    main()
