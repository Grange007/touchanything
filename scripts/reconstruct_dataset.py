#!/usr/bin/env python3
"""Run TouchAnything reconstruction over a directory of object records."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCRIPT = PROJECT_ROOT / "scripts" / "reconstruct_object.sh"


def infer_prompt(record_name: str) -> str:
    name = record_name
    if name.startswith("record_"):
        name = name[len("record_") :]
    tokens = []
    for token in name.split("_"):
        if token.isdigit() and len(token) >= 6:
            break
        if token in {"new", "better", "printed"}:
            continue
        if token.startswith("sample") or token.startswith("noaxis"):
            continue
        tokens.append(token)
    return "a " + (" ".join(tokens) if tokens else "object")


def select_metadata(record: Path, touches: str, filename: str | None) -> Path | None:
    if filename:
        path = record / filename
        return path if path.is_file() else None
    candidates = []
    for path in sorted(record.glob("*.json")):
        with path.open(encoding="utf-8") as source:
            metadata = json.load(source)
        if not isinstance(metadata, dict) or not isinstance(metadata.get("frames"), list):
            continue
        if touches == "all":
            if not path.stem.startswith("meta"):
                continue
        elif len(metadata["frames"]) != int(touches):
            continue
        candidates.append(path)
    # Prefer the processed coordinate convention used by the real-world demo.
    for name in (f"sample_{touches}_noaxis_8.json", "meta_data_noaxis_8.json" if touches == "all" else f"sample_{touches}.json"):
        preferred = record / name
        if preferred in candidates:
            return preferred
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous metadata in {record}: {[p.name for p in candidates]}. Specify --json.")
    return candidates[0] if candidates else None


def record_prompt(record: Path, dataset_root: Path) -> str:
    for path in (record, *record.parents):
        if path.name.startswith("record_"):
            return infer_prompt(path.name)
        if re.fullmatch(r"[0-9a-f]{32}", path.name):
            return "a " + path.parent.name.replace("_", " ")
        if path == dataset_root:
            break
    return infer_prompt(record.name)


def load_prompt_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Prompt map must be a JSON object: record_name -> prompt")
    return {str(k): str(v) for k, v in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run TouchAnything reconstruction for each record in a dataset."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "examples" / "data"),
        help="Directory containing one subdirectory per object record.",
    )
    parser.add_argument(
        "--record-glob",
        default="**",
        help="Glob used to select record directories under --dataset-root.",
    )
    parser.add_argument(
        "--json",
        help="Explicit metadata filename; overrides --touches.",
    )
    parser.add_argument("--touches", default="20", help="Number of touches (default: 20), or all for meta JSON.")
    parser.add_argument("--config-stage1", default="configs/touchanything/stage1_real.yaml")
    parser.add_argument("--config-stage2", default="configs/touchanything/stage2_real.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Small 3-step training and mesh export test.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "touchanything_dataset"),
        help="Output root passed to reconstruct_object.sh.",
    )
    parser.add_argument(
        "--prompt-map",
        type=Path,
        help="Optional JSON object mapping record directory names to prompts.",
    )
    parser.add_argument(
        "--default-prompt",
        help="Prompt used for every record when no --prompt-map entry exists.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Limit the number of records, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Forwarded trainer.max_steps override.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Forwarded mesh export resolution.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running reconstruction.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing records after a failure.",
    )
    parser.add_argument(
        "--script",
        default=str(DEFAULT_SCRIPT),
        help="Path to reconstruct_object.sh.",
    )
    args = parser.parse_args()
    if args.touches != "all" and (not args.touches.isdigit() or int(args.touches) < 1):
        parser.error("--touches must be a positive integer or all")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    script = Path(args.script).expanduser().resolve()
    prompt_map = load_prompt_map(args.prompt_map)

    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    if not script.is_file():
        raise SystemExit(f"Reconstruction script does not exist: {script}")

    directories = sorted({dataset_root, *(p for p in dataset_root.glob(args.record_glob) if p.is_dir())})
    records = [(p, metadata) for p in directories if (metadata := select_metadata(p, args.touches, args.json)) is not None]
    if args.max_records is not None:
        records = records[: args.max_records]
    if not records:
        raise SystemExit("No records with the requested metadata JSON were found.")

    failures = 0
    for record, metadata in records:
        relative = record.relative_to(dataset_root).as_posix()
        prompt = prompt_map.get(relative) or prompt_map.get(record.name) or args.default_prompt or record_prompt(record, dataset_root)
        name = (relative if relative != "." else record.name).replace("/", "_")
        print(f"Record: {record}; JSON: {metadata.name}; prompt: {prompt}", flush=True)
        cmd = [
            "bash",
            str(script),
            "--name",
            name,
            "--data-root",
            str(record),
            "--json",
            metadata.name,
            "--config-stage1",
            args.config_stage1,
            "--config-stage2",
            args.config_stage2,
            "--prompt",
            prompt,
            "--output-dir",
            args.output_root,
            "--resolution",
            str(args.resolution),
        ]
        if args.max_steps is not None:
            cmd += ["--max-steps", str(args.max_steps)]
        if args.smoke_test:
            cmd.append("--smoke-test")
        if args.wandb:
            cmd.append("--wandb")
        else:
            cmd.append("--no-wandb")
        if args.dry_run:
            cmd.append("--dry-run")

        print("+ " + " ".join(subprocess.list2cmdline([part]) for part in cmd), flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures += 1
            if not args.continue_on_error:
                return result.returncode

    if failures:
        print(f"Completed with {failures} failed record(s).")
        return 1
    print(f"Completed {len(records)} record(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
