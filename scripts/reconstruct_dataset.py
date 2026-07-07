#!/usr/bin/env python3
"""Run TouchAnything reconstruction over a directory of object records."""

from __future__ import annotations

import argparse
import json
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
        default="record_*",
        help="Glob used to select record directories under --dataset-root.",
    )
    parser.add_argument(
        "--json",
        default="sample_20_noaxis_8.json",
        help="Metadata JSON name expected inside each record directory.",
    )
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

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    script = Path(args.script).expanduser().resolve()
    prompt_map = load_prompt_map(args.prompt_map)

    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    if not script.is_file():
        raise SystemExit(f"Reconstruction script does not exist: {script}")

    records = sorted(p for p in dataset_root.glob(args.record_glob) if p.is_dir())
    records = [p for p in records if (p / args.json).is_file()]
    if args.max_records is not None:
        records = records[: args.max_records]
    if not records:
        raise SystemExit("No records with the requested metadata JSON were found.")

    failures = 0
    for record in records:
        prompt = prompt_map.get(record.name) or args.default_prompt or infer_prompt(record.name)
        name = record.name.replace("/", "_")
        cmd = [
            "bash",
            str(script),
            "--name",
            name,
            "--data-root",
            str(record),
            "--json",
            args.json,
            "--prompt",
            prompt,
            "--output-dir",
            args.output_root,
            "--resolution",
            str(args.resolution),
        ]
        if args.max_steps is not None:
            cmd += ["--max-steps", str(args.max_steps)]
        if args.wandb:
            cmd.append("--wandb")
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
