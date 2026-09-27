"""Download the first complete object from a published dataset archive."""

import argparse
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["real", "simulation"])
    parser.add_argument("--output-dir", type=Path, default=Path("examples/data/hf_samples"))
    args = parser.parse_args()
    repository = "touchanything_real_world_dataset" if args.kind == "real" else "touchanything_simulation_dataset"
    filename = "%20" if args.kind == "real" else "%20/camera.tar.gz"
    url = f"https://huggingface.co/Grange007/{repository}/resolve/main/{filename}?download=true"
    destination = args.output_dir.resolve() / args.kind
    record = None
    count = 0
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if not member.isfile():
                    continue
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive path: {path}")
                if record is None:
                    record = path.parent
                    relative_record = Path(str(record)) if args.kind == "simulation" else Path(record.name)
                    print(f"Downloading object: {record}", flush=True)
                if not path.is_relative_to(record):
                    break
                # Reconstruction needs frame files and JSON, not cached point clouds.
                if path.suffix not in {".json", ".png", ".npy"}:
                    continue
                target = destination / relative_record / path.relative_to(record)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                count += 1
                if count % 100 == 0:
                    print(f"Downloaded {count} files", flush=True)
    if record is None:
        raise RuntimeError("No object found in archive")
    print(f"Object directory: {destination / relative_record}")


if __name__ == "__main__":
    main()
