# Reproducing The Demo

1. Prepare the Python environment and model weights from the main README.
2. Confirm the sample metadata references existing files:

```bash
python scripts/reconstruct_dataset.py --dataset-root examples/data --dry-run --max-records 1
```

3. Run a command-construction check:

```bash
bash scripts/reconstruct_object.sh --dry-run --max-steps 2
```

4. On a GPU machine with weights installed, remove `--dry-run` to execute.
