# TouchAnything

TouchAnything reconstructs a textured 3D object from tactile/visual record data and a short text prompt. The public release includes a compact camera example so the repository can be exercised immediately after dependencies and model weights are prepared.

## What Is Included

- A two-stage single-object reconstruction entrypoint.
- A dataset runner that applies the same entrypoint to many record folders.
- A 20-frame sample record at `examples/data/record_printed_camera_sample20`.
- Clean default configs under `configs/touchanything/`.

## Installation

The project is tested on Linux with NVIDIA GPUs. A CUDA-capable PyTorch environment is required.

```bash
conda create -n ta python=3.9 -y
conda activate ta
pip install -r requirements.txt
```

The exported requirements come from the `ta` environment used for TouchAnything. If your platform cannot resolve `pytorch3d==0.7.4` from pip, install a PyTorch3D build matching PyTorch 2.0.1 and CUDA 11.8 first, then rerun `pip install -r requirements.txt`.

You can also create the environment from the repository root with:

```bash
conda env create -f environment.yml
conda activate ta
```

Install any CUDA extensions required by your environment, such as `nvdiffrast`, `tinycudann`, and the local rasterizer, following the package-specific CUDA/PyTorch version requirements.

## Model Weights

Prepare diffusion and CLIP weights under `pretrained_models/`. The helper scripts use ModelScope mirrors for the common checkpoints:

```bash
python tools/download_nd_models.py
python tools/download_sd_models.py
```

If you already have Hugging Face caches, you may instead link them:

```bash
mkdir -p pretrained_models
ln -s ~/.cache/huggingface pretrained_models/huggingface
```

The expected layout is:

```text
pretrained_models/
  Damo_XR_Lab/Normal-Depth-Diffusion-Model/
  huggingface/hub/
```

## Quick Start

Run the bundled sample record:

```bash
bash scripts/reconstruct_object.sh \
  --data-root examples/data/record_printed_camera_sample20 \
  --json sample_20_noaxis_8.json \
  --prompt "a camera"
```

For a short smoke test:

```bash
bash scripts/reconstruct_object.sh --max-steps 2 --dry-run
```

Remove `--dry-run` to execute the commands. A real smoke test still requires GPU access and model weights.

## Dataset Reconstruction

Put each object record in its own folder:

```text
my_dataset/
  record_object_a/
    sample_20_noaxis_8.json
    000000_rgb.png
    000000_depth.npy
    ...
  record_object_b/
    sample_20_noaxis_8.json
    ...
```

Then run:

```bash
python scripts/reconstruct_dataset.py \
  --dataset-root my_dataset \
  --json sample_20_noaxis_8.json \
  --output-root outputs/touchanything_dataset
```

Use `--prompt-map prompts.json` to provide custom prompts:

```json
{
  "record_object_a": "a camera",
  "record_object_b": "a bottle"
}
```

## Data Format

Each record JSON follows an OPENCV camera format with a `frames` list. Every frame should reference:

- `rgb_path`
- `mono_depth_path`
- `mono_normal_path`
- `foreground_mask`
- `intrinsics`
- `camtoworld`

See `docs/data_format.md` for details.

## Outputs

By default, outputs are written to:

```text
outputs/touchanything/<experiment-name>/
  stage1-geo-neus/
  stage1-export-mesh/
  stage2-refine-dmtetra/
  stage2-export-mesh/
```

Generated outputs, caches, model weights, and full datasets are ignored by git.

## Acknowledgements

We thank the authors and maintainers of RichDreamer, threestudio, normal-depth-diffusion, Stable Diffusion, and the other open-source projects that made this release possible.

## License

This repository is released under the Apache-2.0 license. See `LICENSE` and `NOTICE` for attribution details.
