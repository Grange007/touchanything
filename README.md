# TouchAnything: Diffusion-Guided 3D Reconstruction from Sparse Robot Touches

**Accepted by ECCV 2026.**

[Project Page](https://grange007.github.io/touchanything/) | [Paper](https://arxiv.org/abs/2604.08945)

Langzhe Gu, Hung-Jui Huang\*, Mohamad Qadri\*, Michael Kaess, Wenzhen Yuan

\* Equal contribution.

## Overview

TouchAnything reconstructs detailed 3D object geometry from sparse physical
robot touches. It transfers semantic and geometric priors from pretrained 2D
diffusion models to the tactile domain, combining local contact constraints with
a coarse class-level text prompt. This enables open-world reconstruction of
previously unseen objects without training a category-specific reconstruction
network.

The reconstruction follows a coarse-to-fine pipeline:

- **Stage 1 - Coarse geometry:** learns an implicit SDF represented by a
  multi-resolution hash grid and an MLP, supervised by tactile depth and normals
  together with diffusion-based Score Distillation Sampling (SDS).
- **Stage 2 - Fine geometry:** converts the geometry to an explicit DMTet
  representation for high-resolution differentiable rendering and detailed
  surface refinement.

This repository includes the two-stage reconstruction pipeline, dataset batch
runner, default configs, and a 20-touch camera example under
`examples/data/record_printed_camera_sample20`.

## Open-Source Roadmap

We plan to release the following components of TouchAnything:

- [x] Reconstruction code
- [x] Real-world tactile dataset
- [x] Simulation tactile dataset
- [ ] Simulation data processing pipeline

The reconstruction code and both datasets are available. The simulation data
processing pipeline will be released in a future update.

## Installation

The tested environment uses Linux, Python 3.9, CUDA 11.8, PyTorch 2.7.1, and an
NVIDIA GPU. GCC/G++ 11 must be available at `/usr/bin/gcc-11` and
`/usr/bin/g++-11` to compile the CUDA extensions.

### Create the base environment

Using `environment.yml`:

```bash
conda env create -f environment.yml
conda activate ta
```

Or installing the same base environment manually:

```bash
conda create -n ta python=3.9 pip -y
conda activate ta
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8.0 -y
```

Both methods install CUDA Toolkit 11.8 and `nvcc` inside the Conda environment;
they do not replace the system CUDA installation.

### Install PyTorch and dependencies

PyTorch 2.7.1 with `cu118` matches the CUDA 11.8 toolkit used above:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install "setuptools<81" wheel ninja

export CUDA_HOME="$CONDA_PREFIX"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11

python -m pip install --no-build-isolation -r requirements.txt
```

PyTorch must be installed first because `tinycudann`, `nvdiffrast`, and
`nerfacc` compile against the installed PyTorch and CUDA. Choose the correct gcc/g++ version for your cuda version.

## Model Weights

Download the diffusion and CLIP checkpoints through ModelScope:

```bash
python tools/download_sd_models.py
```

The default configs load the downloaded models from `pretrained_models/`,
including:

```text
pretrained_models/
  AI-ModelScope/
    stable-diffusion-2-1-base/
    clip-vit-large-patch14/
    CLIP-ViT-H-14-laion2B-s32B-b79K/
```

### Tetrahedral Grid

Stage 2 uses a 256-resolution DMTet grid. The file is too large to include in
the Git repository and must be downloaded separately:

[Download `256_tets.npz` from Google Drive](https://drive.google.com/drive/folders/1071sh1FjmuSWzV8nXOhPuUJAi7pl5MNm)

Place the downloaded file at:

```text
load/tets/256_tets.npz
```

## Quick Start

Run the bundled 20-touch camera example (tested on A40 and A100 ):

```bash
bash scripts/reconstruct_object.sh \
  --data-root examples/data/record_printed_camera_sample20 \
  --json sample_20_noaxis_8.json \
  --prompt "a camera"
```

If you are using a GPU with less memory (like an RTX 4090), you may use `stage2_real_less_mem.yaml` for Stage 2 refinement, which uses a smaller batch size and lower resolution.

```bash
bash scripts/reconstruct_object.sh \
  --data-root examples/data/record_printed_camera_sample20 \
  --json sample_20_noaxis_8.json \
  --prompt "a camera" \
  --config-stage2 configs/touchanything/stage2_real_less_mem.yaml
```

The pipeline trains and exports the Stage 1 geometry, then refines and exports
the Stage 2 geometry. By default, results are written to:

```text
outputs/touchanything/<experiment-name>/
  stage1-geo-neus/
  stage1-export-mesh/
  stage2-refine-dmtetra/
  stage2-export-mesh/
```

## Dataset Reconstruction

Published data: [Real-world dataset](https://huggingface.co/Grange007/touchanything_real_world_dataset)
and [Simulation dataset](https://huggingface.co/Grange007/touchanything_simulation_dataset).
Download and extract the archives, preserving the original directory structure.
These uploads currently use Hugging Face model repositories, rather than the
`datasets` library format.

### Download and test one object from each dataset

From the repository root, with the environment activated and model weights installed:

```bash
python tools/download_dataset_example.py real
python tools/download_dataset_example.py simulation

bash scripts/train_real_dataset.sh examples/data/hf_samples/real \
  --max-records 1 --smoke-test
bash scripts/train_simulation_dataset.sh examples/data/hf_samples/simulation \
  --max-records 1 --smoke-test
```

The downloader streams the first object from each archive without saving the
entire archive (the simulation example comes from `camera.tar.gz`). It preserves
the category directory used to infer simulation prompts. The current remote
archive paths contain a space; the downloader handles their URL encoding.

Both examples have been tested with 20 touches on an RTX 4090: the real-world
`record_screwdriver_new_20250909_115114/scale_noaxis_8` and simulation
`camera/a600991b042b2d5492348cc032adf089`.
The smoke test runs **three optimization steps per stage**, including diffusion
guidance, and exports both meshes using small grids and batches. It checks the
pipeline, not reconstruction quality or convergence.

### Full reconstruction

Remove `--smoke-test` to use the normal training settings:

```bash
bash scripts/train_real_dataset.sh examples/data/hf_samples/real --max-records 1
bash scripts/train_simulation_dataset.sh examples/data/hf_samples/simulation --max-records 1
```

For the complete extracted datasets, use their root directories and omit
`--max-records` to process all matching objects:

```bash
bash scripts/train_real_dataset.sh /path/to/touchanything_real_world_dataset
bash scripts/train_simulation_dataset.sh /path/to/simulation_dataset
```

The real-world wrapper selects `**/scale_noaxis_8` to avoid reconstructing the
same object from multiple coordinate variants. The simulation wrapper searches
recursively, including category directories such as `camera/<object-id>`.
Both use the two-stage reconstruction configs and disable W&B by default;
add `--wandb` to enable it. Outputs go to `outputs/real_world_dataset/` and
`outputs/simulation_dataset/`. For lower-memory full training, append
`--config-stage2 configs/touchanything/stage2_real_less_mem.yaml`.

### Touch counts and prompts

The default is **20 touches**, matching the main experiments. JSON selection
checks the actual number of entries in `frames`, so both
`sample_20_noaxis_8.json` and `<object-id>_20.json` are supported. When several
JSONs match, `sample_20_noaxis_8.json` is preferred, then `sample_20.json`;
otherwise an ambiguity error asks you to choose explicitly.

```bash
bash scripts/train_real_dataset.sh /path/to/dataset --touches 40
bash scripts/train_simulation_dataset.sh /path/to/dataset --touches 10
# Use all touches from a meta*.json file, where available:
bash scripts/train_real_dataset.sh /path/to/dataset --touches all
# Explicit filenames override --touches:
bash scripts/train_real_dataset.sh /path/to/dataset --json sample_20_noaxis_8.json
```

`--touches all` requires a `meta*.json`; it does not silently select the largest
numeric subset. For simulation records with only numeric subsets, choose the
available count, for example `--touches 100`.

Prompts are inferred from real-world `record_*` names or simulation category
directories. Override them with `--default-prompt "a camera"`, or use
`--prompt-map prompts.json` with dataset-relative record paths:

```json
{
  "record_screwdriver_new_20250909_115114/scale_noaxis_8": "a screwdriver",
  "camera/a600991b042b2d5492348cc032adf089": "a camera"
}
```

Add `--dry-run` to inspect selected records, JSONs, prompts, and commands
without starting training. Missing requested subsets are skipped; if no
matching records exist, the runner exits with an error.

## Single-Object EMD Evaluation

Evaluate a reconstructed mesh against its ground-truth mesh using the supplied
simulation evaluation protocol:

```bash
python tools/evaluate_object_emd.py \
  --gt-mesh /path/to/ground_truth.obj \
  --pred-mesh /path/to/stage2-export-mesh/save/it3000-export/model.obj \
  --output outputs/evaluation/object_emd.json
```

The defaults reproduce the original preprocessing: scale the prediction by
`5/9`, keep its largest connected component by vertex count, remove duplicate
and degenerate faces, attempt hole filling and normal repair, then translate
the prediction by the GT bounding-box center. The GT is not normalized.
Use the same prediction export coordinate convention as the original experiment;
do not apply this transform a second time to an already aligned export.

### Simulation scale convention

Simulation preprocessing centers the input mesh at its bounding-box center and
scales it by `0.2`. TouchAnything export then multiplies depth, camera positions,
and scene bounds by `9`, giving an overall geometry scale of `0.2 * 9 = 1.8`
relative to the input mesh. The evaluation factor `5/9 = 1/1.8` reverses this
scale; translating by the original GT bounding-box center restores its position.
This default applies when the GT is the **original input mesh**.

Both surfaces are sampled independently with 4096 points and seed 0. EMD is
the mean **Euclidean (not squared)** distance under optimal one-to-one matching
using SciPy, in the GT coordinate units; lower is better. The dense distance
matrix alone takes about 128 MiB at 4096 points, and larger sample counts can
be expensive. Change the count with `--num-samples` and optionally save the
points and matching indices with `--save-matching outputs/evaluation/matching.npz`.

For meshes already in the same coordinate system, disable the legacy transforms:

```bash
python tools/evaluate_object_emd.py \
  --gt-mesh /path/to/ground_truth.obj --pred-mesh /path/to/prediction.obj \
  --pred-scale 1 --no-gt-center-translation --no-cleanup
```

Independent surface sampling means even identical meshes can have nonzero EMD.
This script evaluates one object only; benchmark traversal and aggregation are
left to users. It does not require PyTorch3D, Open3D, or a rendering backend.

## Data Format

Each record uses an `OPENCV` camera model and contains a `frames` list. Every
frame references its RGB image, depth array, normal array, foreground mask,
4x4 camera intrinsics (with the usual 3x3 matrix in the upper-left block), and
a 4x4 camera-to-world transform. See
[docs/data_format.md](docs/data_format.md) and the bundled camera example for
the complete layout.

## Citation

```bibtex
@misc{gu2026touchanythingdiffusionguided3dreconstruction,
  title={TouchAnything: Diffusion-Guided 3D Reconstruction from Sparse Robot Touches},
  author={Langzhe Gu and Hung-Jui Huang and Mohamad Qadri and Michael Kaess and Wenzhen Yuan},
  year={2026},
  eprint={2604.08945},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.08945}
}
```

## Acknowledgements

This work is built on many amazing research works and open-source projects:

- [threestudio](https://github.com/threestudio-project/threestudio)
- [RichDreamer](https://github.com/modelscope/richdreamer)
- [Fantasia3D](https://github.com/Gorilla-Lab-SCUT/Fantasia3D)
- [gs_sdk](https://github.com/joehjhuang/gs_sdk)
- [Taxim](https://github.com/Robo-Touch/Taxim)

Thanks for their excellent work!

## License

This repository is released under the Apache-2.0 license. See [LICENSE](LICENSE)
and [NOTICE](NOTICE) for details.
