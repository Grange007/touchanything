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
- [ ] Real-world tactile dataset
- [ ] Simulation data processing pipeline

The reconstruction code is currently available. The real-world dataset and
simulation data processing pipeline will be released in future updates.

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

Place each object record in a separate directory:

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

Run all records matching `record_*`:

```bash
python scripts/reconstruct_dataset.py \
  --dataset-root my_dataset \
  --json sample_20_noaxis_8.json \
  --output-root outputs/touchanything_dataset
```

Use `--prompt-map prompts.json` to provide prompts for individual records:

```json
{
  "record_object_a": "a camera",
  "record_object_b": "a bottle"
}
```

## Data Format

Each record uses an `OPENCV` camera model and contains a `frames` list. Every
frame references its RGB image, depth array, normal array, foreground mask,
camera intrinsics, and camera-to-world transform. See
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
