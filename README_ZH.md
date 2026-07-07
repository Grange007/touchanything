# TouchAnything

TouchAnything 用触觉/视觉记录数据和简短文本提示重建单个物体的三维形状。公开版包含一个裁剪后的 camera 示例数据，准备好依赖和模型权重后可以直接运行。

## 包含内容

- 单物体两阶段重建入口。
- 数据集目录批量重建入口。
- 示例数据：`examples/data/record_printed_camera_sample20`。
- 公开配置：`configs/touchanything/`。

## 安装

推荐在 Linux + NVIDIA GPU 环境中运行：

```bash
conda create -n ta python=3.9 -y
conda activate ta
pip install -r requirements.txt
```

当前 `requirements.txt` 是从 TouchAnything 使用的 `ta` 环境导出的。如果你的平台无法直接通过 pip 解析 `pytorch3d==0.7.4`，请先安装与 PyTorch 2.0.1 和 CUDA 11.8 匹配的 PyTorch3D，再重新运行 `pip install -r requirements.txt`。

也可以在仓库根目录通过 `environment.yml` 创建环境：

```bash
conda env create -f environment.yml
conda activate ta
```

请根据你的 CUDA/PyTorch 版本安装 `nvdiffrast`、`tinycudann` 和本地 rasterizer 等扩展。

## 模型权重

将扩散模型和 CLIP 权重准备到 `pretrained_models/`：

```bash
python tools/download_nd_models.py
python tools/download_sd_models.py
```

如果本机已有 Hugging Face 缓存，也可以软链：

```bash
mkdir -p pretrained_models
ln -s ~/.cache/huggingface pretrained_models/huggingface
```

## 快速运行

运行仓库内置示例：

```bash
bash scripts/reconstruct_object.sh \
  --data-root examples/data/record_printed_camera_sample20 \
  --json sample_20_noaxis_8.json \
  --prompt "a camera"
```

只检查命令构造：

```bash
bash scripts/reconstruct_object.sh --max-steps 2 --dry-run
```

去掉 `--dry-run` 后会真正训练和导出 mesh，需要 GPU 与模型权重。

## 数据集批量重建

每个物体一个 record 目录：

```text
my_dataset/
  record_object_a/
    sample_20_noaxis_8.json
    000000_rgb.png
    000000_depth.npy
    ...
```

运行：

```bash
python scripts/reconstruct_dataset.py \
  --dataset-root my_dataset \
  --json sample_20_noaxis_8.json \
  --output-root outputs/touchanything_dataset
```

如需自定义 prompt，可传入 JSON 映射文件：

```json
{
  "record_object_a": "a camera",
  "record_object_b": "a bottle"
}
```

## 致谢

感谢 RichDreamer、threestudio、normal-depth-diffusion、Stable Diffusion 以及相关开源项目。

## License

本仓库使用 Apache-2.0 license。归属信息见 `LICENSE` 和 `NOTICE`。
