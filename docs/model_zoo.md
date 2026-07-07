# Model Weights

The default configs reference:

- `pretrained_models/Damo_XR_Lab/Normal-Depth-Diffusion-Model/nd_mv_ema.ckpt`
- Hugging Face cache entries for Stable Diffusion 1.5, Stable Diffusion 2.1,
  CLIP ViT-L/14, and CLIP ViT-H/14.

Use:

```bash
python tools/download_nd_models.py
python tools/download_sd_models.py
```

Large model files are intentionally ignored by git.
