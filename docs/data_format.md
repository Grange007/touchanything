# Data Format

TouchAnything expects one directory per object record. Paths in the metadata
JSON are relative to that record directory.

Required top-level JSON fields:

- `camera_model`: use `OPENCV`.
- `height` and `width`: image resolution.
- `frames`: list of frame records.

Required per-frame fields:

- `rgb_path`: RGB image path.
- `mono_depth_path`: NumPy depth array path.
- `mono_normal_path`: NumPy normal array path.
- `foreground_mask`: foreground mask image path.
- `intrinsics`: 4x4 camera intrinsics matrix, as stored in the released data.
- `camtoworld`: 4x4 camera-to-world transform.

The `intrinsics` field embeds the usual 3x3 camera matrix in a 4x4 matrix:

```text
[
  [fx,  0, cx, 0],
  [ 0, fy, cy, 0],
  [ 0,  0,  1, 0],
  [ 0,  0,  0, 1]
]
```

`fx`, `fy`, `cx`, and `cy` are in pixels. For example, the bundled camera
sample and the tested real-world and simulation records use:

```json
"intrinsics": [
  [320.0, 0.0, 160.0, 0.0],
  [0.0, 320.0, 120.0, 0.0],
  [0.0, 0.0, 1.0, 0.0],
  [0.0, 0.0, 0.0, 1.0]
]
```

The dataloader reads `fx = intrinsics[0][0]`, `fy = intrinsics[1][1]`,
`cx = intrinsics[0][2]`, and `cy = intrinsics[1][2]`. The fourth row and
column are padding, not additional camera parameters. Keep the released 4x4
representation when preparing new records; the values above are examples,
not fixed intrinsics for every dataset.

The bundled sample at `examples/data/record_printed_camera_sample20` is the
reference layout.
