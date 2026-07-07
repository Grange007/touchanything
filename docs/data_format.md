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
- `intrinsics`: 3x3 camera intrinsics.
- `camtoworld`: 4x4 camera-to-world transform.

The bundled sample at `examples/data/record_printed_camera_sample20` is the
reference layout.
