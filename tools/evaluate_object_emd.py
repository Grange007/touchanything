#!/usr/bin/env python3
"""Evaluate one mesh pair using the supplied simulation EMD protocol."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def earth_mover_distance(points_gt, points_pred):
    """Return mean Euclidean optimal-assignment distance and matching indices."""
    points_gt = np.asarray(points_gt, dtype=np.float64)
    points_pred = np.asarray(points_pred, dtype=np.float64)
    if (points_gt.ndim != 2 or points_gt.shape[1] != 3
            or points_pred.shape != points_gt.shape or len(points_gt) == 0):
        raise ValueError("Point clouds must have the same nonempty (N, 3) shape")
    if not np.isfinite(points_gt).all() or not np.isfinite(points_pred).all():
        raise ValueError("Point clouds must contain finite coordinates")
    distances = cdist(points_gt, points_pred)
    assignment = linear_sum_assignment(distances)
    return float(distances[assignment].mean()), assignment


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"No triangle mesh in {path}")
    if not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError(f"Invalid mesh surface in {path}")
    return mesh


def preprocess_mesh(mesh):
    components = mesh.split(only_watertight=False)
    if len(components):
        mesh = max(components, key=lambda component: len(component.vertices))
    # These APIs also work on trimesh versions that removed remove_*_faces.
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    try:
        mesh.fill_holes()
    except Exception:
        pass
    try:
        mesh.fix_normals()
    except Exception:
        pass
    if not len(mesh.faces) or mesh.area <= 0:
        raise ValueError("Prediction has no surface after cleanup")
    return mesh


def evaluate_object(gt_path, pred_path, num_samples=4096, seed=0,
                    pred_scale=5.0 / 9.0, translate_to_gt_center=True, cleanup=True):
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if not np.isfinite(pred_scale) or pred_scale <= 0:
        raise ValueError("pred_scale must be finite and positive")
    gt_mesh = load_mesh(gt_path)
    pred_mesh = load_mesh(pred_path)
    pred_mesh.apply_scale(pred_scale)
    if cleanup:
        pred_mesh = preprocess_mesh(pred_mesh)
    if translate_to_gt_center:
        pred_mesh.apply_translation(gt_mesh.bounds.mean(axis=0))
    # Preserve the original protocol's seeded GT-then-prediction sampling order.
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        gt_points, _ = trimesh.sample.sample_surface(gt_mesh, num_samples)
        pred_points, _ = trimesh.sample.sample_surface(pred_mesh, num_samples)
    finally:
        np.random.set_state(state)
    emd, assignment = earth_mover_distance(gt_points, pred_points)
    return emd, gt_points, pred_points, assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-mesh", required=True, type=Path)
    parser.add_argument("--pred-mesh", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pred-scale", type=float, default=5.0 / 9.0)
    parser.add_argument("--no-gt-center-translation", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional result JSON")
    parser.add_argument("--save-matching", type=Path, help="Optional sampled points and matching NPZ")
    args = parser.parse_args()
    try:
        emd, gt_points, pred_points, assignment = evaluate_object(
            args.gt_mesh, args.pred_mesh, args.num_samples, args.seed,
            args.pred_scale, not args.no_gt_center_translation, not args.no_cleanup,
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    result = {
        "gt_mesh": str(args.gt_mesh.resolve()),
        "pred_mesh": str(args.pred_mesh.resolve()),
        "emd": emd,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "pred_scale": args.pred_scale,
        "translate_to_gt_center": not args.no_gt_center_translation,
        "cleanup": not args.no_cleanup,
        "distance": "mean Euclidean optimal assignment (not squared)",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.save_matching:
        args.save_matching.parent.mkdir(parents=True, exist_ok=True)
        with args.save_matching.open("wb") as output:
            np.savez_compressed(output, gt_points=gt_points, pred_points=pred_points,
                                row_ind=assignment[0], col_ind=assignment[1])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
