"""
DTU with Refine Camera Data Module
结合触觉传感器监督数据和refine相机位姿的数据模块
- 监督数据：触觉传感器的RGB、深度、法线等
- Refine相机：基于触觉传感器位姿和mesh表面采样的相机位姿
"""
import gzip
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Union, Any

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio import register
from threestudio.data.refine_camera import (
    RefineCameraDataModuleConfig,
    RefineCameraDataset,
    RefineCameraIterableDataset,
)
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *
from threestudio.cameras import camera_utils

def get_image(image_filename, alpha_color=None) -> torch.Tensor:
    """Returns a 3 channel image."""
    pil_image = Image.open(image_filename)
    np_image = np.array(pil_image, dtype="uint8")
    assert len(np_image.shape) == 3
    assert np_image.dtype == np.uint8
    assert np_image.shape[2] in [3, 4], f"Image shape of {np_image.shape} is incorrect."
    image = torch.from_numpy(np_image.astype("float32") / 255.0)
    if alpha_color is not None and image.shape[-1] == 4:
        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
    else:
        image = image[:, :, :3]
    return image

def _load_npy_depth(depth_path) -> np.ndarray:
    return np.load(depth_path).astype(np.float32)

def _load_png_mask(mask_path) -> np.ndarray:
    mask = np.array(Image.open(mask_path), dtype="uint8")
    if len(mask.shape) == 3:
        mask = mask[..., 0]  # use first channel
    return mask.astype(np.float32) / 255.0

@dataclass
class DtuRefineDataModuleConfig:
    # 基础路径和文件配置
    root_dir: str = ""
    json_path: str = "meta_data.json"
    
    # 图像和批次配置
    batch_size: int = 4
    n_view: int = 1
    height: int = 384
    width: int = 384
    
    # 数据加载配置
    load_preprocessed: bool = False
    cam_scale_factor: float = 1.00
    max_num_frames: int = 300
    use_mask: bool = True
    box_crop: bool = False
    box_crop_mask_thr: float = 0.4
    box_crop_context: float = 0.3
    train_num_rays: int = -1
    train_views: Optional[list] = None
    
    # 数据集分割
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    skip_every_for_val_split: int = 8
    train_val_no_overlap: bool = False
    
    # 缩放和转换配置
    scale_radius: float = 1.0
    scale_factor: float = 1.0
    auto_scale_poses: bool = False
    auto_orient: bool = False
    orientation_method: str = "up"
    
    # 数据类型配置
    include_mono_prior: bool = True
    include_sensor_depth: bool = False
    include_foreground_mask: bool = True
    include_sfm_points: bool = False
    
    # 射线噪声
    rays_noise_scale: float = 0.0
    
    # 渲染路径
    render_path: str = "circle"
    
    # Refine相机配置
    use_refine_camera: bool = True
    refine_camera: dict = field(default_factory=dict)
    
    # 相机距离范围
    camera_distance_range: List[float] = field(
        default_factory=lambda: [0.5, 4.5]
    )


class DtuRefineDatasetBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: DtuRefineDataModuleConfig = cfg

        self.use_mask = self.cfg.use_mask
        cam_scale_factor = self.cfg.cam_scale_factor

        assert os.path.exists(self.cfg.root_dir), f"{self.cfg.root_dir} doesn't exist!"
        
        metadata_path = os.path.join(self.cfg.root_dir, self.cfg.json_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        worldtogt = np.array(metadata.get("worldtogt", np.eye(4)))
        frames = metadata["frames"]

        images, depths, origin_normals, intrinsics, extrinsics, masks = [], [], [], [], [], []
        fovys = []
        self.all_directions = []
        self.all_fg_masks = []
        
        for frame in frames:
            H = self.cfg.height
            W = self.cfg.width
            
            rgb_path = os.path.join(self.cfg.root_dir, frame["rgb_path"])
            img = get_image(rgb_path).numpy()
            
            camtoworld = np.array(frame["camtoworld"])
            intrinsic = np.array(frame["intrinsics"])
            
            # 加载深度
            if self.cfg.include_mono_prior:
                depth_path = os.path.join(self.cfg.root_dir, frame["mono_depth_path"])
                depth = _load_npy_depth(depth_path)
            elif self.cfg.include_sensor_depth:
                depth_path = os.path.join(self.cfg.root_dir, frame["sensor_depth_path"])
                depth = _load_npy_depth(depth_path)
            else:
                depth = np.zeros_like(img[..., 0])
            
            # 加载法线
            normal = None
            if self.cfg.include_mono_prior:
                normal_path = os.path.join(self.cfg.root_dir, frame["mono_normal_path"])
                normal = np.load(normal_path)
                normal = normal.reshape(H, W, 3)
            
            # 加载掩码
            if self.cfg.include_foreground_mask:
                mask_path = os.path.join(self.cfg.root_dir, frame["foreground_mask"])
                mask = _load_png_mask(mask_path)
            else:
                mask = np.ones_like(img[..., 0])
            
            # 计算方向和fovy
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
            directions = get_ray_directions(H, W, (fx, fy), (cx, cy))
            fovy = 2.0 * np.arctan(H / (2.0 * fy))
            fovy = torch.tensor(fovy, dtype=torch.float32)
            
            images.append(img)
            depths.append(depth)
            origin_normals.append(normal)
            intrinsics.append(intrinsic)
            extrinsics.append(camtoworld)
            masks.append(mask)
            self.all_directions.append(directions)
            fovys.append(fovy)
        
        # 转换extrinsics到OpenGL坐标系
        extrinsics = np.stack(extrinsics)
        extrinsics[:, 0:3, 1:3] *= -1.0  # OpenCV to OpenGL
        
        # 转换为tensor
        self.all_images = torch.from_numpy(np.stack(images)).float()
        self.all_depths = torch.from_numpy(np.stack(depths)).float()
        self.all_origin_normals = torch.from_numpy(np.stack(origin_normals)).float()
        intrinsics = np.stack(intrinsics)
        self.all_fg_masks = np.stack(masks)
        
        # 将法线从相机空间转换到世界空间
        if self.cfg.include_mono_prior and len(origin_normals) > 0:
            final_normals = []
            final_extrinsics = torch.from_numpy(extrinsics).float()

            for i in range(len(final_extrinsics)):
                rot = final_extrinsics[i, :3, :3]
                cam_space_normal = self.all_origin_normals[i]

                H, W, _ = cam_space_normal.shape
                normal_map = cam_space_normal.reshape(-1, 3).T
                normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)
                world_space_normal = rot @ normal_map
                world_space_normal = world_space_normal.T.reshape(H, W, 3)
                final_normals.append(world_space_normal)

            self.all_normals = torch.stack(final_normals)
        
        self.all_c2w = torch.from_numpy(extrinsics).float()
        self.all_fovys = torch.stack(fovys, dim=0)
        self.all_directions = torch.stack(self.all_directions, dim=0)
        self.all_positions = self.all_c2w[:, :3, 3]
        
        # 保存触觉传感器位姿用于可视化
        self.tactile_sensor_c2w = self.all_c2w.clone()  # 保存完整的位姿矩阵
        
        # 设置refine相机
        if self.cfg.use_refine_camera:
            refine_camera_cfg = parse_structured(
                RefineCameraDataModuleConfig, self.cfg.get("refine_camera", {})
            )
            # 从metadata设置路径
            refine_camera_cfg.root_dir = self.cfg.root_dir
            refine_camera_cfg.json_path = self.cfg.json_path
            
            if split == "train":
                self.refine_camera_generator = RefineCameraIterableDataset(
                    refine_camera_cfg
                )
            else:
                self.refine_camera_generator = RefineCameraDataset(
                    refine_camera_cfg, split
                )
        
        # 数据集分割
        num_frames = len(frames)
        indices = list(range(num_frames))
        
        if split != "train" and self.cfg.skip_every_for_val_split >= 1:
            indices = indices[:: self.cfg.skip_every_for_val_split]
        elif self.cfg.train_val_no_overlap:
            indices = [i for i in indices if i % self.cfg.skip_every_for_val_split != 0]
        
        i_split = {"train": indices, "val": indices, "test": indices}
        
        self.all_images = self.all_images[i_split[self.split]]
        self.all_c2w = self.all_c2w[i_split[self.split]]
        self.all_positions = self.all_positions[i_split[self.split]].to(self.rank)
        self.all_directions = self.all_directions[i_split[self.split]].to(self.rank)
        self.all_fg_masks = torch.from_numpy(self.all_fg_masks)[i_split[self.split]]
        self.all_depths = self.all_depths[i_split[self.split]]
        self.all_origin_normals = self.all_origin_normals[i_split[self.split]]
        if hasattr(self, 'all_normals'):
            self.all_normals = self.all_normals[i_split[self.split]]
        
        # Scene box
        meta_scene_box = metadata["scene_box"]
        self.scene_box = {
            "aabb": torch.tensor(meta_scene_box["aabb"], dtype=torch.float32),
            "near": meta_scene_box["near"],
            "far": meta_scene_box["far"],
            "radius": meta_scene_box["radius"]
        }
        
        # 移到设备
        self.all_c2w = self.all_c2w.float().to(self.rank)
        self.all_images = self.all_images.float().to(self.rank)
        self.all_fg_masks = self.all_fg_masks.float().to(self.rank)
        self.all_depths = self.all_depths.float().to(self.rank)
        self.all_origin_normals = self.all_origin_normals.float().to(self.rank)
        if hasattr(self, 'all_normals'):
            self.all_normals = self.all_normals.float().to(self.rank)
        
        # 自动保存触觉传感器位姿可视化（仅在训练集且rank 0时保存）
        if split == "train" and self.rank == 0:
            tactile_save_path = os.path.join(self.cfg.root_dir, "tactile_sensor_poses.ply")
            self.save_tactile_sensor_poses_as_pointcloud(tactile_save_path)

    def save_tactile_sensor_poses_as_pointcloud(self, output_path: str = "tactile_sensor_poses.ply"):
        """将触觉传感器位姿保存为点云文件（带箭头可视化）
        
        Args:
            output_path: 输出文件路径
        """
        try:
            import open3d as o3d
            
            # 提取位置和方向
            positions = self.tactile_sensor_c2w[:, :3, 3].cpu().numpy()  # [N, 3]
            # 触觉传感器朝向 = -z轴方向（OpenGL约定）
            directions = -self.tactile_sensor_c2w[:, :3, 2].cpu().numpy()  # [N, 3]
            
            # 创建标准点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)
            pcd.normals = o3d.utility.Vector3dVector(directions)
            
            # 使用颜色表示方向
            colors = (directions + 1.0) / 2.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 保存点云
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"[INFO] Saved {len(positions)} tactile sensor poses to {output_path}")
            
            # 保存带箭头的可视化版本
            vis_path = output_path.replace('.ply', '_vis.ply')
            self._save_tactile_visualization(positions, directions, vis_path)
            
        except Exception as e:
            print(f"[WARNING] Failed to save tactile sensor pointcloud: {e}")
    
    def _save_tactile_visualization(self, positions: np.ndarray, directions: np.ndarray, output_path: str):
        """保存带箭头的触觉传感器可视化
        
        Args:
            positions: 传感器位置 [N, 3]
            directions: 传感器方向 [N, 3]
            output_path: 输出文件路径
        """
        try:
            import open3d as o3d
            
            geometries = []
            arrow_length = 0.05  # 箭头长度，可以调整
            
            for i in range(len(positions)):
                start = positions[i]
                direction = directions[i]
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                
                # 创建箭头mesh
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=arrow_length * 0.1,
                    cone_radius=arrow_length * 0.2,
                    cylinder_height=arrow_length * 0.6,
                    cone_height=arrow_length * 0.4,
                )
                
                # 计算旋转矩阵
                z_axis = np.array([0, 0, 1])
                rotation_axis = np.cross(z_axis, direction)
                rotation_axis_norm = np.linalg.norm(rotation_axis)
                
                if rotation_axis_norm > 1e-6:
                    rotation_axis = rotation_axis / rotation_axis_norm
                    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                    
                    # Rodrigues旋转公式
                    K = np.array([
                        [0, -rotation_axis[2], rotation_axis[1]],
                        [rotation_axis[2], 0, -rotation_axis[0]],
                        [-rotation_axis[1], rotation_axis[0], 0]
                    ])
                    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                else:
                    R = np.eye(3) if np.dot(z_axis, direction) > 0 else -np.eye(3)
                
                # 应用变换
                arrow.rotate(R, center=[0, 0, 0])
                arrow.translate(start)
                
                # 设置颜色 - 触觉传感器用绿色
                arrow.paint_uniform_color([0.2, 0.8, 0.2])
                geometries.append(arrow)
                
                # 添加小球标记传感器位置
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=arrow_length * 0.15)
                sphere.translate(start)
                sphere.paint_uniform_color([0.8, 0.2, 0.8])  # 紫色球体
                geometries.append(sphere)
            
            # 合并所有几何体
            combined_mesh = o3d.geometry.TriangleMesh()
            for geom in geometries:
                combined_mesh += geom
            
            combined_mesh.compute_vertex_normals()
            
            # 保存
            o3d.io.write_triangle_mesh(output_path, combined_mesh)
            print(f"[INFO] Saved tactile sensor visualization with {len(positions)} arrows to {output_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to save tactile sensor visualization: {e}")

    def get_all_images(self):
        return self.all_images


class DtuRefineDataset(Dataset, DtuRefineDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)

    def __len__(self):
        if self.split == "test":
            if self.cfg.render_path == "circle":
                return len(self.refine_camera_generator)
            else:
                return len(self.all_images)
        else:
            return len(self.refine_camera_generator)
            # return len(self.all_images)

    def prepare_data(self, index):
        """准备监督数据"""
        c2w = self.all_c2w[index]
        light_positions = c2w[..., :3, -1]
        directions = self.all_directions[index]
        
        # 添加batch维度以保持与refine_camera一致的形状
        c2w_batch = c2w # [1, 4, 4]
        directions_batch = directions  # [1, H, W, 3]
        
        rays_o, rays_d = get_rays(
            directions_batch, c2w_batch, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )
        # 现在 rays_o 和 rays_d 的形状都是 [1, H, W, 3]
        
        fovy = self.all_fovys[index]
        rgb = self.all_images[index]
        depth = self.all_depths[index]
        normal = self.all_normals[index]
        origin_normal = self.all_origin_normals[index]
        mask = self.all_fg_masks[index]
        camera_distances = torch.norm(
            c2w[..., :3, -1], dim=-1, keepdim=True
        )
        camera_distances_relative = camera_distances

        # Dynamic near/far calculation to avoid near plane clipping
        # near = max(camera_distance - radius * sqrt(3), min_near)
        import math
        radius = self.scene_box["radius"]
        dynamic_near = max(
            camera_distances.item() - radius * math.sqrt(3),
            0.01  # Minimum near plane distance
        )
        dynamic_far = camera_distances.item() + radius * math.sqrt(3)
        
        # Use dynamic values or scene_box values (comment out the one you don't want)
        near_plane = dynamic_near  # Use this for dynamic calculation
        far_plane = dynamic_far    # Use this for dynamic calculation
        # near_plane = self.scene_box["near"]  # Or use scene_box values
        # far_plane = self.scene_box["far"]    # Or use scene_box values

        proj_mtx = get_projection_matrix(
            fovy.unsqueeze(0),
            self.cfg.width / self.cfg.height,
            near_plane,
            far_plane,
        ).squeeze(0)
        
        c2w = c2w.to(self.rank)
        proj_mtx = proj_mtx.to(self.rank)
        mvp_mtx = get_mvp_matrix(c2w.unsqueeze(0), proj_mtx.unsqueeze(0)).squeeze(0)
        
        # light_positions 也需要添加batch维度
        light_positions_batch = light_positions  # [1, 3]
        
        batch = {
            "index": index,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": c2w[..., :3, -1],
            "c2w": c2w,
            "light_positions": light_positions_batch,  # 使用batch版本
            "elevation": 0,
            "azimuth": 0,
            "camera_distances": camera_distances,
            "camera_distances_relative": camera_distances_relative, 
            "height": self.cfg.height,
            "width": self.cfg.width,
            "fovy": fovy,
            "rgb": rgb,
            "ref_depth": depth,
            "ref_normal": normal,
            "origin_normal": origin_normal,
            "mask": mask,
            "scene_box": self.scene_box,
        }

        return batch

    def __getitem__(self, index):
        # return self.prepare_data(index)
        
        if self.split == "test":
            if self.cfg.render_path == "circle":
                return self.refine_camera_generator[index]
            else:
                return self.prepare_data(index)
        else:
            return self.refine_camera_generator[index]


class DtuRefineIterableDataset(IterableDataset, DtuRefineDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)
        self.idx = 0
        self.image_perm = torch.randperm(len(self.all_images))

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        """准备监督数据batch"""
        idx = self.image_perm[self.idx]
        c2w = self.all_c2w[idx][None]
        light_positions = c2w[..., :3, -1]
        directions = self.all_directions[idx][None]
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )
        rgb = self.all_images[idx][None]
        depth = self.all_depths[idx][None]
        normal = self.all_normals[idx][None]
        origin_normal = self.all_origin_normals[idx][None]
        mask = self.all_fg_masks[idx][None]
        camera_distances = torch.norm(
            c2w[..., :3, -1], dim=-1, keepdim=True
        )
        fovy = self.all_fovys[idx][None]
        camera_distances_relative = camera_distances
        
        # Dynamic near/far calculation
        import math
        radius = self.scene_box["radius"]
        dynamic_near = max(
            camera_distances.item() - radius * math.sqrt(3),
            0.01
        )
        dynamic_far = camera_distances.item() + radius * math.sqrt(3)
        
        near_plane = dynamic_near
        far_plane = dynamic_far
        
        proj_mtx = get_projection_matrix(
            fovy,
            self.cfg.width / self.cfg.height,
            near_plane,
            far_plane,
        )
        proj_mtx = proj_mtx.to(self.rank)
        c2w = c2w.to(self.rank)
        mvp_mtx = get_mvp_matrix(c2w, proj_mtx)
        
        # 随机采样射线
        if (
            self.cfg.train_num_rays != -1
            and self.cfg.train_num_rays < self.cfg.height * self.cfg.width
        ):
            _, height, width, _ = rays_o.shape
            x = torch.randint(
                0, width, size=(self.cfg.train_num_rays,), device=rays_o.device
            )
            y = torch.randint(
                0, height, size=(self.cfg.train_num_rays,), device=rays_o.device
            )

            rays_o = rays_o[:, y, x].unsqueeze(-2)
            rays_d = rays_d[:, y, x].unsqueeze(-2)
            rgb = rgb[:, y, x].unsqueeze(-2)
            mask = mask[:, y, x].unsqueeze(-1)
            depth = depth[:, y, x].unsqueeze(-1)
            normal = normal[:, y, x].unsqueeze(-2)
            origin_normal = origin_normal[:, y, x].unsqueeze(-2)
        
        mask = mask.unsqueeze(-1)
        depth = depth.unsqueeze(-1)

        batch = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": c2w[..., :3, -1],
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": None,
            "azimuth": None,
            "camera_distances": camera_distances,
            "camera_distances_relative": camera_distances_relative,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "fovy": fovy,
            "rgb": rgb,
            "ref_depth": depth,
            "ref_normal": normal,
            "origin_normal": origin_normal,
            "mask": mask,
            "scene_box": self.scene_box,
        }
        
        # 添加refine相机数据
        if self.cfg.use_refine_camera:
            batch["refine_camera"] = self.refine_camera_generator.collate(None)

        self.idx += 1
        if self.idx == len(self.all_images):
            self.idx = 0
            self.image_perm = torch.randperm(len(self.all_images))

        return batch


@register("dtu-refine-datamodule")
class DtuRefineDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(DtuRefineDataModuleConfig, cfg)

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = DtuRefineIterableDataset(self.cfg, self.cfg.train_split)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = DtuRefineDataset(self.cfg, self.cfg.val_split)
        if stage in [None, "test", "predict"]:
            self.test_dataset = DtuRefineDataset(self.cfg, self.cfg.test_split)

    def prepare_data(self):
        pass
    
    def save_tactile_sensor_pointcloud(self, output_path: str = None):
        """保存触觉传感器位姿点云
        
        Args:
            output_path: 输出文件路径，如果为None则使用默认路径
        """
        if hasattr(self, 'train_dataset'):
            if output_path is None:
                output_path = os.path.join(
                    self.cfg.root_dir if self.cfg.root_dir else ".", 
                    "tactile_sensor_poses.ply"
                )
            self.train_dataset.save_tactile_sensor_poses_as_pointcloud(output_path)
        else:
            print("[WARNING] Train dataset not initialized, call setup() first")

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self.general_loader(
            self.train_dataset, batch_size=1, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)
