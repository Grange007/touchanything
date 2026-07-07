"""
Refine Camera Data Module
用于第二阶段refine的相机位姿数据生成
支持三种采样策略（可以在配置中设置权重）:
1. 基于触觉传感器位姿后退生成 (tactile_weight)
2. 基于mesh表面采样生成 (mesh_weight)
3. 随机采样（uncond_my.py逻辑）(random_weight)

配置示例:
  data:
    tactile_weight: 1.0    # 触觉传感器采样权重
    mesh_weight: 1.0       # mesh采样权重
    random_weight: 1.0     # 随机采样权重（uncond_my逻辑）
    # 这些权重会被自动归一化为概率
    # 例如上面的配置会得到: 33.3% tactile, 33.3% mesh, 33.3% random
    
    # 如果只想使用tactile和random:
    # tactile_weight: 2.0
    # mesh_weight: 0.0
    # random_weight: 1.0
    # 结果: 66.7% tactile, 0% mesh, 33.3% random
"""
import bisect
import math
import os
import json
import pytorch_lightning as pl
import random
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing import List, Tuple, Any, Optional

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_mvp_matrix, 
    get_projection_matrix,
    get_ray_directions, 
    get_rays,
)
from threestudio.utils.typing import *


@dataclass
class RefineCameraDataModuleConfig:
    # 基础相机参数
    height: Any = 64
    width: Any = 64
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    
    # 相机范围参数
    fovy_range: Tuple[float, float] = (40, 70)
    elevation_range: Tuple[float, float] = (-10, 90)  # 用于随机采样
    camera_distance_range: Tuple[float, float] = (1.0, 1.5)  # 用于随机采样
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    light_sample_strategy: str = "dreamfusion"
    
    # 数据加载相关参数（从JSON读取）
    root_dir: str = ""  # 数据根目录
    json_path: str = "meta_data.json"  # metadata JSON文件路径
    
    # 触觉传感器相关参数
    tactile_retreat_distance: float = 0.02  # 触觉传感器后退距离 (米)
    tactile_retreat_multiplier: float = 5.0  # 后退距离倍数
    
    # Mesh采样相关参数
    mesh_path: str = ""  # mesh文件路径
    mesh_retreat_distance: float = 0.03  # mesh表面后退距离 (米)
    mesh_retreat_multiplier: float = 3.0  # 后退距离倍数
    mesh_num_samples: int = 100  # mesh表面采样点数量
    
    # 采样策略权重（会被归一化为概率）
    tactile_weight: float = 1.0  # 触觉传感器采样权重
    mesh_weight: float = 1.0  # mesh表面采样权重
    random_weight: float = 1.0  # 随机采样权重（uncond_my.py逻辑）
    
    # 视角扰动参数
    view_perturbation: float = 0.0  # 视角方向扰动
    position_noise: float = 0.0  # 位置噪声
    
    # 其他参数
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    progressive_until: int = 0
    relative_radius: bool = False
    open3d_coord: bool = False  # whether to use Open3D coordinate system (y up, x right, z back)


class RefineCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RefineCameraDataModuleConfig = cfg
        
        # 分辨率设置
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.fovy_range = self.cfg.fovy_range
        
        self.true_global_step = 0
        
        # 加载触觉传感器位姿
        self.tactile_sensor_positions = []
        self.tactile_sensor_directions = []
        
        # 从JSON加载触觉传感器数据
        if self.cfg.root_dir and self.cfg.json_path:
            self._load_tactile_from_json()
        else:
            threestudio.warn("No root_dir or json_path provided, skipping tactile sensor loading")
        
        if len(self.tactile_sensor_positions) > 0:
            threestudio.info(f"Total {len(self.tactile_sensor_positions)} tactile sensor poses loaded")
        else:
            threestudio.warn("No tactile sensor poses loaded")
        
        # 加载并采样mesh
        self.mesh_sample_positions = []
        self.mesh_sample_normals = []
        if self.cfg.mesh_path and os.path.exists(self.cfg.mesh_path):
            self._sample_mesh_surface()
            threestudio.info(f"Sampled {len(self.mesh_sample_positions)} points from mesh")
        
        # 合并所有采样点
        self.all_sample_positions = []
        self.all_sample_directions = []
        self.all_sample_types = []  # 'tactile' or 'mesh'
        
        # 添加触觉传感器采样点
        if len(self.tactile_sensor_positions) > 0:
            for i in range(len(self.tactile_sensor_positions)):
                self.all_sample_positions.append(self.tactile_sensor_positions[i])
                self.all_sample_directions.append(self.tactile_sensor_directions[i])
                self.all_sample_types.append('tactile')
        
        # 添加mesh采样点
        if len(self.mesh_sample_positions) > 0:
            for i in range(len(self.mesh_sample_positions)):
                self.all_sample_positions.append(self.mesh_sample_positions[i])
                self.all_sample_directions.append(self.mesh_sample_normals[i])
                self.all_sample_types.append('mesh')
        
        threestudio.info(f"Total {len(self.all_sample_positions)} camera anchor points")
        
        # 计算归一化后的采样概率
        total_weight = self.cfg.tactile_weight + self.cfg.mesh_weight + self.cfg.random_weight
        self.tactile_prob = self.cfg.tactile_weight / total_weight
        self.mesh_prob = self.cfg.mesh_weight / total_weight
        self.random_prob = self.cfg.random_weight / total_weight
        
        threestudio.info(
            f"Sampling probabilities - Tactile: {self.tactile_prob:.3f}, "
            f"Mesh: {self.mesh_prob:.3f}, Random: {self.random_prob:.3f}"
        )
        
        # 用于存储生成的相机位姿
        self.generated_camera_positions = []
        self.generated_camera_directions = []
        self.save_camera_pointcloud = True  # 是否保存相机位姿点云
        self.max_cameras_to_save = 60  # 最多保存的相机数量
    
    def _load_tactile_from_json(self):
        """从JSON metadata文件加载触觉传感器位姿"""
        try:
            metadata_path = os.path.join(self.cfg.root_dir, self.cfg.json_path)
            if not os.path.exists(metadata_path):
                threestudio.warn(f"Metadata file not found: {metadata_path}")
                return
            
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            frames = metadata.get("frames", [])
            if len(frames) == 0:
                threestudio.warn("No frames found in metadata")
                return
            
            # 先收集所有原始的camtoworld矩阵
            camtoworlds = []
            
            for frame in frames:
                camtoworld = np.array(frame["camtoworld"])
                camtoworlds.append(camtoworld)
            
            # 统一转换到OpenGL坐标系（与dtu_refine.py保持一致）
            camtoworlds = np.stack(camtoworlds)
            camtoworlds[:, 0:3, 1:3] *= -1.0  # OpenCV to OpenGL
            
            # 提取位置和方向（z轴）
            positions = camtoworlds[:, :3, 3]  # [N, 3]
            directions = -camtoworlds[:, :3, 2]  # [N, 3] z轴方向
            
            self.tactile_sensor_positions = torch.from_numpy(positions).float()
            self.tactile_sensor_directions = torch.from_numpy(directions).float()
            
            # 保存完整的 c2w 矩阵，便于在 retreat=0 时直接复用原始位姿
            try:
                self.tactile_sensor_c2w_mats = torch.from_numpy(camtoworlds).float()
            except Exception:
                self.tactile_sensor_c2w_mats = None
            
            threestudio.info(f"Loaded {len(self.tactile_sensor_positions)} tactile sensor poses from JSON")
            
        except Exception as e:
            threestudio.warn(f"Failed to load tactile sensor data from JSON: {e}")

        
    def _sample_mesh_surface(self):
        """从mesh表面采样点和法线"""
        try:
            mesh = o3d.io.read_triangle_mesh(self.cfg.mesh_path)
            mesh.compute_triangle_normals()
            mesh.compute_vertex_normals()
            
            # 使用均匀采样
            pcd = mesh.sample_points_uniformly(number_of_points=self.cfg.mesh_num_samples)
            
            # 获取点和法线
            points = np.asarray(pcd.points)
            
            # 为每个采样点计算法线（使用最近的n个三角形中采样k个的平均法线）
            triangles = np.asarray(mesh.triangles)
            vertices = np.asarray(mesh.vertices)
            triangle_normals = np.asarray(mesh.triangle_normals)
            
            # 计算三角形中心
            triangle_centers = vertices[triangles].mean(axis=1)
            
            # 设置最近邻数量和采样数量
            n_nearest = 10  # 找最近的n个三角形
            k_sample = 7    # 从中采样k个
            
            normals = []
            for point in points:
                # 计算到所有三角形中心的距离
                distances = np.linalg.norm(triangle_centers - point, axis=1)
                
                # 找到最近的n个三角形的索引
                n_actual = min(n_nearest, len(distances))
                nearest_indices = np.argpartition(distances, n_actual - 1)[:n_actual]
                
                # 从最近的n个三角形中随机采样k个
                k_actual = min(k_sample, len(nearest_indices))
                sampled_indices = np.random.choice(nearest_indices, size=k_actual, replace=False)
                
                # 计算这k个三角形法线的平均值
                sampled_normals = triangle_normals[sampled_indices]
                avg_normal = sampled_normals.mean(axis=0)
                
                # 归一化平均法线
                avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-8)
                
                normals.append(avg_normal)
            
            normals = np.array(normals)
            
            # 确保法线指向外侧（远离原点）
            # for i in range(len(normals)):
            #     if np.dot(normals[i], points[i]) < 0:
            #         normals[i] = -normals[i]
            
            self.mesh_sample_positions = torch.from_numpy(points).float()
            self.mesh_sample_normals = -torch.from_numpy(normals).float()
            
        except Exception as e:
            threestudio.warn(f"Failed to load mesh from {self.cfg.mesh_path}: {e}")
    
    def save_cameras_as_pointcloud(self, output_path: str = "camera_poses.ply"):
        """将生成的相机位姿保存为点云文件
        
        Args:
            output_path: 输出文件路径
        """
        if len(self.generated_camera_positions) == 0:
            threestudio.warn("No camera poses to save")
            return
        
        try:
            # 转换为numpy数组
            positions = torch.stack(self.generated_camera_positions).cpu().numpy()
            directions = torch.stack(self.generated_camera_directions).cpu().numpy()
            
            # 创建点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)
            
            # 使用方向作为法线
            pcd.normals = o3d.utility.Vector3dVector(directions)
            
            # 使用颜色表示方向（将方向映射到RGB）
            # 将方向从[-1,1]映射到[0,1]
            colors = (directions + 1.0) / 2.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 保存点云
            o3d.io.write_point_cloud(output_path, pcd)
            threestudio.info(f"Saved {len(positions)} camera poses to {output_path}")
            
            # 同时保存一个带箭头的可视化版本
            self._save_camera_visualization(output_path.replace('.ply', '_vis.ply'))
            
        except Exception as e:
            threestudio.warn(f"Failed to save camera pointcloud: {e}")
    
    def _save_camera_visualization(self, output_path: str):
        """保存带箭头的相机可视化
        
        Args:
            output_path: 输出文件路径
        """
        try:
            positions = torch.stack(self.generated_camera_positions).cpu().numpy()
            directions = torch.stack(self.generated_camera_directions).cpu().numpy()
            
            # 创建箭头几何体列表
            geometries = []
            arrow_length = 0.03  # 增加箭头长度，使其更明显
            
            for i in range(len(positions)):
                # 创建箭头（从位置指向view_direction）
                start = positions[i]
                direction = directions[i]
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                
                # 创建箭头mesh
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=arrow_length * 0.08,  # 增粗一点
                    cone_radius=arrow_length * 0.15,      # 锥体更大
                    cylinder_height=arrow_length * 0.65,
                    cone_height=arrow_length * 0.35,
                )
                
                # 计算旋转矩阵，使箭头指向正确方向
                # 默认箭头指向z轴正方向
                z_axis = np.array([0, 0, 1])
                rotation_axis = np.cross(z_axis, direction)
                rotation_axis_norm = np.linalg.norm(rotation_axis)
                
                if rotation_axis_norm > 1e-6:
                    rotation_axis = rotation_axis / rotation_axis_norm
                    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                    
                    # 使用Rodrigues旋转公式
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
                
                # 使用不同的颜色方案：根据方向的主要分量
                # 红色表示X方向，绿色表示Y方向，蓝色表示Z方向
                abs_direction = np.abs(direction)
                dominant_axis = np.argmax(abs_direction)
                if dominant_axis == 0:  # X主导
                    color = np.array([1.0, 0.3, 0.3])
                elif dominant_axis == 1:  # Y主导
                    color = np.array([0.3, 1.0, 0.3])
                else:  # Z主导
                    color = np.array([0.3, 0.3, 1.0])
                
                arrow.paint_uniform_color(color)
                geometries.append(arrow)
                
                # 同时添加一个小球表示相机位置
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=arrow_length * 0.12)
                sphere.translate(start)
                sphere.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色球体
                geometries.append(sphere)
            
            # 合并所有几何体
            combined_mesh = o3d.geometry.TriangleMesh()
            for geom in geometries:
                combined_mesh += geom
            
            # 计算法线以便更好地渲染
            combined_mesh.compute_vertex_normals()
            
            # 保存
            o3d.io.write_triangle_mesh(output_path, combined_mesh)
            threestudio.info(f"Saved camera visualization with {len(positions)} arrows to {output_path}")
            
        except Exception as e:
            threestudio.warn(f"Failed to save camera visualization: {e}")
    
    def _generate_camera_from_anchor(
        self, 
        anchor_position: torch.Tensor, 
        anchor_direction: torch.Tensor,
        sample_type: str
    ) -> dict:
        """从锚点生成相机位姿
        
        Args:
            anchor_position: 锚点位置 (触觉传感器位置或mesh表面点)
            anchor_direction: 锚点方向 (触觉传感器方向或mesh法线，在OpenGL坐标系中)
            sample_type: 'tactile' or 'mesh'
        """
        # 归一化方向
        direction = F.normalize(anchor_direction.unsqueeze(0), dim=-1).squeeze(0)
        
        # 根据类型选择后退距离
        if sample_type == 'tactile':
            retreat_dist = self.cfg.tactile_retreat_distance * self.cfg.tactile_retreat_multiplier
        else:  # mesh
            retreat_dist = self.cfg.mesh_retreat_distance * self.cfg.mesh_retreat_multiplier
        
        # 沿方向后退
        camera_position = anchor_position - direction * retreat_dist
        
        # 添加位置噪声
        if self.cfg.position_noise > 0:
            noise = torch.randn(3) * self.cfg.position_noise
            camera_position = camera_position + noise
        
        # 计算相机朝向
        # 在dtu_refine中，触觉传感器的方向是c2w[:, 2]（z轴）
        # 相机应该沿着相同的方向看
        # 在OpenGL中，相机看向-z方向，所以如果我们想让相机看向direction，
        # 那么c2w[:, 2]应该是-direction
        view_direction = direction  # 我们希望相机看向这个方向
        
        # 添加视角扰动
        if self.cfg.view_perturbation > 0:
            perturbation = torch.randn(3) * self.cfg.view_perturbation
            view_direction = view_direction + perturbation
            view_direction = F.normalize(view_direction.unsqueeze(0), dim=-1).squeeze(0)
        
        # 构建相机坐标系
        # 在OpenGL中，相机看向-z方向
        # 如果我们想让相机看向view_direction，那么c2w的z轴应该是-view_direction
        camera_z = -view_direction
        
        # 选择up向量
        up = torch.tensor([0.0, 1.0, 0.0])
        if torch.abs(torch.dot(camera_z, up)) > 0.9:
            up = torch.tensor([1.0, 0.0, 0.0])
        
        # camera_x = up × camera_z
        camera_x = torch.cross(up, camera_z)
        camera_x = F.normalize(camera_x.unsqueeze(0), dim=-1).squeeze(0)
        
        # camera_y = camera_z × camera_x
        camera_y = torch.cross(camera_z, camera_x)
        camera_y = F.normalize(camera_y.unsqueeze(0), dim=-1).squeeze(0)
        
        # 构建c2w矩阵 (OpenGL坐标系)
        c2w = torch.eye(4)
        # 使用列向量表示坐标轴，保持与其他代码一致
        c2w[:3, 0] = camera_x
        c2w[:3, 1] = camera_y
        c2w[:3, 2] = camera_z
        c2w[:3, 3] = camera_position
        
        # 计算相机距离
        camera_distance = torch.norm(camera_position)
        
        # 随机fovy
        fovy_deg = random.uniform(self.fovy_range[0], self.fovy_range[1])
        fovy = torch.deg2rad(torch.tensor(fovy_deg))
        
        return {
            'c2w': c2w,
            'camera_position': camera_position,
            'camera_distance': camera_distance,
            'fovy': fovy,
            'sample_type': sample_type,
        }
    
    def _generate_random_camera_single(self) -> dict:
        """生成单个随机相机位姿（基于uncond_my.py逻辑）
        
        Returns:
            包含相机信息的字典
        """
        # 随机生成相机参数（类似uncond_my.py）
        elevation_deg = random.uniform(self.cfg.elevation_range[0], self.cfg.elevation_range[1])
        azimuth_deg = random.uniform(-180, 180)
        camera_distance = random.uniform(
            self.cfg.camera_distance_range[0], 
            self.cfg.camera_distance_range[1]
        )
        
        elevation = torch.deg2rad(torch.tensor(elevation_deg))
        azimuth = torch.deg2rad(torch.tensor(azimuth_deg))
        
        # 计算相机位置
        if self.cfg.open3d_coord:
            # new system: x right, y up, z back
            # old system: x back, y right, z up
            
            # We are essentially performing this mapping:
            # x_new = y_old
            # y_new = z_old
            # z_new = x_old
            #
            # old x = d * cos(e) * cos(a)
            # old y = d * cos(e) * sin(a)
            # old z = d * sin(e)
            camera_position: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distance * torch.sin(elevation),
                    camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                ],
                dim=-1,
            )
        else:
            camera_position: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distance * torch.sin(elevation),
                ],
                dim=-1,
            )
        
        # 构建lookat矩阵
        center = torch.zeros(3)
        if self.cfg.open3d_coord:
            up: Float[Tensor, "B 3"] = torch.tensor([0, 1, 0], dtype=torch.float32)
        else:
            up: Float[Tensor, "B 3"] = torch.tensor([0, 0, 1], dtype=torch.float32)

        lookat = F.normalize((center - camera_position).unsqueeze(0), dim=-1).squeeze(0)
        right = F.normalize(torch.cross(lookat.unsqueeze(0), up.unsqueeze(0), dim=-1).squeeze(0).unsqueeze(0), dim=-1).squeeze(0)
        up = F.normalize(torch.cross(right.unsqueeze(0), lookat.unsqueeze(0), dim=-1).squeeze(0).unsqueeze(0), dim=-1).squeeze(0)
        
        # 构建c2w矩阵（OpenGL坐标系）
        c2w = torch.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -lookat  # OpenGL: 相机看向-z
        c2w[:3, 3] = camera_position
        
        # 随机fovy
        fovy_deg = random.uniform(self.fovy_range[0], self.fovy_range[1])
        fovy = torch.deg2rad(torch.tensor(fovy_deg))
        
        return {
            'c2w': c2w,
            'camera_position': camera_position,
            'camera_distance': torch.tensor(camera_distance),
            'fovy': fovy,
            'sample_type': 'random',
        }
    
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.true_global_step = global_step
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Refine camera - height: {self.height}, width: self.width, batch_size: {self.batch_size}"
        )

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        # 随机选择采样策略（支持batch_size > 1）
        batch_size = self.batch_size
        
        # 收集所有相机数据
        c2w_list = []
        fovy_list = []
        camera_distance_list = []
        light_positions_list = []
        sample_type_list = []
        
        for _ in range(batch_size):
            # 随机选择采样策略
            rand_val = random.random()
            
            if rand_val < self.tactile_prob:
                # 使用触觉传感器采样
                if len(self.tactile_sensor_positions) > 0:
                    idx = random.randint(0, len(self.tactile_sensor_positions) - 1)
                    anchor_pos = self.tactile_sensor_positions[idx]
                    anchor_dir = self.tactile_sensor_directions[idx]
                    sample_type = 'tactile'
                    camera_info = self._generate_camera_from_anchor(anchor_pos, anchor_dir, sample_type)
                else:
                    threestudio.warn("No tactile sensors available, falling back to random")
                    camera_info = self._generate_random_camera_single()
                    sample_type = 'random'
                    
            elif rand_val < self.tactile_prob + self.mesh_prob:
                # 使用mesh采样
                if len(self.mesh_sample_positions) > 0:
                    idx = random.randint(0, len(self.mesh_sample_positions) - 1)
                    anchor_pos = self.mesh_sample_positions[idx]
                    anchor_dir = self.mesh_sample_normals[idx]
                    sample_type = 'mesh'
                    camera_info = self._generate_camera_from_anchor(anchor_pos, anchor_dir, sample_type)
                else:
                    threestudio.warn("No mesh samples available, falling back to random")
                    camera_info = self._generate_random_camera_single()
                    sample_type = 'random'
                    
            else:
                # 使用随机采样（uncond_my.py逻辑）
                camera_info = self._generate_random_camera_single()
                sample_type = 'random'
            
            c2w = camera_info['c2w']
            fovy = camera_info['fovy']
            camera_distance = camera_info['camera_distance']
            
            # 生成光照位置
            if self.cfg.light_sample_strategy == "dreamfusion":
                light_distance = (
                    self.cfg.light_distance_range[0]
                    + torch.rand(1).item() * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
                )
                # 随机方向
                light_direction = F.normalize(torch.randn(3).unsqueeze(0), dim=-1).squeeze(0)
                light_positions = light_direction * light_distance
            else:
                light_positions = c2w[:3, 3]
            
            # 添加光照扰动
            light_positions = light_positions + torch.randn(3) * self.cfg.light_position_perturb
            
            c2w_list.append(c2w)
            fovy_list.append(fovy)
            camera_distance_list.append(camera_distance)
            light_positions_list.append(light_positions)
            sample_type_list.append(sample_type)
        
        # 堆叠成batch
        c2w_batch = torch.stack(c2w_list, dim=0)  # [B, 4, 4]
        fovy_batch = torch.stack(fovy_list, dim=0)  # [B]
        camera_distances_batch = torch.stack(camera_distance_list, dim=0)  # [B]
        light_positions_batch = torch.stack(light_positions_list, dim=0)  # [B, 3]
        
        # 存储生成的相机位姿
        if self.save_camera_pointcloud and len(self.generated_camera_positions) < self.max_cameras_to_save:
            for c2w in c2w_list:
                camera_pos = c2w[:3, 3]
                camera_dir = -c2w[:3, 2]  # 相机朝向 (OpenGL: -z轴)
                self.generated_camera_positions.append(camera_pos.clone())
                self.generated_camera_directions.append(camera_dir.clone())
            
            # 达到最大数量时保存
            if len(self.generated_camera_positions) >= self.max_cameras_to_save:
                save_path = os.path.join(self.cfg.root_dir if self.cfg.root_dir else ".", "generated_camera_poses.ply")
                self.save_cameras_as_pointcloud(save_path)
                self.save_camera_pointcloud = False  # 只保存一次
        
        # 生成射线 (for each camera in batch)
        focal_lengths = 0.5 * self.height / torch.tan(0.5 * fovy_batch)  # [B]
        
        # directions_unit_focal: [H, W, 3]，是 focal=1.0 时的方向
        # 需要对前两个分量（x, y）除以 focal_length 来缩放，z分量保持不变
        directions_batch = self.directions_unit_focal[None, :, :, :].repeat(
            batch_size, 1, 1, 1
        )  # [B, H, W, 3]
        directions_batch[:, :, :, :2] = (
            directions_batch[:, :, :, :2] / focal_lengths[:, None, None, None]
        )
        
        rays_o_batch, rays_d_batch = get_rays(directions_batch, c2w_batch, keepdim=True)
        
        # 投影矩阵
        proj_mtx = get_projection_matrix(
            fovy_batch, self.width / self.height, 0.1, 100.0
        )
        mvp_mtx = get_mvp_matrix(c2w_batch, proj_mtx)
        
        # 计算方位角和仰角（用于兼容性）
        camera_position = c2w_batch[:, :3, 3]  # [B, 3]
        azimuth = torch.atan2(camera_position[:, 0], camera_position[:, 2])
        elevation = torch.asin(camera_position[:, 1] / (torch.norm(camera_position, dim=-1) + 1e-8))

        batch_output = {
            "rays_o": rays_o_batch,
            "rays_d": rays_d_batch,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_position,
            "c2w": c2w_batch,
            "light_positions": light_positions_batch,
            "elevation": elevation,
            "azimuth": azimuth,
            "camera_distances": camera_distances_batch,
            "height": self.height,
            "width": self.width,
            "fovy": fovy_batch,
        }
        
        return batch_output
    
    def _generate_random_camera(self) -> Dict[str, Any]:
        """后备方案：生成随机相机（支持batch_size > 1）"""
        batch_size = self.batch_size
        
        elevation = torch.rand(batch_size) * 180 - 90
        azimuth = torch.rand(batch_size) * 360 - 180
        camera_distance = torch.ones(batch_size) * 1.0
        
        elevation = torch.deg2rad(elevation)
        azimuth = torch.deg2rad(azimuth)
        
        if self.cfg.open3d_coord:
            # new system: x right, y up, z back
            # old system: x back, y right, z up
            
            # We are essentially performing this mapping:
            # x_new = y_old
            # y_new = z_old
            # z_new = x_old
            #
            # old x = d * cos(e) * cos(a)
            # old y = d * cos(e) * sin(a)
            # old z = d * sin(e)
            camera_position: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distance * torch.sin(elevation),
                    camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                ],
                dim=-1,
            )
        else:
            camera_position: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distance * torch.sin(elevation),
                ],
                dim=-1,
            )
        
        center = torch.zeros_like(camera_position)  # [B, 3]
        if self.cfg.open3d_coord:
            up: Float[Tensor, "B 3"] = torch.tensor([0, 1, 0], dtype=torch.float32)
        else:
            up: Float[Tensor, "B 3"] = torch.tensor([0, 0, 1], dtype=torch.float32)

        # 批量计算lookat矩阵
        lookat = F.normalize(center - camera_position, dim=-1)
        right = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
        up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
        
        c2w = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 4, 4]
        c2w[:, :3, 0] = right
        c2w[:, :3, 1] = up
        c2w[:, :3, 2] = -lookat
        c2w[:, :3, 3] = camera_position
        
        # 随机fovy
        fovy_deg = torch.rand(batch_size) * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]
        fovy = torch.deg2rad(fovy_deg)
        
        focal_lengths = 0.5 * self.height / torch.tan(0.5 * fovy)
        
        # 生成方向，对前两个分量除以focal_length
        directions_batch = self.directions_unit_focal[None, :, :, :].repeat(
            batch_size, 1, 1, 1
        )  # [B, H, W, 3]
        directions_batch[:, :, :, :2] = (
            directions_batch[:, :, :, :2] / focal_lengths[:, None, None, None]
        )
        
        rays_o, rays_d = get_rays(directions_batch, c2w, keepdim=True)
        
        proj_mtx = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 100.0
        )
        mvp_mtx = get_mvp_matrix(c2w, proj_mtx)
        
        light_positions = camera_position  # [B, 3]
        
        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_position,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation,
            "azimuth": azimuth,
            "camera_distances": camera_distance,
            "height": self.height,
            "width": self.width,
            "fovy": fovy,
        }


class RefineCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str):
        self.cfg: RefineCameraDataModuleConfig = cfg
        self.split = split
        
        # # 创建一个新的config副本，将batch_size固定为1用于验证/测试
        # import copy
        # cfg_copy = copy.deepcopy(cfg)
        # cfg_copy.batch_size = 1  # 固定为1用于验证/测试
        
        # 创建迭代数据集用于生成
        self.camera_generator = RefineCameraIterableDataset(cfg)
        
        # 预生成固定数量的相机位姿用于验证/测试
        if split == "val":
            self.n_views = self.cfg.n_val_views
        elif split == "test":
            self.n_views = self.cfg.n_test_views
        else:
            self.n_views = 100
        
        self.cached_cameras = []
        views_generated = 0
        while views_generated < self.n_views:
            # camera_generator.collate(None) 返回一个批次
            # 假设 refine_camera.batch_size = 1
            batch_dict = self.camera_generator.collate(None)
            
            # 获取这个批次的大小 (通常是 1)
            current_batch_size = batch_dict["rays_o"].shape[0]
            
            for i in range(current_batch_size):
                if views_generated >= self.n_views:
                    break
                
                # 创建一个新的字典，用于存储单个样本
                single_sample_dict = {}
                
                # 遍历批次字典中的所有键
                for key, value in batch_dict.items():
                    if isinstance(value, torch.Tensor) and value.shape[0] == current_batch_size:
                        # 如果是张量，并且第一个维度是batch维，就取出第 i 个元素
                        single_sample_dict[key] = value[i]
                    elif isinstance(value, list) and len(value) == current_batch_size:
                        # 如果是列表，也取出第 i 个元素
                        single_sample_dict[key] = value[i]
                    else:
                        # 其他值 (如 height, width) 直接复制
                        single_sample_dict[key] = value
                
                # 就像你的旧代码一样，添加一个整数索引
                single_sample_dict["index"] = views_generated
                
                self.cached_cameras.append(single_sample_dict)
                views_generated += 1
    
    def __len__(self):
        return len(self.cached_cameras)
    
    def __getitem__(self, index):
        self.cached_cameras[index]["index"] = index
        return self.cached_cameras[index]
    
    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


@register("refine-camera-datamodule")
class RefineCameraDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RefineCameraDataModuleConfig, cfg)

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = RefineCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RefineCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RefineCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def save_camera_pointcloud(self, output_path: str = None):
        """手动保存相机位姿点云
        
        Args:
            output_path: 输出文件路径，如果为None则使用默认路径
        """
        if hasattr(self, 'train_dataset'):
            if output_path is None:
                output_path = os.path.join(
                    self.cfg.root_dir if self.cfg.root_dir else ".", 
                    "generated_camera_poses.ply"
                )
            self.train_dataset.save_cameras_as_pointcloud(output_path)
        else:
            threestudio.warn("Train dataset not initialized, call setup() first")

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self):
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )

    def test_dataloader(self):
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self):
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
