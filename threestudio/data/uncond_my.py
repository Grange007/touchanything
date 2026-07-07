import bisect
import math
import pytorch_lightning as pl
import random
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset, IterableDataset
import pdb
import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (get_mvp_matrix, get_projection_matrix,
                                   get_ray_directions, get_rays,)
from threestudio.utils.typing import *
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

@dataclass
class RandomCameraDataModuleConditionConfig:
    relative_radius: bool = False
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 64
    width: Any = 64
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    n_view: int = 1  # number of views per sample
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    zoom_range: Tuple[float, float] = (1.0, 1.0)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy
    ele_random_prob: float = 0.5
    open3d_coord: bool = False  # whether to use Open3D coordinate system (y up, x right, z back)
    tactile_sensor_positions: List[Any] = field(default_factory=list)
    tactile_sensor_directions: List[Any] = field(default_factory=list)
    tactile_sensor_radius: float = 0.2 
    density_aware_sampling: bool = False  
    # min_density_weight: float = 0.5 
    # max_density_weight: float = 2.0 
    density_transition_steps: int = 5000  


class RandomCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConditionConfig = cfg
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
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range
        self.zoom_range = self.cfg.zoom_range
        self.ele_random_prob = self.cfg.ele_random_prob
        # print("camera distance range:", self.camera_distance_range)
        
        self.tactile_sensor_positions = torch.tensor(
            self.cfg.tactile_sensor_positions, 
            dtype=torch.float32
        )
        self.tactile_sensor_directions = -torch.tensor(
            self.cfg.tactile_sensor_directions, 
            dtype=torch.float32
        )
        focal_length = 320
        ppmm = 0.0634
        move_distance = focal_length * ppmm / 1000.0 * 20
        normalized_directions = F.normalize(self.tactile_sensor_directions, dim=-1)
        displacements = normalized_directions * move_distance
        self.tactile_sensor_positions += displacements

        self.true_global_step = 0
        
        self.visualization_resolution = 50
        if self.cfg.density_aware_sampling:
            self.visualize_density_field()
        
    def visualize_density_field(self, output_path="density_field.ply", global_step=0):
        """
        
        Args:
            output_path: 输出文件路径
            global_step: 当前训练步数，用于文件名
        """
        if len(self.tactile_sensor_positions) == 0:
            print("No tactile sensor positions available for visualization")
            return
        
        # 确定可视化范围
        sensor_positions = self.tactile_sensor_positions.cpu().numpy()
        
        # 使用camera_distance_range定义的范围
        min_bound = self.cfg.camera_distance_range[0] * np.ones(3)
        max_bound = self.cfg.camera_distance_range[1] * np.ones(3)
        x = np.linspace(-self.camera_distance_range[1], self.camera_distance_range[1], self.visualization_resolution)
        y = np.linspace(-self.camera_distance_range[1], self.camera_distance_range[1], self.visualization_resolution)
        z = np.linspace(-self.camera_distance_range[1], self.camera_distance_range[1], self.visualization_resolution)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        
        # 计算每个点到原点的距离
        distances = np.linalg.norm(grid_points, axis=1)
        
        # 筛选在camera_distance_range范围内的点
        valid_indices = np.where((distances >= self.camera_distance_range[0]) & (distances <= self.camera_distance_range[1]))[0]
        valid_grid_points = grid_points[valid_indices]
        
        grid_points_tensor = torch.from_numpy(valid_grid_points).float().to(self.tactile_sensor_positions.device)
        densities = self._compute_density_at_positions(grid_points_tensor, fx=0.5 * self.height / torch.tan(torch.tensor(0.5 * (0.5 * (self.fovy_range[0] + self.fovy_range[1]) * math.pi / 180))))
        densities_np = densities.cpu().numpy()
        print("Computed densities for", len(densities_np), "points")
        print("Density stats - min:", densities_np.min(), "max:", densities_np.max(), "mean:", densities_np.mean())
        if densities_np.max() > 0:
            normalized_densities = densities_np / densities_np.max()
        else:
            normalized_densities = np.zeros_like(densities_np)
        
        colormap = plt.cm.get_cmap('jet')
        colors = colormap(normalized_densities)[:, :3]  # 忽略alpha通道
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_grid_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        output_file = f"density_field_step_{global_step:06d}.ply" if global_step > 0 else output_path
        o3d.io.write_point_cloud(output_file, pcd)
        
        print(f"Density field visualized and saved to {output_file}")
        
        self._visualize_tactile_sensors(output_file.replace(".ply", "_sensors.ply"))
        
        # pdb.set_trace()
        return pcd
    
    def _visualize_tactile_sensors(self, output_path):
        if len(self.tactile_sensor_positions) == 0:
            return
            
        sensor_positions = self.tactile_sensor_positions.cpu().numpy()
        sensor_directions = self.tactile_sensor_directions.cpu().numpy()
        sensor_radius = self.cfg.tactile_sensor_radius

        # 创建传感器圆盘
        meshes = []
        for i, pos in enumerate(sensor_positions):
            # 创建圆盘表示传感器（使用很矮的圆柱体）
            disk_height = sensor_radius * 0.1  # 圆盘高度
            disk = o3d.geometry.TriangleMesh.create_cylinder(
                radius=sensor_radius, 
                height=disk_height
            )
            
            # 旋转圆盘使其朝向传感器方向
            default_direction = np.array([0, 0, 1])  # 默认圆柱体方向
            rotation = self._rotation_matrix_between_vectors(default_direction, sensor_directions[i])
            disk.rotate(rotation, center=np.array([0, 0, 0]))
            
            # 移动圆盘到正确位置
            disk.translate(pos)
            disk.paint_uniform_color([0, 1, 0])  # 绿色
            
            # 创建方向指示器（小锥体）
            direction = sensor_directions[i]
            direction = direction / np.linalg.norm(direction)  # 归一化
            
            # 创建锥体表示方向
            cone_height = sensor_radius * 0.8
            cone_radius = sensor_radius * 0.3
            cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
            
            # 旋转锥体使其指向正确方向
            cone.rotate(rotation, center=np.array([0, 0, 0]))
            
            # 移动锥体到正确位置（从圆盘中心延伸）
            cone.translate(pos + direction * (disk_height/2 + cone_height/2))
            cone.paint_uniform_color([1, 0, 0])  # 红色
            
            meshes.append(disk)
            meshes.append(cone)

        # 合并所有网格
        combined_mesh = meshes[0]
        for mesh in meshes[1:]:
            combined_mesh += mesh

        o3d.io.write_triangle_mesh(output_path, combined_mesh)

        print(f"Tactile sensors visualized and saved to {output_path}")

    def _rotation_matrix_between_vectors(self, a, b):
        """计算从向量a到向量b的旋转矩阵"""
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        
        v = np.cross(a, b)
        c = np.dot(a, b)
        
        # 处理平行向量
        if np.allclose(v, 0):
            return np.eye(3) if c > 0 else -np.eye(3)
        
        # 使用Rodrigues公式计算旋转矩阵
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        
        R = np.eye(3) + vx + np.dot(vx, vx) * (1 / (1 + c))
        return R
    
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.true_global_step = global_step
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        )
        # progressive view
        self.progressive_view(global_step)

    def __iter__(self):
        while True:
            yield {}

    # def _compute_density_at_positions(self, positions: Float[Tensor, "B 3"], fx) -> Float[Tensor, "BS"]:
    #     if len(self.tactile_sensor_positions) == 0:
    #         return torch.zeros(positions.shape[0], device=get_device())
    #     sensor_radius = self.cfg.tactile_sensor_radius
    #     sensor_positions = self.tactile_sensor_positions
    #     sensor_directions = -self.tactile_sensor_directions
    #     positions_expanded = positions.unsqueeze(1)
    #     sensor_positions_expanded = sensor_positions.unsqueeze(0)
    #     sensor_directions_expanded = sensor_directions.unsqueeze(0)
    #     cam_to_sensor = sensor_positions_expanded - positions_expanded
    #     distances = torch.norm(cam_to_sensor, dim=-1)
    #     cam_directions = -positions_expanded / torch.norm(positions_expanded, dim=-1, keepdim=True)
    #     sensor_normals = sensor_directions_expanded / torch.norm(sensor_directions_expanded, dim=-1, keepdim=True)
    #     print("cam_directions shape:", cam_directions.shape, "sensor_normals shape:", sensor_normals.shape)
        
    #     normal_view_angles = torch.acos(torch.clamp(
    #         torch.sum(cam_directions * sensor_normals, dim=-1), -1.0, 1.0))
        
    #     sensor_plane_angles = torch.pi/2 - normal_view_angles
        
    #     cam_to_sensor_dirs = cam_to_sensor / distances.unsqueeze(-1)
        
    #     normal_sensor_angles = torch.acos(torch.clamp(
    #         torch.sum(sensor_normals * cam_to_sensor_dirs, dim=-1), -1.0, 1.0))
        
    #     plane_sensor_angles = torch.pi/2 - normal_sensor_angles

    #     projection_factor = torch.cos(sensor_plane_angles) * torch.cos(plane_sensor_angles)
        
    #     base_area = torch.pi * (sensor_radius / distances * fx) ** 2
    #     projected_areas = base_area * projection_factor
        
    #     facing_camera = normal_view_angles < (torch.pi / 2)
    #     projected_areas = projected_areas * facing_camera.float()
        
    #     densities = torch.sum(projected_areas, dim=-1)
        
    #     return densities

    def _compute_density_at_positions(self, positions: Float[Tensor, "B 3"], fx) -> Float[Tensor, "BS"]:
        density_max = torch.pi * (self.cfg.tactile_sensor_radius ** 2) * len(self.tactile_sensor_positions)
        
        if len(self.tactile_sensor_positions) == 0:
            return torch.zeros(positions.shape[0], device=get_device())
        
        # 触觉传感器参数
        sensor_radius = self.cfg.tactile_sensor_radius
        sensor_positions = self.tactile_sensor_positions
        sensor_directions = self.tactile_sensor_directions
        
        # 扩展维度
        positions_expanded = positions.unsqueeze(1)  # [B, 1, 3]
        sensor_positions_expanded = sensor_positions.unsqueeze(0)  # [1, N, 3]
        sensor_directions_expanded = sensor_directions.unsqueeze(0)  # [1, N, 3]
        
        # 计算相机到传感器的向量
        cam_to_sensor = sensor_positions_expanded - positions_expanded  # [B, N, 3]
        
        # 计算相机方向（相机看向原点的方向）
        # 注意：在正交投影中，我们使用固定的投影方向（相机看向原点的方向）
        cam_directions = -positions_expanded / torch.norm(positions_expanded, dim=-1, keepdim=True)  # [B, 1, 3]
        
        # 计算传感器法向量
        sensor_normals = sensor_directions_expanded / torch.norm(sensor_directions_expanded, dim=-1, keepdim=True)  # [1, N, 3]
        
        # 计算传感器法向量与相机方向的夹角
        normal_view_angles = torch.acos(torch.clamp(
            torch.sum(cam_directions * sensor_normals, dim=-1), -1.0, 1.0))  # [B, N]
        
        # 计算传感器平面与相机方向的夹角
        # plane_view_angles = torch.pi/2 - normal_view_angles  # [B, N]
        
        # 在正交投影下，投影面积只与传感器尺寸和方向有关，与距离无关
        # 计算投影面积（考虑传感器方向与投影方向的夹角）
        projected_areas = torch.pi * (sensor_radius ** 2) * torch.cos(normal_view_angles)  # [B, N]
        
        facing_camera = normal_view_angles < (torch.pi / 2)  # [B, N]
        same_hemisphere = torch.sum(-positions_expanded * sensor_positions, dim=-1) < 0  # [B, N]
        valid_view = facing_camera & same_hemisphere  # [B, N]
        projected_areas = projected_areas * valid_view.float()  # [B, N]
        
        # 对每个相机位置求和
        densities = torch.sum(projected_areas, dim=-1)  # [B]
        densities = densities / density_max  # 归一化到[0, 1]
        return densities



    def progressive_view(self, global_step):
        if global_step == 0:
            r = 1
        else:
            r = min(1.0, global_step / (self.cfg.progressive_until + 1))
        self.elevation_range = [
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[0],
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[1],
        ]
        self.azimuth_range = [
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[0],
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[1],
        ]
        # self.camera_distance_range = [
        #     (1 - r) * self.cfg.eval_camera_distance
        #     + r * self.cfg.camera_distance_range[0],
        #     (1 - r) * self.cfg.eval_camera_distance
        #     + r * self.cfg.camera_distance_range[1],
        # ]
        # self.fovy_range = [
        #     (1 - r) * self.cfg.eval_fovy_deg + r * self.cfg.fovy_range[0],
        #     (1 - r) * self.cfg.eval_fovy_deg + r * self.cfg.fovy_range[1],
        # ]

    def collate(self, batch) -> Dict[str, Any]:
        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < self.ele_random_prob:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1, 1)
            + torch.arange(self.cfg.n_view).reshape(1, -1)
        ).reshape(-1) / self.cfg.n_view * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth = azimuth_deg * math.pi / 180
        
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        camera_distances_relative = camera_distances
        if self.cfg.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.zoom_range[1] - self.zoom_range[0])
            + self.zoom_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom
        ###########################################

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
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
            camera_positions: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distances * torch.sin(elevation),
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                ],
                dim=-1,
            )
        else:
            camera_positions: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distances * torch.sin(elevation),
                ],
                dim=-1,
            )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        if self.cfg.open3d_coord:
            up: Float[Tensor, "B 3"] = torch.as_tensor([0, 1, 0], dtype=torch.float32)[
                None, :
            ].repeat(self.batch_size, 1)
        else:
            up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                None, :
            ].repeat(self.batch_size, 1)
        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.center_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.up_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        up = up + up_perturb


        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(real_batch_size, 3).repeat_interleave(
                    self.cfg.n_view, dim=0
                )
                * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(real_batch_size) * math.pi - 2 * math.pi
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(real_batch_size) * math.pi / 3 + math.pi / 6
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )
        
        density_weights = torch.ones(self.batch_size, device=get_device())
        if self.cfg.density_aware_sampling and len(self.tactile_sensor_positions) > 0:
            densities = self._compute_density_at_positions(camera_positions, fx=focal_length)
            progress = min(1.0, self.true_global_step / (self.cfg.density_transition_steps + 1))
            if progress < 1.0:
                density_weights = densities
                # density_weights = self.cfg.min_density_weight + densities * (self.cfg.max_density_weight - self.cfg.min_density_weight)
            else:
                density_weights = torch.ones_like(densities)

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "camera_distances_relative": camera_distances_relative,
            "height": self.height,
            "width": self.width,
            "fovy": fovy_deg,
            "density_weights": density_weights
        }


class RandomCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConditionConfig = cfg
        self.split = split

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(0.0, 360.0, self.n_views + 1)[: self.n_views]
        else:
            azimuth_deg = torch.linspace(0.0, 360.0, self.n_views)
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
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
            camera_positions: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distances * torch.sin(elevation),
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                ],
                dim=-1,
            )
        else:
            camera_positions: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distances * torch.sin(elevation),
                ],
                dim=-1,
            )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        if self.cfg.open3d_coord:
            up: Float[Tensor, "B 3"] = torch.as_tensor([0, 1, 0], dtype=torch.float32)[
                None, :
            ].repeat(self.cfg.eval_batch_size, 1)
        else:
            up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                None, :
            ].repeat(self.cfg.eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances
        self.fovy = fovy

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width,
            "fovy": self.cfg.eval_fovy_deg
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


@register("random-camera-condition-datamodule")
class RandomCameraDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConditionConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConditionConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
