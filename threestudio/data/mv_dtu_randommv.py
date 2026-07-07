import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio import register
from threestudio.data.mv_random_multiview import (
    RandomMultiviewCameraDataModuleConfig,
    RandomMultiviewCameraIterableDataset,
)
from threestudio.data.mv_uncond import RandomCameraDataset
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


def get_image(image_filename, alpha_color=None) -> torch.Tensor:
    pil_image = Image.open(image_filename)
    np_image = np.array(pil_image, dtype="uint8")
    assert len(np_image.shape) == 3
    assert np_image.dtype == np.uint8
    assert np_image.shape[2] in [3, 4]
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
        mask = mask[..., 0]
    return mask.astype(np.float32) / 255.0


@dataclass
class DtuRandomMVDataModuleConfig:
    root_dir: str = ""
    json_path: str = "meta_data.json"
    batch_size: int = 4
    n_view: int = 1
    height: int = 384
    width: int = 384
    load_preprocessed: bool = False
    cam_scale_factor: float = 1.00
    max_num_frames: int = 300
    use_mask: bool = True
    box_crop: bool = False
    box_crop_mask_thr: float = 0.4
    box_crop_context: float = 0.3
    train_num_rays: int = -1
    train_views: Optional[list] = None
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    scale_radius: float = 1.0
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 0.0
    render_path: str = "circle"
    include_mono_prior: bool = True
    include_sensor_depth: bool = False
    include_foreground_mask: bool = True
    include_sfm_points: bool = False
    auto_scale_poses: bool = False
    skip_every_for_val_split: int = 8
    train_val_no_overlap: bool = False
    auto_orient: bool = False
    load_dtu_highres: bool = False
    scale_factor: float = 1.0
    camera_distance_range: List[float] = field(default_factory=lambda: [0.5, 4.5])
    orientation_method: str = "up"


class DtuRandomMVDatasetBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: DtuRandomMVDataModuleConfig = cfg

        self.use_mask = self.cfg.use_mask
        assert os.path.exists(self.cfg.root_dir)

        metadata_path = os.path.join(self.cfg.root_dir, self.cfg.json_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

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

            if self.cfg.include_mono_prior:
                depth_path = os.path.join(self.cfg.root_dir, frame["mono_depth_path"])
                depth = _load_npy_depth(depth_path)
            elif self.cfg.include_sensor_depth:
                depth_path = os.path.join(self.cfg.root_dir, frame["sensor_depth_path"])
                depth = _load_npy_depth(depth_path)
            else:
                depth = np.zeros_like(img[..., 0])

            normal = None
            if self.cfg.include_mono_prior:
                normal_path = os.path.join(self.cfg.root_dir, frame["mono_normal_path"])
                normal = np.load(normal_path)
                normal = normal.reshape(H, W, 3)

            if self.cfg.include_foreground_mask:
                mask_path = os.path.join(self.cfg.root_dir, frame["foreground_mask"])
                mask = _load_png_mask(mask_path)
            else:
                mask = np.ones_like(img[..., 0])

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

        extrinsics = np.stack(extrinsics)
        # 这里是把 OpenCV 坐标系 (Right-Down-Forward) 转为 OpenGL (Right-Up-Back) 的相机坐标系
        # 注意：这只是改了相机自身的朝向定义，没有改世界坐标系
        extrinsics[:, 0:3, 1:3] *= -1.0

        self.all_images = torch.from_numpy(np.stack(images)).float()
        self.all_depths = torch.from_numpy(np.stack(depths)).float()
        self.all_origin_normals = torch.from_numpy(np.stack(origin_normals)).float()
        self.all_fg_masks = np.stack(masks)
        self.all_directions = torch.stack(self.all_directions, dim=0)
        self.all_fovys = torch.stack(fovys, dim=0)

        # 1. 基础变换: 将 Y-up 变为 Z-up (绕 X 轴旋转 90 度)
        # Open3D: Right=X, Up=Y, Back=Z
        # Target: Right=X, Back=-Y, Up=Z
        T_base = torch.tensor([
            [1,  0,  0, 0],
            [0,  0, -1, 0],
            [0,  1,  0, 0],
            [0,  0,  0, 1]
        ], dtype=torch.float32)

        # 2. 方位角修正: 绕新 Z 轴旋转
        # 现象: 你原本是 Front(0)，现在变成了 Left。
        # 这意味着你需要把坐标系顺时针或逆时针转 90 度。
        # 尝试修正: 绕 Z 轴旋转 -90 度 (顺时针)
        # 如果修正后变成了 "Right -> Back...", 请把下面的 sin/cos 符号反过来 (+90度)
        theta = -np.pi / 2.0  # -90 度
        c, s = np.cos(theta), np.sin(theta)
        
        R_z_fix = torch.tensor([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ], dtype=torch.float32)

        # 3. 组合变换矩阵
        # 先做基础变换 T_base，再做方位修正 R_z_fix
        # total_trans = R_z_fix @ T_base
        total_trans = torch.matmul(R_z_fix, T_base)
        
        # 4. 应用变换
        c2w_original = torch.from_numpy(extrinsics).float()
        self.all_c2w = torch.matmul(total_trans, c2w_original)
        
        # =========================================================
        # [核心修改] END
        # =========================================================

        # 下面计算法向的逻辑不需要改，因为这里的 rot 已经是从 self.all_c2w (变换后) 里取的了
        # world_normal = R_new @ camera_normal，自然就在新坐标系下了
        if self.cfg.include_mono_prior and len(origin_normals) > 0:
            final_normals = []
            final_extrinsics = self.all_c2w # 使用变换后的 Extrinsics
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

        self.all_positions = self.all_c2w[:, :3, 3]

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomMultiviewCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            random_camera_cfg.n_view = self.cfg.n_view
            if split == "train":
                self.random_pose_generator = RandomMultiviewCameraIterableDataset(random_camera_cfg)
            else:
                self.random_pose_generator = RandomCameraDataset(random_camera_cfg, split)

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
        if hasattr(self, "all_normals"):
            self.all_normals = self.all_normals[i_split[self.split]]

        meta_scene_box = metadata["scene_box"]
        # =========================================================
        # [修改] Scene Box 变换
        # =========================================================
        # 原始的 aabb 是基于旧坐标系的，我们也需要旋转它，或者重新定义一个标准包围盒
        # 如果你的物体是在原点附近，并且经过归一化，通常直接定义一个简单的 Z-up aabb 即可
        # 这里演示如何旋转原始 aabb (假设它是 centered at origin 的)
        
        original_aabb = torch.tensor(meta_scene_box["aabb"], dtype=torch.float32)
        # AABB 有两行: [min_point, max_point]
        # 旋转 AABB 比较麻烦，因为旋转后的 AABB 可能变大。
        # 简单粗暴的方法：直接变换 min 和 max 点，然后取新的 min/max（前提是旋转是90度整数倍）
        # T_aabb = coord_trans @ point
        # 这里为了稳妥，如果你的场景已经归一化到了 [-1, 1]，我们可以直接重置为标准 Z-up AABB
        
        # 建议：既然你用了 SDS，通常希望物体在单位球内。
        # 我们可以不管 json 里的 aabb，直接给一个标准的：
        self.scene_box = {
            "aabb": torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
            "near": 0.1, # 根据你的尺度调整
            "far": 100.0,
            "radius": 1.0,
        }
        # 如果你非常依赖原始 json 里的 near/far，可以保留
        # self.scene_box["near"] = meta_scene_box["near"]
        # self.scene_box["far"] = meta_scene_box["far"]

        # 确保数据都在 GPU 上
        self.all_c2w = self.all_c2w.float().to(self.rank)
        self.all_images = self.all_images.float().to(self.rank)
        self.all_fg_masks = self.all_fg_masks.float().to(self.rank)
        self.all_depths = self.all_depths.float().to(self.rank)
        self.all_origin_normals = self.all_origin_normals.float().to(self.rank)
        if hasattr(self, "all_normals"):
            self.all_normals = self.all_normals.float().to(self.rank)

    def get_all_images(self):
        return self.all_images


class DtuRandomMVDataset(Dataset, DtuRandomMVDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)

    def __len__(self):
        if self.split == "test":
            if self.cfg.render_path == "circle":
                return len(self.random_pose_generator)
            else:
                return len(self.all_images)
        else:
            return len(self.random_pose_generator)

    def prepare_data(self, index):
        c2w = self.all_c2w[index]
        light_positions = c2w[..., :3, -1]
        directions = self.all_directions[index]
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )
        fovy = self.all_fovys[index]
        rgb = self.all_images[index]
        depth = self.all_depths[index]
        normal = self.all_normals[index]
        origin_normal = self.all_origin_normals[index]
        mask = self.all_fg_masks[index]
        camera_distances = torch.norm(c2w[..., :3, -1], dim=-1, keepdim=True)
        camera_distances_relative = camera_distances

        proj_mtx = get_projection_matrix(
            fovy.unsqueeze(0),
            self.cfg.width / self.cfg.height,
            self.scene_box["near"],
            self.scene_box["far"],
        )

        c2w = c2w.to(self.rank)
        proj_mtx = proj_mtx.to(self.rank)
        mvp_mtx = get_mvp_matrix(c2w.unsqueeze(0), proj_mtx.unsqueeze(0))

        batch = {
            "index": index,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": c2w[..., :3, -1],
            "c2w": c2w,
            "light_positions": light_positions,
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
        if self.split == "test":
            if self.cfg.render_path == "circle":
                return self.random_pose_generator[index]
            else:
                return self.prepare_data(index)
        else:
            return self.random_pose_generator[index]


class DtuRandomMVIterableDataset(IterableDataset, DtuRandomMVDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)
        self.idx = 0
        self.image_perm = torch.randperm(len(self.all_images))

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
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
        camera_distances = torch.norm(c2w[..., :3, -1], dim=-1, keepdim=True)
        fovy = self.all_fovys[idx][None]
        camera_distances_relative = camera_distances
        proj_mtx = get_projection_matrix(
            fovy,
            self.cfg.width / self.cfg.height,
            self.scene_box["near"],
            self.scene_box["far"],
        )
        proj_mtx = proj_mtx.to(self.rank)
        c2w = c2w.to(self.rank)
        mvp_mtx = get_mvp_matrix(c2w, proj_mtx)

        if (
            self.cfg.train_num_rays != -1
            and self.cfg.train_num_rays < self.cfg.height * self.cfg.width
        ):
            _, height, width, _ = rays_o.shape
            x = torch.randint(0, width, size=(self.cfg.train_num_rays,), device=rays_o.device)
            y = torch.randint(0, height, size=(self.cfg.train_num_rays,), device=rays_o.device)
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

        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        self.idx += 1
        if self.idx == len(self.all_images):
            self.idx = 0
            self.image_perm = torch.randperm(len(self.all_images))

        return batch


@register("mv-dtu-randommv-datamodule")
class DtuRandomMVDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(DtuRandomMVDataModuleConfig, cfg)

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = DtuRandomMVIterableDataset(self.cfg, self.cfg.train_split)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = DtuRandomMVDataset(self.cfg, self.cfg.val_split)
        if stage in [None, "test", "predict"]:
            self.test_dataset = DtuRandomMVDataset(self.cfg, self.cfg.test_split)

    def prepare_data(self):
        pass

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
