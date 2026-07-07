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
from threestudio.data.uncond_my import (
    RandomCameraDataModuleConditionConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.config import parse_structured
from threestudio.utils.base import Updateable
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
class DtuDataModuleConfig:
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
    
    # New options from SDFStudio parser
    include_mono_prior: bool = True
    include_sensor_depth: bool = False
    include_foreground_mask: bool = True  # Enabled by default for DTU
    include_sfm_points: bool = False
    auto_scale_poses: bool = False
    # orientation_method: Literal["up", "none"] = "up"
    skip_every_for_val_split: int = 8
    train_val_no_overlap: bool = False
    auto_orient: bool = False
    load_dtu_highres: bool = False
    scale_factor: float = 1.0
    camera_distance_range: List[float] = field(
        default_factory=lambda: [0.5, 4.5]
    )
    orientation_method: str = "up"  # Options: "up", "none"
    stage1_batch_size: Optional[int] = None
    stage2_batch_size: Optional[int] = None

class DtuDatasetBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: DtuDataModuleConfig = cfg

       

        self.use_mask = self.cfg.use_mask
        cam_scale_factor = self.cfg.cam_scale_factor

        assert os.path.exists(self.cfg.root_dir), f"{self.cfg.root_dir} doesn't exist!"
        
        metadata_path = os.path.join(self.cfg.root_dir, self.cfg.json_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        worldtogt = np.array(metadata["worldtogt"])

        frames = metadata["frames"]

        images, depths, origin_normals, normals, intrinsics, extrinsics, masks = [], [], [], [], [], [], []
        fovys = []
        self.all_directions = []
        self.all_fg_masks = []
        
        for frame in frames:
            
            H = self.cfg.height
            W = self.cfg.width
            
            rgb_path = os.path.join(self.cfg.root_dir, frame["rgb_path"])
            img = get_image(rgb_path).numpy()
            
            camtoworld = np.array(frame["camtoworld"])
            # camtoworld[:3, :3] *= -1.0  # OpenCV to OpenGL
            # camtoworld[:, [1, 2]] = camtoworld[:, [2, 1]]  # Swap y and z axes
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
            normal_map = None
            if self.cfg.include_mono_prior:
                normal_path = os.path.join(self.cfg.root_dir, frame["mono_normal_path"])
                normal = np.load(normal_path)
                # normal = normal * 2.0 - 1.0
                normal = normal.reshape(H, W, 3)
                # rot = camtoworld[:3, :3]
                # normal_map = normal.reshape(-1, 3).T
                # normal_map = torch.from_numpy(normal_map).float()
                # normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)
                # normal_map = torch.from_numpy(rot).float() @ normal_map
                # normal_map = normal_map.T
                # print(f"normal shape: {normal.shape}, normal_map shape: {normal_map.shape}")
                # normal_map = normal_map.reshape(H, W, 3)
                # normal_map = normal_map.numpy()

                # normal_map = normal.reshape(3, -1)
                # normal_map = torch.from_numpy(normal_map).float()
                # normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)
                # normal_map = torch.from_numpy(rot).float() @ normal_map
                # normal_map = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)
                # normal_map = normal_map.numpy()
                
                # print(f"normal_map shape: {normal_map.shape}, normal shape: {normal.shape}")
                # save the old and new normal maps
                # base_filename = os.path.splitext(os.path.basename(frame["mono_normal_path"]))[0]
                # save_dir = os.path.join(self.cfg.root_dir, "normals_debug") # 建议换个目录名以防冲突
                # os.makedirs(save_dir, exist_ok=True)
                # old_normal_save_path = os.path.join(save_dir, f"{base_filename}_old.png")
                # new_normal_save_path = os.path.join(save_dir, f"{base_filename}_new_world.png")

                # # (normal * 127.5 + 127.5) 是将 [-1, 1] 范围的法线转换到 [0, 255] 的颜色范围
                # cv2.imwrite(old_normal_save_path, (normal * 127.5 + 127.5).astype(np.uint8))
                # cv2.imwrite(new_normal_save_path, (normal_map * 127.5 + 127.5).astype(np.uint8))
            
            if self.cfg.include_foreground_mask:
                mask_path = os.path.join(self.cfg.root_dir, frame["foreground_mask"])
                mask = _load_png_mask(mask_path)
            else:
                mask = np.ones_like(img[..., 0])
            
            
            
            # camtoworld = worldtogt @ camtoworld
            # camtoworld[:3, 3] *= cam_scale_factor
            
            
            
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
        extrinsics[:, 0:3, 1:3] *= -1.0  # OpenCV to OpenGL
        self.all_images = torch.from_numpy(np.stack(images)).float()
        self.all_depths = torch.from_numpy(np.stack(depths)).float()
        self.all_origin_normals = torch.from_numpy(np.stack(origin_normals)).float()
        # self.all_normals = torch.from_numpy(np.stack(normals)).float()
        intrinsics = np.stack(intrinsics)
        self.all_fg_masks = np.stack(masks)
        
        if self.cfg.include_mono_prior and len(origin_normals) > 0:
            final_normals = []
            final_extrinsics = torch.from_numpy(extrinsics).float() 

            # all_origin_normals_tensor = torch.from_numpy(np.stack(origin_normals)).float() # 假设 origin_normals 是 list of numpy

            for i in range(len(final_extrinsics)):
                rot = final_extrinsics[i, :3, :3] 
                cam_space_normal = self.all_origin_normals[i] # self.all_origin_normals 已经是 tensor

                H, W, _ = cam_space_normal.shape

                normal_map = cam_space_normal.reshape(-1, 3).T
                normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

                world_space_normal = rot @ normal_map

                world_space_normal = world_space_normal.T.reshape(H, W, 3)
                final_normals.append(world_space_normal)

        self.all_normals = torch.stack(final_normals)
        
        
        # flip_matrix = np.array([
        #     [ 1,  0,  0,  0],
        #     [ 0,  0,  1,  0],
        #     [ 0,  1,  0,  0],
        #     [ 0,  0,  0,  1]
        # ])
        # extrinsics = extrinsics @ flip_matrix  # Swap y and z axes
        # extrinsics *= -1.0
           # extrinsics[:, 0:3, 1] *= -1.0
        # extrinsics[:, 0:3, 3] = extrinsics[:, 0:3, 3] @ np.diag([1, -1, 1])

        # if self.cfg.auto_orient:
        #     extrinsics, transform = camera_utils.auto_orient_and_center_poses(
        #         torch.from_numpy(extrinsics).float(),
        #         method=self.cfg.orientation_method,
        #         center_poses=True,
        #     )
        #     extrinsics = extrinsics.numpy()
        #     transform = transform.numpy()
            
        #     if self.cfg.include_mono_prior and len(normals) > 0:
        #         normals_aligned = []
        #         for normal in normals:
        #             h, w, _ = normal.shape
        #             normal = transform[:3, :3] @ normal.reshape(-1, 3).T
        #             normal = normal.T.reshape(h, w, 3)
        #             normals_aligned.append(normal)
        #         self.all_normals = torch.from_numpy(np.stack(normals_aligned)).float()
        
        # scale_factor = 1.0
        # if self.cfg.auto_scale_poses:
        #     scale_factor /= float(np.max(np.abs(extrinsics[:, :3, 3])))
        # scale_factor *= self.cfg.scale_factor
        # extrinsics[:, :3, 3] *= scale_factor
        
        self.all_c2w = torch.from_numpy(extrinsics).float()
        self.all_fovys = torch.stack(fovys, dim=0)
        self.all_directions = torch.stack(self.all_directions, dim=0)
        self.all_positions = self.all_c2w[:, :3, 3]
        
        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConditionConfig, self.cfg.get("random_camera", {})
            )
            random_camera_cfg.tactile_sensor_positions = self.all_c2w[:, :3, 3].cpu().numpy().tolist()
            random_camera_cfg.tactile_sensor_directions = self.all_c2w[:, :3, 2].cpu().numpy().tolist()
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )
        
        
        num_frames = len(frames)
        indices = list(range(num_frames))
        
        if split != "train" and self.cfg.skip_every_for_val_split >= 1:
            indices = indices[:: self.cfg.skip_every_for_val_split]
        elif self.cfg.train_val_no_overlap:
            indices = [i for i in indices if i % self.cfg.skip_every_for_val_split != 0]
        
        i_split = {"train": indices, "val": indices, "test": indices}
        
        self.all_images = self.all_images[i_split[self.split]]
        self.all_c2w = self.all_c2w[i_split[self.split]]
        self.all_fovys = self.all_fovys[i_split[self.split]]
        self.all_positions = self.all_positions[i_split[self.split]].to(self.rank)
        self.all_directions = self.all_directions[i_split[self.split]].to(self.rank)
        self.all_fg_masks = torch.from_numpy(self.all_fg_masks)[i_split[self.split]]
        self.all_depths = self.all_depths[i_split[self.split]]
        self.all_origin_normals = self.all_origin_normals[i_split[self.split]]
        if hasattr(self, 'all_normals'):
            self.all_normals = self.all_normals[i_split[self.split]]
        
        meta_scene_box = metadata["scene_box"]
        self.scene_box = {
            "aabb": torch.tensor(meta_scene_box["aabb"], dtype=torch.float32),
            "near": meta_scene_box["near"],
            "far": meta_scene_box["far"],
            "radius": meta_scene_box["radius"]
        }
        
        self.all_c2w = self.all_c2w.float().to(self.rank)
        self.all_images = self.all_images.float().to(self.rank)
        self.all_fg_masks = self.all_fg_masks.float().to(self.rank)
        self.all_depths = self.all_depths.float().to(self.rank)
        self.all_origin_normals = self.all_origin_normals.float().to(self.rank)
        self.all_fovys = self.all_fovys.float().to(self.rank)
        if hasattr(self, 'all_normals'):
            self.all_normals = self.all_normals.float().to(self.rank)

    def get_all_images(self):
        return self.all_images

class DtuDataset(Dataset, DtuDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)

    def __len__(self):
        if self.split == "train":
            if self.cfg.use_random_camera and hasattr(self, "random_pose_generator"):
                return len(self.random_pose_generator)
            return len(self.all_images)

        if self.split == "test" and self.cfg.render_path == "circle":
            if self.cfg.use_random_camera and hasattr(self, "random_pose_generator"):
                return len(self.random_pose_generator)

        # val (and non-circle test) use dataset camera poses directly.
        return len(self.all_images)

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
        )
        
        c2w = c2w.to(self.rank)
        proj_mtx = proj_mtx.to(self.rank)
        # proj_mtx is already [1, 4, 4] for a single view; avoid adding an
        # extra singleton dim here, otherwise DataLoader will produce 5D mvp.
        mvp_mtx = get_mvp_matrix(c2w.unsqueeze(0), proj_mtx).squeeze(0)
        
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
        if self.split == "train":
            if self.cfg.use_random_camera and hasattr(self, "random_pose_generator"):
                return self.random_pose_generator[index]
            return self.prepare_data(index)

        if self.split == "test" and self.cfg.render_path == "circle":
            if self.cfg.use_random_camera and hasattr(self, "random_pose_generator"):
                return self.random_pose_generator[index]

        # val (and non-circle test) use dataset camera poses directly.
        return self.prepare_data(index)

class DtuIterableDataset(IterableDataset, DtuDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)
        self.idx = 0
        self.image_perm = torch.randperm(len(self.all_images))

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        num_images = len(self.all_images)

        # stage1: sampled rays, stage2: full image
        if self.cfg.train_num_rays == -1:
            selected_batch_size = self.cfg.stage2_batch_size
        else:
            selected_batch_size = self.cfg.stage1_batch_size

        if selected_batch_size is None:
            selected_batch_size = self.cfg.batch_size

        if selected_batch_size == -1:
            num_sample_images = num_images
        else:
            num_sample_images = max(1, min(int(selected_batch_size), num_images))

        if self.idx + num_sample_images <= num_images:
            idx_cpu = self.image_perm[self.idx : self.idx + num_sample_images]
            self.idx += num_sample_images
            if self.idx == num_images:
                self.idx = 0
                self.image_perm = torch.randperm(num_images)
        else:
            idx_head = self.image_perm[self.idx :]
            self.image_perm = torch.randperm(num_images)
            remain = num_sample_images - idx_head.shape[0]
            idx_tail = self.image_perm[:remain]
            idx_cpu = torch.cat([idx_head, idx_tail], dim=0)
            self.idx = remain

        idx = idx_cpu
        c2w = self.all_c2w[idx]
        light_positions = c2w[..., :3, -1]
        directions = self.all_directions[idx]
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )
        rgb = self.all_images[idx]
        depth = self.all_depths[idx]
        normal = self.all_normals[idx]
        origin_normal = self.all_origin_normals[idx]
        mask = self.all_fg_masks[idx]
        camera_distances = torch.norm(c2w[..., :3, -1], dim=-1, keepdim=True)
        fovy = self.all_fovys[idx]
        camera_distances_relative = camera_distances
        
        # Dynamic near/far calculation
        import math
        radius = self.scene_box["radius"]
        dynamic_near = max(
            camera_distances.min().item() - radius * math.sqrt(3),
            0.01
        )
        dynamic_far = camera_distances.max().item() + radius * math.sqrt(3)
        
        near_plane = dynamic_near
        far_plane = dynamic_far
        
        proj_mtx = get_projection_matrix(
            fovy,
            self.cfg.width / self.cfg.height,
            near_plane,
            far_plane,
        )
        # print(proj_mtx)
        proj_mtx = proj_mtx.to(self.rank)
        c2w = c2w.to(self.rank)
        mvp_mtx = get_mvp_matrix(c2w, proj_mtx)
        # print("rgb shape:", rgb.shape)
        # print("depth shape:", depth.shape)
        # print("mask shape:", mask.shape)
        
        if (
            self.cfg.train_num_rays != -1
            and self.cfg.train_num_rays < self.cfg.height * self.cfg.width
        ):
            num_views, height, width, _ = rays_o.shape
            rays_per_view = max(1, self.cfg.train_num_rays // num_views)

            x = torch.randint(
                0, width, size=(num_views, rays_per_view), device=rays_o.device
            )
            y = torch.randint(
                0, height, size=(num_views, rays_per_view), device=rays_o.device
            )
            view_ids = torch.arange(num_views, device=rays_o.device).unsqueeze(-1)

            rays_o = rays_o[view_ids, y, x].unsqueeze(-2)
            rays_d = rays_d[view_ids, y, x].unsqueeze(-2)
            rgb = rgb[view_ids, y, x].unsqueeze(-2)
            mask = mask[view_ids, y, x].unsqueeze(-1)
            depth = depth[view_ids, y, x].unsqueeze(-1)
            normal = normal[view_ids, y, x].unsqueeze(-2)
            origin_normal = origin_normal[view_ids, y, x].unsqueeze(-2)
        
        mask = mask.unsqueeze(-1)
        depth = depth.unsqueeze(-1)
        
        # print("rays_o shape:", rays_o.shape, "rays_d shape:", rays_d.shape, "rgb shape:", rgb.shape, "mask shape:", mask.shape, "depth shape:", depth.shape, "normal shape:", normal.shape)

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

        return batch

@register("dtu-datamodule-plane-batch-visualize")
class DtuDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(DtuDataModuleConfig, cfg)

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = DtuIterableDataset(self.cfg, self.cfg.train_split)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = DtuDataset(self.cfg, self.cfg.val_split)
        if stage in [None, "test", "predict"]:
            self.test_dataset = DtuDataset(self.cfg, self.cfg.test_split)

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
