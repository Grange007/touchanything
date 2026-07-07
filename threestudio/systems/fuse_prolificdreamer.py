import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from threestudio.systems.pc_project import point_e, render_depth_from_cloud
from threestudio.systems.pytorch3d.renderer import PointsRasterizationSettings
import open3d as o3d
import numpy as np
import torchvision
from typing import Any, Dict
from collections import namedtuple

PointCloud = namedtuple('PointCloud', ['coords', 'channels'])
def read_ply_to_custom_format(ply_filepath: str):
    """
    讀取 .ply 檔案，並將其轉換為自訂的 PointCloud 格式。

    Args:
        ply_filepath (str): .ply 檔案的路徑。

    Returns:
        PointCloud: 一個包含 .coords 和 .channels 屬性的物件。
                    如果檔案讀取失敗，則返回 None。
    """
    try:
        # 2. 使用 open3d 讀取 .ply 檔案
        pcd = o3d.io.read_point_cloud(ply_filepath)
        # downsample the point cloud if too large
        if len(pcd.points) > 10000:
            pcd = pcd.random_down_sample(10000 / len(pcd.points))

        if not pcd.has_points():
            print(f"警告：檔案 {ply_filepath} 無法讀取或不包含任何點。")
            return None

        # 3. 將 open3d 的點座標轉換為 NumPy 陣列
        # pcd.points 的類型是 open3d.utility.Vector3dVector
        coords_np = np.asarray(pcd.points).astype(np.float32)
        # coords_np = coords_np * 2.0
        center = coords_np.mean(axis=0)
        coords_np -= center

        # 2. 计算点云到原点的最大距离
        max_dist = np.max(np.linalg.norm(coords_np, axis=1))
        
        # 3. 将所有点缩放，使得整个点云包含在一个半径为0.5的球体内
        #    (即所有坐标都在 [-0.5, 0.5] 的范围内)
        coords_np /= (max_dist * 2.0)
        # 4. 處理顏色數據
        channels_dict = {}
        if pcd.has_colors():
            # 將 open3d 的顏色轉換為 NumPy 陣列，其形狀為 [N, 3]
            colors_np = np.asarray(pcd.colors)
            
            # 將 [N, 3] 的顏色陣列拆分為三個獨立的 R, G, B 通道 (一維陣列)
            channels_dict['R'] = colors_np[:, 0]
            channels_dict['G'] = colors_np[:, 1]
            channels_dict['B'] = colors_np[:, 2]
        else:
            # 如果點雲沒有顏色資訊,把颜色赋值为黑色
            num_points = coords_np.shape[0]
            channels_dict['R'] = np.zeros(num_points)
            channels_dict['G'] = np.zeros(num_points)
            channels_dict['B'] = np.zeros(num_points)
            

        # 5. 使用提取出的數據，組裝成我們定義的 PointCloud 物件並返回
        custom_point_cloud = PointCloud(coords=coords_np, channels=channels_dict)
        return custom_point_cloud

    except Exception as e:
        print(f"讀取或處理檔案時發生錯誤：{e}")
        return None


@threestudio.register("fuse-prolificdreamer-system")
class ProlificDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'geometry', 'texture']
        stage: str = "coarse"
        visualize_samples: bool = False
        threefuse: bool = True
        image_dir: str = "hello"
        ply_file: Optional[str] = None

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        
        self.threefuse = self.cfg.threefuse
        self.image_dir = self.cfg.image_dir
        
        if self.threefuse is True and self.cfg.ply_file:
            # self.cond_pc = point_e(device="cuda", exp_dir=self.image_dir)
            ply_file = self.cfg.ply_file
            self.cond_pc = read_ply_to_custom_format(ply_file)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.stage == "geometry":
            render_out = self.renderer(**batch, render_rgb=False)
        else:
            render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        
        out = self(batch)
        
        if self.threefuse:
            with torch.no_grad():
                points = self.cond_pc
                device = self.device
                
                raster_settings = PointsRasterizationSettings(
                        image_size= 800,
                        radius = 0.005,
                        points_per_pixel = 10
                    )
                
                cam_radius = batch["camera_distances"]
                
                depth_map = render_depth_from_cloud(points, batch, raster_settings, cam_radius, device, calibration_value=0)

                # output_dir = "outputs/depth_outputs"
                # os.makedirs(output_dir, exist_ok=True)

                # # 2. 生成一个唯一的文件名，例如 "depth_map_0000.png", "depth_map_0001.png"
                # filename = os.path.join(output_dir, f"depth_map_{self.true_global_step:04d}.png")
    
                # # 3. 使用 torchvision.utils.save_image 来保存张量
                # #    这个函数非常方便，它会自动处理从 [0,1] 范围的浮点数到 [0,255] 整数图片的转换
                # torchvision.utils.save_image(depth_map, filename)

            if self.cfg.stage == "geometry":
                guidance_inp = out["comp_normal"]
                guidance_out = self.guidance(
                    guidance_inp, self.prompt_utils, **batch, depth_map=depth_map, rgb_as_latents=False
                )
            else:
                guidance_inp = out["comp_rgb"]
                guidance_out = self.guidance(
                    guidance_inp, self.prompt_utils, **batch, depth_map=depth_map, rgb_as_latents=False
                )
            
        else:
            if self.cfg.stage == "geometry":
                guidance_inp = out["comp_normal"]
                guidance_out = self.guidance(
                    guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False
                )
            else:
                guidance_inp = out["comp_rgb"]
                guidance_out = self.guidance(
                    guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False
                )
                
        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.cfg.stage == "coarse":
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        elif self.cfg.stage == "geometry":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
        elif self.cfg.stage == "texture":
            pass
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        # import pdb; pdb.set_trace()

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        
        if self.threefuse:
            with torch.no_grad():
                points = self.cond_pc
                device = self.device
                
                raster_settings = PointsRasterizationSettings(
                        image_size= 800,
                        radius = 0.02,
                        points_per_pixel = 10
                    )
                
                cam_radius = batch["camera_distances"]
                
                depth_map = render_depth_from_cloud(points, batch, raster_settings, cam_radius, device, calibration_value=0)
                # output_dir = "outputs/depth_outputs"
                # os.makedirs(output_dir, exist_ok=True)

                # # 2. 生成一个唯一的文件名，例如 "depth_map_0000.png", "depth_map_0001.png"
                # filename = os.path.join(output_dir, f"depth_map_{self.true_global_step:04d}.png")
    
                # # 3. 使用 torchvision.utils.save_image 来保存张量
                # #    这个函数非常方便，它会自动处理从 [0,1] 范围的浮点数到 [0,255] 整数图片的转换
                # torchvision.utils.save_image(depth_map, filename)
                
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": "jet", "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "grayscale",
                    "img": out["depth"][0, :, :, 0],
                    "kwargs": {"cmap": "jet", "data_range": None},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale",
                        "img": depth_map[0, :, :],
                        "kwargs": {"cmap": "jet", "data_range": None},
                    }
                ]
                if self.threefuse
                else []
            )
            ,
            name="validation_step",
            step=self.true_global_step,
        )

        if self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance.sample(
                            self.prompt_utils, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance.sample_lora(self.prompt_utils, **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        if self.threefuse:
            with torch.no_grad():
                points = self.cond_pc
                device = self.device
                
                raster_settings = PointsRasterizationSettings(
                        image_size= 800,
                        radius = 0.02,
                        points_per_pixel = 10
                    )
                
                cam_radius = batch["camera_distances"]
                
                depth_map = render_depth_from_cloud(points, batch, raster_settings, cam_radius, device, calibration_value=0)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": "jet", "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "grayscale",
                    "img": out["depth"][0, :, :, 0],
                    "kwargs": {"cmap": "jet", "data_range": None},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale",
                        "img": depth_map[0, :, :],
                        "kwargs": {"cmap": "jet", "data_range": None},
                    }
                ]
                if self.threefuse
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        
        
