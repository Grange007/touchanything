import os
import random
import shutil
from dataclasses import dataclass, field
import math
import pdb
import torch
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef
import cv2
import numpy as np
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.model_components.losses import ScaleAndShiftInvariantLoss, monosdf_normal_loss
import matplotlib.pyplot as plt

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
    Reads a .ply file and converts it to a custom PointCloud format.
    """
    try:
        pcd = o3d.io.read_point_cloud(ply_filepath)
        if len(pcd.points) > 10000:
            pcd = pcd.random_down_sample(10000 / len(pcd.points))

        if not pcd.has_points():
            print(f"Warning: File {ply_filepath} could not be read or contains no points.")
            return None

        # rotation_matrix = np.array([
        #     [-1, 0, 0],
        #     [0, -1, 0], 
        #     [0, 0, 1]
        # ])
        # pcd.rotate(rotation_matrix, center=(0, 0, 0))
        
        coords_np = np.asarray(pcd.points).astype(np.float32)
        center = coords_np.mean(axis=0)
        coords_np -= center

        # max_dist = np.max(np.linalg.norm(coords_np, axis=1))
        # coords_np /= (max_dist * 1.7)
        
        channels_dict = {}
        if pcd.has_colors():
            colors_np = np.asarray(pcd.colors)
            channels_dict['R'] = colors_np[:, 0]
            channels_dict['G'] = colors_np[:, 1]
            channels_dict['B'] = colors_np[:, 2]
        else:
            num_points = coords_np.shape[0]
            channels_dict['R'] = np.zeros(num_points)
            channels_dict['G'] = np.zeros(num_points)
            channels_dict['B'] = np.zeros(num_points)
            
        return PointCloud(coords=coords_np, channels=channels_dict)
    except Exception as e:
        print(f"Error reading or processing file: {e}")
        return None

@threestudio.register("image-condition-controlnet-system")
class ImageConditionControlNetSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        stage: str = "coarse"
        ply_file: Optional[str] = None
        freq: dict = field(default_factory=dict)
        refinement: bool = False
        ambient_ratio_min: float = 0.5
        visualize_samples: bool = False
        threefuse: bool = True
        texture: bool = True
        do_init: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        self.threefuse = self.cfg.threefuse
        if self.threefuse is True and self.cfg.ply_file:
            # self.cond_pc = point_e(device="cuda", exp_dir=self.image_dir)
            ply_file = self.cfg.ply_file
            self.cond_pc = read_ply_to_custom_format(ply_file)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # if self.cfg.stage == "geometry":
            # render_out = self.renderer(**batch, render_rgb=False)
        # else:
        render_out = self.renderer(**batch)
        return {**render_out}

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if not self.cfg.texture or self.cfg.do_init:
            self.geometry.initialize_shape()
        self.pearson = PearsonCorrCoef().to(self.device)
    
    def configure_optimizers(self):
        optim_cfg = self.cfg.optimizer 
        param_groups = []
        
        learning_rates = {
            "geometry.encoding": optim_cfg.params.geometry["encoding"].lr,
            "geometry.sdf_network": optim_cfg.params.geometry["sdf_network"].lr,
            "geometry.feature_network": optim_cfg.params.geometry["feature_network"].lr,
            "background": optim_cfg.params.background.lr,
            "renderer": optim_cfg.params.renderer.lr
        }
        default_params = []
        default_lr = 0.001
        grouped_names = set()

        for group_name, lr in learning_rates.items():
            group_params = []
            for name, param in self.named_parameters():
                if name.startswith(group_name):
                    group_params.append(param)
                    grouped_names.add(name)
            if group_params:
                param_groups.append({"params": group_params, "lr": lr})

        for name, param in self.named_parameters():
            if name not in grouped_names:
                default_params.append(param)
        
        if default_params:
            param_groups.append({"params": default_params, "lr": default_lr})
            threestudio.info(f"Added a default parameter group with lr={default_lr}")

        optimizer = torch.optim.Adam(
            param_groups,
            betas=tuple(optim_cfg.args.betas),
            eps=optim_cfg.args.eps
        )

        def lr_lambda(current_step):
            return 1.0 # Simple constant learning rate

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, "interval": "step", "frequency": 1,
            },
        }

    def training_substep(self, batch, batch_idx, guidance: str):
        loss_prefix = f"loss_{guidance}_"
        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        if guidance == "ref":
            ambient_ratio = 1.0
            shading = "diffuse"
            batch["shading"] = shading
        elif guidance == "guidance":
            batch = batch["random_camera"]
            ambient_ratio = (
                self.cfg.ambient_ratio_min
                + (1 - self.cfg.ambient_ratio_min) * random.random()
            )

        batch["bg_color"] = None
        batch["ambient_ratio"] = ambient_ratio
        out = self(batch)
        guidance_eval = (
            guidance == "guidance"
            and self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )
        if guidance == "ref":
            gt_rgb = batch["rgb"]
            gt_mask = batch["mask"]
            gt_mask_channel = gt_mask.repeat(1, 1, 1, 3)
            gt_rgb = gt_rgb * gt_mask_channel.float() + out["comp_rgb_bg"] * (1 - gt_mask_channel.float())
            if self.C(self.cfg.loss.lambda_rgb) >= 0:
                set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"]))
            if self.C(self.cfg.loss.lambda_mask) >= 0:
                set_loss("mask", F.mse_loss(gt_mask.float(), out["opacity"]))

            # depth_loss = ScaleAndShiftInvariantLoss(scales=1, alpha=0.5)
            # depth loss
            if self.C(self.cfg.loss.lambda_depth) >= 0:
                # print(f"gt_mask shape: {gt_mask.shape}, ref_depth shape: {batch['ref_depth'].shape}")
                # print(f"out[depth] shape: {out['depth'].shape}")
                # gt_depth = batch["ref_depth"]  # [B, H, W]
                gt_mask_depth = gt_mask
                gt_depth = batch["ref_depth"] * gt_mask_depth  # [B, H, W]
                pred_depth = out["depth"] * gt_mask_depth  # [B, H, W]
                set_loss(
                    "depth", F.l1_loss(gt_depth, pred_depth)
                )
            
            # if self.C(self.cfg.loss.lambda_freespace_loss) > 0:
                
            # SDF loss
            z_vals = out["t_points"]
            pred_sdf = out["sdf"]
            depth_gt = batch["ref_depth"]
            ray_indices = out["ray_indices"].long()  # [
            depth_truncation = self.C(
                self.cfg.loss.depth_truncation
            )
            depth_gt_expanded = depth_gt.view(-1)[ray_indices].unsqueeze(-1)
            valid_gt_mask = (gt_mask > 0.0).view(-1)[ray_indices].unsqueeze(-1)  # [B, 1]
            front_mask = valid_gt_mask & (z_vals < (depth_gt_expanded - depth_truncation))
            back_mask = valid_gt_mask & (z_vals > (depth_gt_expanded + depth_truncation))
            if self.C(self.cfg.loss.lambda_freespace) > 0:
                free_space_loss = torch.mean((F.relu(depth_truncation - pred_sdf) * front_mask) ** 2)
                set_loss("freespace", free_space_loss)

            if self.C(self.cfg.loss.lambda_sdf) > 0:
                sdf_mask = valid_gt_mask & (~front_mask) & (~back_mask)
                sdf_loss = torch.mean(((z_vals + pred_sdf) - depth_gt_expanded) ** 2 * sdf_mask)
                set_loss("sdf", sdf_loss)

            # relative depth loss
            if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask]  # [B,]
                valid_pred_depth = out["depth"][gt_mask]  # [B,]
                set_loss(
                    "depth_rel", 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                )

            # normal loss
            normals_loss = monosdf_normal_loss
            if self.C(self.cfg.loss.lambda_normal) > 0:
                valid_gt_normal = batch["ref_normal"][gt_mask.squeeze(-1) > 0]  
                valid_pred_normal = out["comp_normal"][gt_mask.squeeze(-1) > 0]  
                # valid_pred_normal = (
                #     2 * out["comp_normal"][gt_mask.squeeze(-1)] - 1
                # )  # [B, 3]
                set_loss(
                    "normal",
                    normals_loss(valid_pred_normal, valid_gt_normal),
                )
        elif guidance == "guidance":
            # Render depth map from point cloud to use as ControlNet condition
            if self.threefuse:
                with torch.no_grad():
                    points = self.cond_pc
                    device = self.device
                    raster_settings = PointsRasterizationSettings(
                        image_size=800, radius=0.005, points_per_pixel=10
                    )
                    cam_radius = batch["camera_distances"]
                    depth_map = render_depth_from_cloud(points, batch, raster_settings, cam_radius, device, calibration_value=0)
                    depth_map = depth_map.unsqueeze(0)
                    print(f"Depth map shape: {depth_map.shape}, min: {depth_map.min()}, max: {depth_map.max()}")
            
            if self.cfg.stage == "geometry":
                guidance_inp = out["comp_normal"]
            elif self.cfg.stage == "coarse":
                guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp, self.prompt_utils, **batch, depth_map=depth_map, guidance_eval=guidance_eval, rgb_as_latents=False
            )
            
            # Call the StableDiffusionControlNetGuidance
            # guidance_out = self.guidance(
            #     rgb=out["comp_rgb"],
            #     prompt_utils=self.prompt_utils,
            #     depth_map=depth_map,
            #     guidance_eval=guidance_eval,
            #     **batch,
            # )
            print(f"Guidance out keys: {list(guidance_out.keys())}")

            # Apply guidance loss
            for name, value in guidance_out.items():
                if name.startswith("loss_"):
                    set_loss(name.split("_")[1], value)

        if (hasattr(self.cfg.loss, "lambda_eikonal") and self.C(self.cfg.loss.lambda_eikonal) > 0):
            loss_eikonal = ((torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2).mean()
            set_loss("eikonal", loss_eikonal)
            
        if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            # print(f"normal min: {normal.min()}, max: {normal.max()}")
            # print(f"normal shape: {normal.shape}")
            set_loss(
                "normal_smooth",
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean()
            )

        if self.C(self.cfg.loss.lambda_3d_normal_smooth) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for normal smooth loss, no normal is found in the output."
                )
            if "normal_perturb" not in out:
                raise ValueError(
                    "normal_perturb is required for normal smooth loss, no normal_perturb is found in the output."
                )
            normals = out["normal"]
            normals_perturb = out["normal_perturb"]
            set_loss("3d_normal_smooth", (normals - normals_perturb).abs().mean())

        if not self.cfg.refinement:
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                set_loss(
                    "orient",
                    (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum()
                    / (out["opacity"] > 0).sum(),
                )

            # if guidance != "ref" and self.C(self.cfg.loss.lambda_sparsity) > 0:
            if self.C(self.cfg.loss.lambda_sparsity) > 0:
                set_loss("sparsity", (out["opacity"] ** 2 + 0.01).sqrt().mean())

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                set_loss(
                    "opaque", binary_cross_entropy(opacity_clamped, opacity_clamped)
                )
        else:
            if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                set_loss("normal_consistency", out["mesh"].normal_consistency())
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                set_loss("laplacian_smoothness", out["mesh"].laplacian())

        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted
                
        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        total_loss = 0.0
        
        # Guidance step
        # if self.true_global_step > self.cfg.freq.ref_only_steps:
        #     out_guidance = self.training_substep(batch, batch_idx, guidance="guidance")
        #     total_loss += out_guidance["loss"]

        # Reference image supervision step
        out_ref = self.training_substep(batch, batch_idx, guidance="guidance")
        total_loss += out_ref["loss"]

        self.log("train/loss", total_loss, prog_bar=True)
        return {"loss": total_loss}

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
            # + (
            #     [
            #         {
            #             "type": "grayscale",
            #             "img": batch["ref_depth"][0].squeeze(-1) * batch["mask"][0].squeeze(-1),
            #             "kwargs": {"cmap": "jet", "data_range": None},
            #         }
            #     ]
            #     if "ref_depth" in batch
            #     else []
            # )
            ,
            name="validation_step",
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
        
        
