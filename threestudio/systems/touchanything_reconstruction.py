import bisect
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

@threestudio.register("touchanything-reconstruction-system")
class TouchAnythingReconstructionSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        freq: dict = field(default_factory=dict)
        refinement: bool = False
        ambient_ratio_min: float = 0.5
        visualize_samples: bool = False
        latent_steps: int = 1000
        nd_latent_steps: int = 1000
        texture: bool = True
        do_init: bool = False
        density_aware_guidance: bool = False
        density_power: float = 3.0
        min_guidance_scale: float = 10.0
        max_guidance_scale: float = 60.0
        min_lambda_sds: float = 0.1
        max_lambda_sds: float = 2.0
        lambda_sds_power: float = 3.0
        anneal_normal_stone: Optional[Any] = None
        # Runtime SDF subtraction schedule (applied to the whole scene SDF).
        # See configs for expected structure.
        sdf_subtract: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.has_nd_guidance = (self.cfg.nd_guidance_type != "none") and hasattr(
            self.cfg.loss, "lambda_nd"
        )  # and (self.cfg.loss.lambda_nd > 0)
        self.has_rgb_sd_guidanece = (self.cfg.guidance_type != "none") and hasattr(
            self.cfg.loss, "lambda_rgb_sd"
        )  # and (self.cfg.loss.lambda_rgb_sd > 0)
        threestudio.info(
            f"================has_nd_guidance:{self.has_nd_guidance}, has_rgb_sd_guidanece:{self.has_rgb_sd_guidanece}================="
        )

        if self.has_rgb_sd_guidanece:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            # self.guidance.requires_grad_(False)
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.prompt_utils = self.prompt_processor()

        if self.has_nd_guidance:
            self.nd_guidance = threestudio.find(self.cfg.nd_guidance_type)(
                self.cfg.nd_guidance
            )
            self.nd_guidance.requires_grad_(False)
            self.nd_prompt_processor = threestudio.find(
                self.cfg.nd_prompt_processor_type
            )(self.cfg.nd_prompt_processor)
            self.nd_prompt_utils = self.nd_prompt_processor()

    def on_load_checkpoint(self, checkpoint):
        for k in list(checkpoint["state_dict"].keys()):
            if k.startswith("guidance."):
                return
            if k.startswith("nd_guidance."):
                return
        if self.has_rgb_sd_guidanece:
            if hasattr(self.guidance, "state_dict"):
                guidance_state_dict = {
                    "guidance." + k: v for (k, v) in self.guidance.state_dict().items()
                }
                checkpoint["state_dict"] = {
                    **checkpoint["state_dict"],
                    **guidance_state_dict,
                }

        if self.has_nd_guidance:
            guidance_nd_state_dict = {
                "nd_guidance." + k: v
                for (k, v) in self.nd_guidance.state_dict().items()
            }
            checkpoint["state_dict"] = {
                **checkpoint["state_dict"],
                **guidance_nd_state_dict,
            }

        return

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint["state_dict"].keys()):
            if k.startswith("guidance."):
                checkpoint["state_dict"].pop(k)
            if k.startswith("nd_guidance."):
                checkpoint["state_dict"].pop(k)
        return

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(
            **batch, render_rgb=self.cfg.texture or self.has_rgb_sd_guidanece
        )

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # # only used in training
        # self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
        #     self.cfg.prompt_processor
        # )
        # self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # if not self.cfg.texture or self.cfg.do_init:
            # initialize SDF
            # FIXME: what if using other geometry types?
            # self.geometry.initialize_shape()
            
        all_images = self.trainer.datamodule.train_dataloader().dataset.get_all_images()
        # self.save_image_grid(
        #     "all_training_images.png",
        #     [
        #         {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
        #         for image in all_images
        #     ],
        #     name="on_fit_start",
        #     step=self.true_global_step,
        # )

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

        total_steps = self.trainer.max_steps
        warmup_steps = 500

        def lr_lambda(current_step):
            # if current_step < warmup_steps:
            #     return float(current_step) / float(max(1, warmup_steps))
            # progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            # return 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        threestudio.info("================== Using custom configure_optimizers with LR scheduler! ==================")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _compute_sdf_subtract(self, global_step: int) -> float:
        cfg = getattr(self.cfg, "sdf_subtract", None) or {}

        total_subtract = 0.0

        step_cfg = cfg.get("step", None) or {}
        step_enabled = bool(step_cfg.get("enabled", step_cfg.get("enable", False)))
        if step_enabled:
            trigger_step = int(step_cfg.get("step", 0))
            step_value = float(step_cfg.get("value", 0.0))
            if global_step >= trigger_step:
                total_subtract += step_value

        ramp_cfg = cfg.get("ramp", None) or {}
        ramp_enabled = bool(ramp_cfg.get("enabled", ramp_cfg.get("enable", False)))
        if ramp_enabled:
            start_step = int(ramp_cfg.get("start_step", 0))
            end_step = int(ramp_cfg.get("end_step", start_step))
            start_value = float(ramp_cfg.get("start_value", 0.0))
            end_value = float(ramp_cfg.get("end_value", 0.0))

            if global_step < start_step:
                ramp_value = 0.0
            elif end_step <= start_step:
                ramp_value = end_value
            elif global_step >= end_step:
                ramp_value = end_value
            else:
                t = (global_step - start_step) / float(end_step - start_step)
                ramp_value = start_value + t * (end_value - start_value)

            total_subtract += ramp_value

        return float(total_subtract)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        sdf_subtract = self._compute_sdf_subtract(global_step)

        if hasattr(self, "geometry") and hasattr(self.geometry, "set_sdf_subtract"):
            self.geometry.set_sdf_subtract(sdf_subtract)
        elif hasattr(self, "geometry"):
            # Best-effort fallback for other geometries.
            self.geometry.sdf_subtract = float(sdf_subtract)

        if getattr(self, "_last_sdf_subtract", None) != sdf_subtract:
            threestudio.info(
                f"[sdf_subtract] step={global_step} value={sdf_subtract:.6f}"
            )
            self._last_sdf_subtract = sdf_subtract

        if (not on_load_weights) and self.training:
            self.log("train/sdf_subtract", sdf_subtract)

        
    def collect_inputs(self, out, collect_inputs):
        inputs = [out[key] for key in collect_inputs]
        return torch.cat(inputs, dim=-1)

    def training_substep(self, batch, batch_idx, guidance: str):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "guidance"
        """
        # self.geometry.initialize_shape()
        # if guidance == "ref":
        #     # bg_color = torch.rand_like(batch['rays_o'])
        #     ambient_ratio = 1.0
        #     shading = "diffuse"
        #     batch["shading"] = shading
        # elif guidance == "guidance":
        #     ambient_ratio = (
        #         self.cfg.ambient_ratio_min
        #         + (1 - self.cfg.ambient_ratio_min) * random.random()
        #     )
        if guidance == "guidance":
            batch = batch["random_camera"]
            self.has_nd_guidance = (
                (self.cfg.nd_guidance_type != "none")
                and hasattr(self.cfg.loss, "lambda_nd")
                and (self.C(self.cfg.loss.lambda_nd) > 0)
            )
            self.has_rgb_sd_guidance = (
                (self.cfg.guidance_type != "none")
                and hasattr(self.cfg.loss, "lambda_rgb_sd")
                and (self.C(self.cfg.loss.lambda_rgb_sd) > 0)
            )

        batch["bg_color"] = None
        # batch["ambient_ratio"] = ambient_ratio

        density_weights = batch.get("density_weights", None)
        if self.cfg.density_aware_guidance and density_weights is not None:
            # density_weights越大，guidance scale越小
            # 使用指数函数来调整，density_power控制调整的强度
            guidance_scale_factor = (1.0 - density_weights).pow(self.cfg.density_power)
            # 将因子缩放到[min_guidance_scale, max_guidance_scale]范围
            adjusted_guidance_scale = (
                self.cfg.min_guidance_scale + 
                guidance_scale_factor * (self.cfg.max_guidance_scale - self.cfg.min_guidance_scale)
            )
            # 记录调整后的guidance scale
            self.log("train/adjusted_guidance_scale", adjusted_guidance_scale.mean())
            self.log("train/density_weights", density_weights.mean())
            
            lambda_sds_factor = (1.0 - density_weights).pow(self.cfg.lambda_sds_power)
            adjusted_lambda_sds = (
                self.cfg.min_lambda_sds + 
                lambda_sds_factor * (self.cfg.max_lambda_sds - self.cfg.min_lambda_sds)
            )
            self.log("train/adjusted_lambda_sds", adjusted_lambda_sds.mean())
        else:
            adjusted_guidance_scale = None
            adjusted_lambda_sds = None

        out = self(batch)
        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        guidance_eval = (
            guidance == "guidance"
            and self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )

        if guidance == "ref":
            gt_rgb = batch["rgb"]
            gt_mask = batch["mask"]
            # gt_mask = gt_mask.permute(0, 2, 1)  # [B, H, W, C] -> [B, C, H, W]
            # set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"]))
            # mask loss
            # set_loss("mask", F.mse_loss(gt_mask.float(), out["opacity"]))

            # depth loss - Modified for 1/|Ω| normalization
            if self.C(self.cfg.loss.lambda_depth) >= 0:
                mask_bool = gt_mask > 0.5
                if mask_bool.sum() > 0:
                    gt_depth = batch["ref_depth"][mask_bool]
                    pred_depth = out["depth"][mask_bool]
                    set_loss("depth", F.l1_loss(gt_depth, pred_depth))
                else:
                    set_loss("depth", torch.tensor(0.0, device=out["depth"].device, requires_grad=True))
            
            if self.C(self.cfg.loss.lambda_freespace) > 0 or self.C(self.cfg.loss.lambda_sdf) > 0:
                # SDF loss - Modified for 1/|Ω| normalization
                t_points = out["t_points"]  # ray parameter t values
                t_dirs = out["t_dirs"]  # ray directions at sample points
                pred_sdf = out["sdf"]
                depth_gt = batch["ref_depth"]
                ray_indices = out["ray_indices"].long()
                depth_truncation = self.C(self.cfg.loss.depth_truncation)
                
                # 计算ray direction的归一化长度
                directions_norm = torch.norm(t_dirs, dim=-1, keepdim=True)
                # 将ray parameter转换为真实深度: z_vals = t / ||d||
                z_vals = t_points / directions_norm
                
                depth_gt_expanded = depth_gt.view(-1)[ray_indices].unsqueeze(-1)
                valid_gt_mask = (gt_mask > 0.0).view(-1)[ray_indices].unsqueeze(-1)
                
                front_mask = valid_gt_mask & (z_vals < (depth_gt_expanded - depth_truncation))
                back_mask = valid_gt_mask & (z_vals > (depth_gt_expanded + depth_truncation))
                sdf_mask = valid_gt_mask & (~front_mask) & (~back_mask)
                
                # 添加自适应权重机制，参考sdfstudio实现
                num_fs_samples = front_mask.sum().float()
                num_sdf_samples = sdf_mask.sum().float()
                num_samples = num_fs_samples + num_sdf_samples + 1e-6
                # fs_weight = 1.0 - num_fs_samples / num_samples
                # sdf_weight = 1.0 - num_sdf_samples / num_samples
                fs_weight = 1.0
                sdf_weight = 1.0
                
                if self.C(self.cfg.loss.lambda_freespace) > 0:
                    if front_mask.sum() > 0:
                        free_space_loss = torch.mean((F.relu(depth_truncation - pred_sdf)[front_mask]) ** 2) * fs_weight
                        set_loss("freespace", free_space_loss)
                    else:
                        set_loss("freespace", torch.tensor(0.0, device=pred_sdf.device, requires_grad=True))

                if self.C(self.cfg.loss.lambda_sdf) > 0:
                    if sdf_mask.sum() > 0:
                        sdf_loss = torch.mean(((z_vals + pred_sdf) - depth_gt_expanded)[sdf_mask] ** 2) * sdf_weight
                        set_loss("sdf", sdf_loss)
                    else:
                        set_loss("sdf", torch.tensor(0.0, device=pred_sdf.device, requires_grad=True))

            # relative depth loss
            if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask]  # [B,]
                valid_pred_depth = out["depth"][gt_mask]  # [B,]
                set_loss(
                    "depth_rel", 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                )

            # normal loss - Modified for 1/|Ω| normalization
            if self.C(self.cfg.loss.lambda_normal) > 0:
                # Get the spatial mask [B, H, W]
                mask_bool_normal = (gt_mask > 0.5).squeeze(-1) 
                
                if mask_bool_normal.sum() > 0:
                    # Index to get valid [M, 3] tensors directly
                    valid_gt_normal = batch["origin_normal"][mask_bool_normal]
                    valid_pred_normal = out["comp_normal_cam"][mask_bool_normal] * 2.0 - 1.0
                    set_loss("normal", F.l1_loss(valid_pred_normal, valid_gt_normal))
                else:
                    set_loss("normal", torch.tensor(0.0, device=out["comp_normal_cam"].device, requires_grad=True))

        elif guidance == "guidance":
            
            if not self.cfg.texture:  # geometry training
                if self.has_nd_guidance:
                    if self.true_global_step < self.cfg.nd_latent_steps:
                        nd_guidance_inp = self.collect_inputs(
                            out, self.cfg.nd_guidance.collect_inputs_lat
                        )
                        nd_guidance_inp = nd_guidance_inp * 2.0 - 1.0

                        nd_guidance_out = self.nd_guidance(
                            nd_guidance_inp,
                            self.nd_prompt_utils,
                            **batch,
                            rgb_as_latents=True,
                            guidance_eval=False,
                            adjusted_guidance_scale=adjusted_guidance_scale,
                        )
                    else:
                        nd_guidance_inp = self.collect_inputs(
                            out, self.cfg.nd_guidance.collect_inputs
                        )

                        nd_guidance_out = self.nd_guidance(
                            nd_guidance_inp,
                            self.nd_prompt_utils,
                            **batch,
                            rgb_as_latents=False,
                            guidance_eval=False,
                            adjusted_guidance_scale=adjusted_guidance_scale,
                        )

                if self.has_rgb_sd_guidanece:
                    timestep = (
                        nd_guidance_out["timestep"]
                        if self.cfg.guidance.share_t and self.has_nd_guidance
                        else None
                    )
                    if self.true_global_step < self.cfg.latent_steps:
                        guidance_inp = self.collect_inputs(
                            out, self.cfg.guidance.collect_inputs_lat
                        )
                        guidance_inp = guidance_inp * 2.0 - 1.0

                        guidance_out = self.guidance(
                            guidance_inp,
                            self.prompt_utils,
                            **batch,
                            rgb_as_latents=True,
                            guidance_eval=guidance_eval,
                            timestep=timestep,
                            adjusted_guidance_scale=adjusted_guidance_scale,
                        )
                    else:
                        collect_inps = self.cfg.guidance.collect_inputs
                        if self.cfg.switch_ginp:
                            collect_inps = [
                                collect_inps[
                                    self.true_global_step % self.cfg.switch_freq == 0
                                ]
                            ]

                        guidance_inp = self.collect_inputs(out, collect_inps)

                        guidance_out = self.guidance(
                            guidance_inp,
                            self.prompt_utils,
                            **batch,
                            rgb_as_latents=False,
                            guidance_eval=guidance_eval,
                            timestep=timestep,
                            adjusted_guidance_scale=adjusted_guidance_scale,
                        )

                if "mesh" in out:
                    if (
                        hasattr(self.cfg.loss, "lambda_normal_consistency")
                        and self.C(self.cfg.loss.lambda_normal_consistency) > 0
                    ):
                        loss_normal_consistency = out["mesh"].normal_consistency()
                        
                        # anneal refine strategy
                        if self.cfg.anneal_normal_stone is None:
                            anneal_weights = 1.0
                        else:
                            anneal_idx = bisect.bisect_left(
                                self.cfg.anneal_normal_stone, self.true_global_step
                            )
                            anneal_weights = (10**anneal_idx)
                        
                        set_loss(
                            "normal_consistency", loss_normal_consistency * anneal_weights
                        )
                    if (
                        hasattr(self.cfg.loss, "lambda_laplacian_smoothness")
                        and self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0
                    ):
                        loss_laplacian_smoothness = out["mesh"].laplacian()
                        set_loss(
                            "laplacian_smoothness", loss_laplacian_smoothness
                        )
            else:  # texture training
                if self.has_nd_guidance:
                    nd_guidance_inp = self.collect_inputs(
                        out, self.cfg.nd_guidance.collect_inputs
                    )
                    nd_guidance_out = self.nd_guidance(
                        nd_guidance_inp, self.nd_prompt_utils, **batch, rgb_as_latents=False,
                        adjusted_guidance_scale=adjusted_guidance_scale,
                    )

                if self.has_rgb_sd_guidanece:
                    timestep = (
                        nd_guidance_out["timestep"]
                        if self.cfg.guidance.share_t and self.has_nd_guidance
                        else None
                    )
                    guidance_inp = self.collect_inputs(
                        out, self.cfg.guidance.collect_inputs
                    )
                    guidance_out = self.guidance(
                        guidance_inp,
                        self.prompt_utils,
                        **batch,
                        rgb_as_latents=False,
                        timestep=timestep,
                        current_step_ratio=self.true_global_step / self.trainer.max_steps,
                        adjusted_guidance_scale=adjusted_guidance_scale,
                    )

        if (
            hasattr(self.cfg.loss, "lambda_eikonal")
            and self.C(self.cfg.loss.lambda_eikonal) > 0
        ):
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
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
                loss_normal_consistency = out["mesh"].normal_consistency()
                
                # anneal refine strategy
                if self.cfg.anneal_normal_stone is None:
                    anneal_weights = 1.0
                else:
                    anneal_idx = bisect.bisect_left(
                        self.cfg.anneal_normal_stone, self.true_global_step
                    )
                    anneal_weights = (10**anneal_idx)
                
                set_loss("normal_consistency", loss_normal_consistency * anneal_weights)
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
        
        if guidance == "guidance":
            # pdb.set_trace()
            loss_rgb_sd = 0
            if self.has_rgb_sd_guidanece:
                for name, value in guidance_out.items():
                    if name != "timestep" and name != "eval":
                        self.log(f"train/rgb_{name}", value)
                    if name.startswith("loss_"):
                        if name == "loss_sds" and adjusted_lambda_sds is not None:
                            value = value * adjusted_lambda_sds.mean()
                        loss_weighted = value * self.C(
                            self.cfg.loss[name.replace("loss_", "lambda_rgb_")]
                        )
                        self.log(f"train/rgb_{name}_w", loss_weighted)
                        loss_rgb_sd += loss_weighted
                loss += loss_rgb_sd  # * self.C(self.cfg.loss.lambda_rgb_sd)

            loss_nd = 0
            if self.has_nd_guidance:
                for name, value in nd_guidance_out.items():
                    if name != "timestep" and name != "eval":
                        self.log(f"train/nd_{name}", value)
                    if name.startswith("loss_"):
                        if name == "loss_sds" and adjusted_lambda_sds is not None:
                            value = value * adjusted_lambda_sds.mean()
                        loss_nd += value * self.C(
                            self.cfg.loss[name.replace("loss_", "lambda_nd_")]
                        )
                nd_weight = (
                    self.C(self.cfg.loss.lambda_nd_w)
                    if hasattr(self.cfg.loss, "lambda_nd_w")
                    else 1.0
                )
                loss += loss_nd * nd_weight  # * self.C(self.cfg.loss.lambda_nd)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        if guidance_eval:
            if self.has_rgb_sd_guidanece:
                self.guidance_evaluation_save(
                    out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                    guidance_out["eval"],
                )

        return {"loss": loss, "out": out}


    def training_step(self, batch, batch_idx):
        total_loss = 0.0
        # print(batch.keys())
        # guidance
        if self.true_global_step > self.cfg.freq.ref_only_steps and (not hasattr(self.cfg.freq, 'ref_only_steps_post') or self.true_global_step < self.cfg.freq.ref_only_steps_post):            
            out = self.training_substep(batch, batch_idx, guidance="guidance")
            total_loss += out["loss"]

        # ref
        out = self.training_substep(batch, batch_idx, guidance="ref")
        total_loss += out["loss"]

        # out_out = self(batch)
        # # 添加depth可视化 - 每100步或每个epoch的第一个batch进行可视化
        # if True:
        # # if "ref_depth" in batch and "depth" in out:
        #     # 获取第一个batch的数据进行可视化
        #     print(f"batch mask shape: {batch['mask'].shape}, batch ref_depth shape: {batch['ref_depth'].shape}, out depth shape: {out_out['depth'].shape}")
        #     gt_mask = batch["mask"][0]
        #     gt_depth = batch["ref_depth"][0]
        #     pred_depth = out_out["depth"][0]  # [H, W, 1] or [H, W]
        #     gt_normal = batch["ref_normal"][0]
        #     pred_normal = out_out["comp_normal"][0]
            
            
        #     # 应用mask到depth数据
        #     if gt_mask is not None:
        #         gt_depth_masked = gt_depth * gt_mask
        #         pred_depth_masked = pred_depth * gt_mask
        #     else:
        #         gt_depth_masked = gt_depth
        #         pred_depth_masked = pred_depth
            
        #     # 创建可视化图像列表
        #     vis_images = [
        #         {
        #             "type": "rgb",
        #             "img": gt_normal,
        #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
        #         },
        #         {
        #             "type": "rgb",
        #             "img": pred_normal,
        #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
        #         },
        #         {
        #             "type": "grayscale",
        #             "img": gt_depth,
        #             "kwargs": {"cmap": "jet", "data_range": (0, 3)},
        #         },
        #         {
        #             "type": "grayscale", 
        #             "img": pred_depth,
        #             "kwargs": {"cmap": "jet", "data_range": (0, 3)},
        #         },
        #         {
        #             "type": "grayscale",
        #             "img": torch.abs(gt_depth - pred_depth_masked),
        #             "kwargs": {"cmap": "jet", "data_range": None},
        #         }
        #     ]
            
        #     # 如果有RGB图像，也加入可视化
        #     # if "rgb" in batch:
        #     #     gt_rgb = batch["rgb"][0]  # [H, W, 3]
        #     #     vis_images.insert(0, {
        #     #         "type": "rgb",
        #     #         "img": gt_rgb,
        #     #         "kwargs": {"data_format": "HWC"},
        #     #     })
            
        #     # if "comp_rgb" in out:
        #     #     pred_rgb = out["comp_rgb"][0]  # [H, W, 3]
        #     #     vis_images.insert(-3, {
        #     #         "type": "rgb", 
        #     #         "img": pred_rgb,
        #     #         "kwargs": {"data_format": "HWC"},
        #     #     })
            
        #     # 保存可视化图像
        #     # self.save_image_grid(
        #     #     f"it{self.true_global_step:06d}.png",
        #     #     vis_images,
        #     #     name="train_depth_visualization",
        #     #     step=self.true_global_step,
        #     # )

        self.log("train/loss", total_loss, prog_bar=True)
        # sch = self.lr_schedulers()
        # sch.step()

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        # 绘制三个normal通道分布的直方图对比
        # if "comp_normal" in out and "comp_normal_cam" in out and "ref_normal" in batch and "origin_normal" in batch:
        #     # 获取数据并转换为numpy数组
        #     comp_normal = out["comp_normal"][0].detach().cpu().numpy()  # [H, W, 3]
        #     comp_normal_cam = out["comp_normal_cam"][0].detach().cpu().numpy()  # [H, W, 3]
        #     ref_normal = batch["ref_normal"][0].detach().cpu().numpy() # [H, W, 3]
        #     ref_normal = (ref_normal + 1.0) / 2.0  # 将[-1, 1]范围转换为[0, 1]
        #     origin_normal = batch["origin_normal"][0].detach().cpu().numpy()  # [H, W, 3]
        #     origin_normal = (origin_normal + 1.0) / 2.0  # 将[-1, 1]范围转换为[0, 1]
        #     # 创建4x3的子图，每行对应一个normal类型，每列对应RGB三个通道
        #     fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        #     fig.suptitle(f'Normal Channels Distribution Comparison - Step {self.true_global_step}', fontsize=16)
            
        #     # 定义颜色和标签
        #     colors = ['red', 'green', 'blue']
        #     channel_names = ['R Channel', 'G Channel', 'B Channel']
        #     normal_types = ['comp_normal', 'comp_normal_cam', 'ref_normal', 'origin_normal']
        #     normal_data = [comp_normal, comp_normal_cam, ref_normal, origin_normal]

        #     # 为每个normal类型和每个通道绘制直方图
        #     for i, (normal_type, data) in enumerate(zip(normal_types, normal_data)):
        #         for j, (color, channel_name) in enumerate(zip(colors, channel_names)):
        #             ax = axes[i, j]
                    
        #             # 提取当前通道数据并展平
        #             channel_data = data[:, :, j].flatten()
                    
        #             # 绘制直方图
        #             ax.hist(channel_data, bins=50, color=color, alpha=0.7, density=True)
        #             ax.set_title(f'{normal_type} - {channel_name}')
        #             ax.set_xlabel('Value')
        #             ax.set_ylabel('Density')
        #             ax.grid(True, alpha=0.3)
                    
        #             # 添加统计信息
        #             mean_val = np.mean(channel_data)
        #             std_val = np.std(channel_data)
        #             ax.axvline(mean_val, color='black', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
        #             ax.text(0.02, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
        #                    transform=ax.transAxes, verticalalignment='top', fontsize=8,
        #                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        #     plt.tight_layout()
            
        #     # 保存直方图
        #     histogram_path = f"outputs/normal/normal_histograms_step_{self.true_global_step:06d}_batch_{batch_idx:04d}.png"
        #     os.makedirs(os.path.dirname(histogram_path), exist_ok=True)
        #     plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        #     plt.close()
            
        #     print(f"Normal channels histogram saved to: {histogram_path}")

        self.save_image_grid(
            f"it{self.true_global_step:06d}-{batch['index'][0]:04d}.png",
            # (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": out["comp_rgb"][0],
            #             "kwargs": {"data_format": "HWC"},
            #         },
            #     ]
            #     if "comp_rgb" in out
            #     else []
            # )
            # + (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": out["comp_normal_white_vis"][0],
            #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            #         }
            #     ]
            #     if "comp_normal_white_vis" in out
            #     else []
            # )
            # + 
            (
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
            # +
            # (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": out["comp_normal_cam"][0],
            #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            #         }
            #     ]
            #     if "comp_normal_cam" in out
            #     else []
            # )
            # +
            # (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": (batch["ref_normal"][0] + 1.0) / 2.0,
            #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            #         }
            #     ]
            #     if "ref_normal" in batch
            #     else []
            # )
            # +
            # (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": (batch["origin_normal"][0] + 1.0) / 2.0,
            #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            #         }
            #     ]
            #     if "origin_normal" in batch
            #     else []
            # )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0].squeeze(-1),
                        "kwargs": {"cmap": "jet", "data_range": None},
                    }
                ]
                if "depth" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0].squeeze(-1),
                        "kwargs": {"cmap": "jet", "data_range": None},
                    }
                ]
                if "opacity" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["disparity"][0].squeeze(-1),
                        "kwargs": {"cmap": "jet", "data_range": (0, 1)},
                    }
                ]
                if "disparity" in out
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
        # save_dir = f"outputs/data_analysis_test/step_test_{batch['index'][0]:04d}"
        # os.makedirs(save_dir, exist_ok=True)
        # pred_depth = out["depth"][0]  # [H, W, 1] or [H, W]
        # if pred_depth.dim() == 3:
        #     pred_depth = pred_depth.squeeze(-1)  # [H, W]
        # pred_normal_full = out["comp_normal"][0]  # [H, W, 3]
        # pred_normal_full_vis = (pred_normal_full + 1.0) / 2.0
        # np.save(f"{save_dir}/pred_depth.npy", pred_depth.detach().cpu().numpy())
        # np.save(f"{save_dir}/pred_normal_full.npy", pred_normal_full.detach().cpu().numpy())
        # plt.figure(figsize=(8, 6))
        # plt.imshow(pred_depth.detach().cpu().numpy(), cmap='jet')
        # plt.colorbar()
        # plt.title('Predicted Depth')
        # plt.savefig(f"{save_dir}/pred_depth.png", dpi=150, bbox_inches='tight')
        # plt.close()
        
        # plt.figure(figsize=(8, 6))
        # plt.imshow(pred_normal_full_vis.detach().cpu().numpy())
        # plt.title('Predicted Normal')
        # plt.axis('off')
        # plt.savefig(f"{save_dir}/pred_normal.png", dpi=150, bbox_inches='tight')
        # plt.close()
        
        self.save_image_grid(
            f"it{self.true_global_step:06d}-test/{batch['index'][0]:04d}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
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
            # + (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": out["comp_normal_cam_white_vis"][0],
            #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            #         }
            #     ]
            #     if "comp_normal_cam_white_vis" in out
            #     else []
            # )
            + 
            (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0].squeeze(-1),
                        "kwargs": {"cmap": "jet", "data_range": None},
                    }
                ]
                if "depth" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
                f"it{self.true_global_step:06d}-test",
                f"it{self.true_global_step:06d}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        
