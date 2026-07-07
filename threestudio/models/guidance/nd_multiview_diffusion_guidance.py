import cv2
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List
from tqdm import tqdm

import threestudio
from extern.nd_sd.model_zoo import build_model
from extern.nd_sd.ldm.camera_utils import (convert_opengl_to_blender,
                                           normalize_camera,)
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


@threestudio.register("nd-multiview-diffusion-guidance")
class MultiviewDiffusionGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        model_name: str = "nd-4view"
        ckpt_path: Optional[
            str
        ] = None  # path to local checkpoint (None for loading from url)
        guidance_scale: float = 50.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_view: int = 2
        image_size: int = 256
        recon_loss: bool = True
        recon_std_rescale: float = 0.5
        collect_inputs: Optional[list] = field(default_factory=lambda: ["comp_rgb"])
        collect_inputs_lat: Optional[list] = field(default_factory=lambda: ["comp_rgb"])
        camera_distance: float = 2.0
        rotate_z: bool = False
        weighting_strategy: str = "sds"
        cam_method: str = "abs_spec"
        generate_img: bool = False
        half_precision_weights: bool = True
        
        # Add for guidance evaluation
        max_items_eval: int = 4  # Maximum number of items to evaluate

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Diffusion ...")

        self.model, self.model_cfg = build_model(
            self.cfg.model_name, ckpt_path=self.cfg.ckpt_path, return_cfg=True
        )
        self.cond_method = (
            self.model_cfg.model.params.cond_method
            if hasattr(self.model_cfg.model.params, "cond_method")
            else "ori"
        )

        from extern.nd_sd.ldm.models.diffusion.ddim import DDIMSampler

        self.sampler = DDIMSampler(self.model)

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
                
        self.model.eval()
        self.model.to(self.device, dtype=self.weights_dtype)
        self.alphas_cumprod: Float[Tensor, "..."] = self.model.alphas_cumprod.to(self.device)

        threestudio.info(f"Loaded Multiview Diffusion!")
        self.count = 0

    def get_cond_input(self, input, cond_method: str, c: dict, image_size):
        if cond_method == "ori":
            pass
        elif cond_method == "cat_n":
            normal = input["normal"]
            normal_z = torch.nn.functional.interpolate(
                normal, size=(image_size // 8, image_size // 8), mode="nearest"
            )

            c["c_concat"] = normal_z.repeat(2, 1, 1, 1).detach()

        elif cond_method == "cat_d":
            depth = input["depth"]
            depth_z = torch.nn.functional.interpolate(
                depth, size=(image_size // 8, image_size // 8), mode="nearest"
            )
            c["c_concat"] = depth_z.repeat(2, 1, 1, 1).detach()

        elif cond_method == "cat_nd":
            normal = input["normal"]
            depth = input["depth"]

            normal_z = torch.nn.functional.interpolate(
                normal, size=(image_size // 8, image_size // 8), mode="nearest"
            )

            depth_z = torch.nn.functional.interpolate(
                depth, size=(image_size // 8, image_size // 8), mode="nearest"
            )

            nd_z = torch.cat([normal_z, depth_z], dim=1)
            c["c_concat"] = nd_z.repeat(2, 1, 1, 1).detach()

        else:
            raise NotImplementedError
        return c

    def generate_img(
        self, c, image_size, batch_size, other_inp=None, scale=10, as_latents=False
    ):
        c_ = {}
        uc_ = {}
        for k, v in c.items():
            print(k, type(v))
            if isinstance(v, torch.Tensor):
                c_[k] = v[:batch_size]
                uc_[k] = v[batch_size:]
            else:
                c_[k] = v
                uc_[k] = v

        self.model.device = self.device
        shape = [4, image_size // 8, image_size // 8]
        step = 50
        ddim_eta = 0.0

        samples_ddim, _ = self.sampler.sample(
            S=step,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_,
            eta=ddim_eta,
            x_T=None,
        )

        if not as_latents:
            x_sample = self.model.decode_first_stage(samples_ddim)
        else:
            x_sample = F.interpolate(
                samples_ddim, (image_size, image_size), mode="bilinear"
            )

        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).cpu().numpy()

        if "depth" in other_inp:
            depth = other_inp["depth"][:, 0].cpu().numpy()
            depth = (depth + 1.0) / 2 * 255
            # os.makedirs("debug", exist_ok=True)
            gen_img = np.concatenate(list(x_sample.astype(np.uint8)), axis=1)[
                :, :, (2, 1, 0)
            ]
            depth = np.concatenate(list(depth.astype(np.uint8)), axis=1)[
                :, :, np.newaxis
            ]
            depth = cv2.resize(
                np.tile(depth, (1, 1, 3)),
                dsize=(gen_img.shape[1], gen_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            vis_img = np.concatenate([depth, gen_img], axis=0)
            # cv2.imwrite(f"debug/debug_sample.jpg", vis_img)
            return vis_img
        else:
            # os.makedirs("debug", exist_ok=True)
            gen_img = np.concatenate(list(x_sample.astype(np.uint8)), axis=1)[
                :, :, (2, 1, 0)
            ]
            # cv2.imwrite(f"debug/debug_sample.jpg", gen_img)
            return gen_img
        self.count += 1

    def get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
        distances=1,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.view(-1, 4, 4)

            if self.cfg.rotate_z:
                from scipy.spatial.transform import Rotation as R

                r = R.from_euler("z", -90, degrees=True).as_matrix()
                rotate_mat = torch.eye(4, dtype=camera.dtype, device=camera.device)
                rotate_mat[:3, :3] = torch.from_numpy(r)
                rotate_mat = rotate_mat.unsqueeze(0).repeat(camera.shape[0], 1, 1)
                camera = torch.matmul(rotate_mat, camera)

            if isinstance(distances, torch.Tensor):
                distances = distances.unsqueeze(1)

            camera[..., :3, 3] = camera[..., :3, 3] * distances
            camera = camera.flatten(start_dim=1)

        else:
            raise NotImplementedError(
                f"Unknown camera_condition_type={self.cfg.camera_condition_type}"
            )
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs.to(self.weights_dtype))
        )
        return latents.to(input_dtype)  # [B, 4, 32, 32] Latent space image

    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 32 32"],
    ) -> Float[Tensor, "B 3 256 256"]:
        input_dtype = latents.dtype
        latents = latents.to(self.weights_dtype)
        image = self.model.decode_first_stage(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def parse_input(self, input, cond_method):
        other_inp = {}
        if cond_method == "ori":
            raw_input = input

        elif cond_method == "cat_n":
            assert input.shape[1] == 6
            raw_input = input[:, :3]
            other_inp["normal"] = input[:, 3:] * 2 - 1.0

        elif cond_method == "cat_d":
            assert input.shape[1] == 4
            raw_input = input[:, :3]
            other_inp["depth"] = input[:, 3:] * 2 - 1.0

        elif cond_method == "cat_nd":
            assert input.shape[1] == 7
            raw_input = input[:, :3]
            other_inp["normal"] = input[:, 3:6] * 2 - 1.0
            other_inp["depth"] = input[:, 6:] * 2 - 1.0
        else:
            raise NotImplementedError
        return raw_input, other_inp

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_distances_relative: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        current_step_ratio=0,
        adjusted_guidance_scale=None,
        guidance_eval=False,  # Add guidance_eval parameter
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        camera = c2w
        input_dtype = rgb.dtype

        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        rgb_BCHW, other_inp = self.parse_input(rgb_BCHW, self.cond_method)

        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 32 32"]
            if rgb_as_latents:
                # latents = F.interpolate(rgb_BCHW, (32, 32), mode='bilinear', align_corners=False)  # need [-1, 1]
                latents = F.adaptive_avg_pool2d(rgb_BCHW, (32, 32))
            else:
                # interp to 256x256 to be fed into vae.
                pred_rgb = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size, self.cfg.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        # sample timestep
        if timestep is None:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=latents.device,
            )
            t_expand = t.repeat(text_embeddings.shape[0])

        else:
            # assert timestep >= 0 and timestep < self.num_train_timesteps
            # t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
            # t_expand = t.repeat(text_embeddings.shape[0])]
            t_expand = timestep

        # predict the noise residual with unet, NO grad!
        with torch.no_grad(), torch.autocast(dtype=self.weights_dtype, device_type="cuda"):
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            if camera is not None:
                if self.cfg.cam_method == "abs_spec":
                    camera = self.get_camera_cond(
                        camera, fovy, distances=self.cfg.camera_distance
                    )
                elif self.cfg.cam_method == "abs":
                    camera = self.get_camera_cond(
                        camera, fovy, distances=camera_distances
                    )
                elif self.cfg.cam_method == "rel_x2":
                    camera = self.get_camera_cond(
                        camera, fovy, distances=camera_distances_relative * 2
                    )
                elif self.cfg.cam_method == "rel_xauto":
                    cam_dist = (camera_distances_relative - 0.8) / (1.0 - 0.8) * (
                        2.0 - 1.4
                    ) + 1.4
                    camera = self.get_camera_cond(camera, fovy, distances=cam_dist)
                else:
                    raise NotImplementedError

                camera = camera.repeat(2, 1).to(text_embeddings)
                context = {
                    "context": text_embeddings.to(self.weights_dtype),
                    "camera": camera.to(self.weights_dtype),
                    "num_frames": self.cfg.n_view,
                }
            else:
                context = {"context": text_embeddings.to(self.weights_dtype)}

            context = self.get_cond_input(
                input=other_inp,
                cond_method=self.cond_method,
                c=context,
                image_size=self.cfg.image_size,
            )

            
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

            noise_pred = noise_pred.to(input_dtype)

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(
            2
        )  # Note: flipped compared to stable-dreamfusion
        
        # Use adjusted_guidance_scale if provided, otherwise use default
        guidance_scale = adjusted_guidance_scale if adjusted_guidance_scale is not None else self.cfg.guidance_scale
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(
                latents_noisy, t, noise_pred
            )

            # clip or rescale x0
            if self.cfg.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(
                    latents_noisy, t, noise_pred_text
                )
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(
                    -1, self.cfg.n_view, *latents_recon_nocfg.shape[1:]
                )
                latents_recon_reshape = latents_recon.view(
                    -1, self.cfg.n_view, *latents_recon.shape[1:]
                )
                factor = (
                    latents_recon_nocfg_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8
                ) / (latents_recon_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8)

                latents_recon_adjust = latents_recon.clone() * factor.squeeze(
                    1
                ).repeat_interleave(self.cfg.n_view, dim=0)
                latents_recon = (
                    self.cfg.recon_std_rescale * latents_recon_adjust
                    + (1 - self.cfg.recon_std_rescale) * latents_recon
                )

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = (
                0.5
                * F.mse_loss(latents, latents_recon.detach(), reduction="sum")
                / latents.shape[0]
            )
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # # Original SDS
            # # w(t), sigma_t^2
            # w = (1 - self.alphas_cumprod[t])

            if self.cfg.weighting_strategy == "sds":
                # w(t), sigma_t^2, alphas t:[0, 1000] -> [1, 0]
                w = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
            elif self.cfg.weighting_strategy == "uniform":
                w = 1
            elif self.cfg.weighting_strategy == "fantasia3d":
                w = (self.alphas_cumprod[t] ** 0.5 * (1 - self.alphas_cumprod[t])).view(
                    -1, 1, 1, 1
                )
            elif self.cfg.weighting_strategy == "fantasia3d_1":
                w = (1.0 / (1 - self.alphas[t])).view(-1, 1, 1, 1)
            elif self.cfg.weighting_strategy == "fantasia3d_2":
                if current_step_ratio <= 0.2:
                    w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
                else:
                    w = (1.0 / (1 - self.alphas[t])).view(-1, 1, 1, 1)
            else:
                raise ValueError(
                    f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
                )

            grad = w * (noise_pred - noise)

            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        result = {
            "loss_sds": loss,
            "grad_norm": grad.norm(),
            "timestep": t_expand,
        }

        # Add guidance evaluation if requested
        if guidance_eval:
            guidance_eval_utils = {
                # "text_embeddings": text_embeddings,
                "t_orig": t,
                "latents_noisy": latents_noisy,
                # "noise_pred": noise_pred,
                # "camera": camera,
                "context": context,
                "rgb_input": rgb,  # <--- 将原始输入rgb传递过去
            }
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            if self.cfg.generate_img:
                gen_img = self.generate_img(
                    context, self.cfg.image_size, batch_size, other_inp, scale=10
                )
                guidance_eval_out["gen_img"] = gen_img
            result.update({"eval": guidance_eval_out})

        return result

    def _get_noise_pred(
        self,
        latents: Float[Tensor, "B 4 32 32"],
        t: Int[Tensor, "B"],
        context: dict,
        guidance_scale: float,
    ) -> Float[Tensor, "B 4 32 32"]:
        """
        辅助函数：预测并合并带指导的噪声。
        """
        # 将输入复制为条件和无条件两部分
        latent_model_input = torch.cat([latents] * 2)
        t_input = t.repeat(2)
        print(f"t_input shape: {t_input.shape}")
        # 使用 self.model.apply_model (即 UNet) 预测噪声
        noise_pred = self.model.apply_model(latent_model_input, t_input, context)

        # 分离条件和无条件的预测结果
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        
        # 应用无分类器指导 (Classifier-Free Guidance)
        noise_pred_guided = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        
        return noise_pred_guided


    # @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig: Int[Tensor, "B"],
        latents_noisy: Float[Tensor, "B 4 32 32"],
        context: dict,
        rgb_input: Float[Tensor, "B H W C"], # 接收原始输入用于可视化
    ):
        # 0. 准备工作
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )
        self.model.device = self.device
        self.sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)
        # 1. 对齐时间步
        sampling_timesteps = torch.linspace(self.num_train_timesteps - 1, 0, 50, device=self.device).long()
        t_orig_bs = t_orig.repeat(bs)
        large_enough_idxs = sampling_timesteps.expand([bs, -1]) >= t_orig_bs.unsqueeze(-1)
        idxs = torch.where(
            torch.any(large_enough_idxs, dim=1),
            torch.min(large_enough_idxs, dim=1)[1],
            torch.tensor(len(sampling_timesteps) - 1, device=self.device)
        )
        t = sampling_timesteps[idxs]

        # 2. 可视化原始输入和加噪后的图像
        out = {
            "bs": bs,
            "noise_levels": (t.float() / self.num_train_timesteps).cpu().numpy(),
            "imgs_noisy": self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)
        }
        if rgb_input is not None:
            out["imgs_normal_input"] = (rgb_input[:bs, ..., :3] * 0.5 + 0.5).clamp(0, 1)
            disparity_rgb = rgb_input[:bs, ..., 3:].repeat(1, 1, 1, 3)
            out["imgs_disparity_input"] = (disparity_rgb * 0.5 + 0.5).clamp(0, 1)

        # 3. 执行一步去噪并可视化
        noise_pred = self._get_noise_pred(latents_noisy[:bs], t, context, self.cfg.guidance_scale)
        
        # We need to create the unconditional conditioning dictionary for the sampler

        prev_sample_list = []
        pred_x0_list = []
        
        # Loop through each item in the batch
        for b in range(bs):
            # Get data for the single item
            latents_b = latents_noisy[b:b+1]
            t_b = t[b:b+1]
            index_b = idxs[b].item() # Get the scalar integer index

            # Create contexts for the single item
            cond_context_b = {k: v[b:b+1] if isinstance(v, torch.Tensor) else v for k, v in context.items()}
            uncond_context_b = {k: v[b+bs:b+bs+1] if isinstance(v, torch.Tensor) else v for k, v in context.items()}
            
            # Call p_sample_ddim for the single item
            prev_sample_b, pred_x0_b = self.sampler.p_sample_ddim(
                latents_b, cond_context_b, t_b,
                index=index_b,
                unconditional_guidance_scale=self.cfg.guidance_scale, 
                unconditional_conditioning=uncond_context_b
            )
            
            prev_sample_list.append(prev_sample_b)
            pred_x0_list.append(pred_x0_b)
        
        prev_sample = torch.cat(prev_sample_list, dim=0)
        pred_x0 = torch.cat(pred_x0_list, dim=0)
        
        out["imgs_1step"] = self.decode_latents(prev_sample).permute(0, 2, 3, 1)
        out["imgs_1orig"] = self.decode_latents(pred_x0).permute(0, 2, 3, 1)
        
        # 4. 从当前状态继续，完成所有剩余的去噪步骤
        # We will loop through each item in the batch individually to handle the indices correctly
        latents_final_list = []
        for b in range(bs):
            latents = prev_sample[b:b+1]
            start_index = idxs[b].item()

            # Loop from the next step after our starting one
            for i in range(start_index + 1, len(sampling_timesteps)):
                t_loop = sampling_timesteps[i]
                
                # FIX: Create separate cond and uncond contexts for the single item
                # This is the same logic we used for the one-step prediction
                cond_context_b = {k: v[b:b+1] if isinstance(v, torch.Tensor) else v for k, v in context.items()}
                uncond_context_b = {k: v[b+bs:b+bs+1] if isinstance(v, torch.Tensor) else v for k, v in context.items()}
                
                # FIX: Remove the redundant _get_noise_pred call
                # noise_pred_b = self._get_noise_pred(...)

                # FIX: Call p_sample_ddim with the corrected separate contexts
                latents, _ = self.sampler.p_sample_ddim(
                    latents, cond_context_b, t_loop.repeat(1),
                    index=i,
                    unconditional_guidance_scale=self.cfg.guidance_scale,
                    unconditional_conditioning=uncond_context_b
                )
            latents_final_list.append(latents)
            
        latents_final = torch.cat(latents_final_list, dim=0)
        out["imgs_final"] = self.decode_latents(latents_final).permute(0, 2, 3, 1)
        
        return out
    
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
