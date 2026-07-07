import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-controlnet-guidance")
class StableDiffusionControlNetGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        view_dependent_prompting: bool = True
        
        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion with ControlNet...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Delete the text encoder as it is not used here
        del self.pipe.text_encoder
        cleanup()

        # Set up components
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.controlnet = self.pipe.controlnet.eval()
        self.scheduler = self.pipe.scheduler

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.controlnet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None
        threestudio.info(f"Loaded Stable Diffusion with ControlNet!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        down_block_additional_residuals,
        mid_block_additional_residual,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
        
    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            latents = self.encode_images(rgb_BCHW_512)
        return latents

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        depth_map=None,
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)

        # Get view-dependent text embeddings
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )

        # 确保数据类型一致
        text_embeddings = text_embeddings.to(self.weights_dtype)
        depth_map = depth_map.to(self.weights_dtype)

        # SDS logic
        with torch.no_grad():
            # Sample a random timestep
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )
            # Add noise to latents
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            
            # Prepare inputs for CFG
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            t_input = t.repeat(2)
            
            # Prepare ControlNet condition
            controlnet_cond = torch.cat([depth_map] * 2, dim=0).to(self.weights_dtype)

            # Get ControlNet residuals
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t_input,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=controlnet_cond,
                return_dict=False,
            )

            # Predict noise with UNet
            noise_pred = self.forward_unet(
                latent_model_input,
                t_input,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=[
                    res.to(self.weights_dtype) for res in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(self.weights_dtype),
            )

        # Perform Classifier-Free Guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # Compute SDS loss
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        # Clip grad for stable training
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # Reparameterization trick
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if guidance_eval:
            guidance_eval_utils = {
                "t_orig": t,
                "latents_noisy": latents_noisy,
                "noise_pred": noise_pred,
                "text_embeddings": text_embeddings,
                "controlnet_cond": depth_map, # Pass the un-duplicated condition
            }
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})
            print(f"Guidance eval: {guidance_out['eval']['bs']} items")

        return guidance_out

    # ADDED: Helper for the evaluation loop
    @torch.no_grad()
    def _get_noise_pred_eval(
        self,
        latents: Float[Tensor, "B 4 H W"],
        t: Int[Tensor, "B"],
        text_embeddings: Float[Tensor, "BB 77 1024"],
        controlnet_cond: Float[Tensor, "B C H W"],
    ):
        # 确保所有输入都转换为正确的数据类型
        latents_input = torch.cat([latents] * 2).to(self.weights_dtype)
        t_input = t.repeat(2).to(self.weights_dtype)
        controlnet_cond_input = torch.cat([controlnet_cond] * 2).to(self.weights_dtype)
        text_embeddings = text_embeddings.to(self.weights_dtype)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latents_input,
            t_input,
            encoder_hidden_states=text_embeddings,
            controlnet_cond=controlnet_cond_input,
            timestep_cond=None,
            return_dict=False,
        )
        
        noise_pred = self.forward_unet(
            latents_input,
            t_input,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=[res.to(self.weights_dtype) for res in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample.to(self.weights_dtype),
        )
        
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred_guided = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred_guided

    # ADDED: Main evaluation method
    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        controlnet_cond,
    ):
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[:bs].unsqueeze(-1)
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            alpha_prod_t = self.scheduler.alphas_cumprod[t[b]]
            beta_prod_t = 1 - alpha_prod_t
            
            # 这就是计算公式的实现
            pred_original_sample_b = (
                latents_noisy[b : b + 1] - (beta_prod_t ** 0.5) * noise_pred[b : b + 1]
            ) / (alpha_prod_t ** 0.5)
            
            pred_1orig.append(pred_original_sample_b)
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1]
            )
            # print(f"Step {t[b].item()}: alpha {self.scheduler.alphas_cumprod[t[b]].item():.4f} -> {self.scheduler.alphas_cumprod[self.scheduler.timesteps_gpu[idxs[b]+1]].item():.4f}")
            # print(f"step output keys: {step_output.keys()}")
            latents_1step.append(step_output["prev_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            # 确保使用正确的数据类型
            text_emb_cond = text_embeddings[[b, b + bs], ...].to(self.weights_dtype)
            cond_b = controlnet_cond[b: b+1].to(self.weights_dtype)
            latents = latents.to(self.weights_dtype)

            for t_loop in tqdm(self.scheduler.timesteps_gpu[i + 1 :], leave=False):
                noise_pred_loop = self._get_noise_pred_eval(
                    latents, t_loop.reshape(1), text_emb_cond, cond_b
                )
                latents = self.scheduler.step(noise_pred_loop, t_loop, latents)["prev_sample"]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )