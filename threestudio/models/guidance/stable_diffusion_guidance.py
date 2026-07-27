import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-guidance")
class StableDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        enable_torch_compile: bool = False
        torch_compile_mode: str = "max-autotune"
        torch_compile_fullgraph: bool = False
        torch_compile_dynamic: Optional[bool] = None
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None

        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

        collect_inputs: Optional[list] = field(default_factory=lambda: ["comp_rgb"])
        collect_inputs_lat: Optional[list] = field(default_factory=lambda: [])
        guidance_eval_return_noise: bool = False

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
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
            self.pipe.vae.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        self._vae_decode_compiled: Optional[Callable[[Tensor], Tensor]] = None
        if self.cfg.enable_torch_compile:
            if parse_version(torch.__version__) < parse_version("2.0"):
                threestudio.warn(
                    f"torch.compile requires PyTorch>=2.0, got {torch.__version__}; skipping."
                )
            elif self.cfg.enable_sequential_cpu_offload:
                threestudio.warn(
                    "torch.compile is not supported with `enable_sequential_cpu_offload`; skipping."
                )
            else:
                try:
                    compile_kwargs = {
                        "mode": self.cfg.torch_compile_mode,
                        "fullgraph": self.cfg.torch_compile_fullgraph,
                        "dynamic": self.cfg.torch_compile_dynamic,
                    }
                    self.unet = torch.compile(self.unet, **compile_kwargs)
                    # NOTE: diffusers `AutoencoderKL.decode()` constructs dataclass outputs even when
                    # `return_dict=False`, which can break torch.compile. Use a tensor-only wrapper.
                    if getattr(self.vae, "use_slicing", False):
                        threestudio.warn(
                            "VAE slicing is enabled; skipping torch.compile for VAE decode."
                        )
                    else:
                        self._vae_decode_compiled = torch.compile(
                            self._vae_decode_tensor, **compile_kwargs
                        )
                    if self._vae_decode_compiled is not None:
                        threestudio.info(
                            "Enabled torch.compile for Stable Diffusion UNet and VAE decode wrapper."
                        )
                    else:
                        threestudio.info(
                            "Enabled torch.compile for Stable Diffusion UNet."
                        )
                except Exception as e:
                    threestudio.warn(
                        f"torch.compile failed, continuing without it: {type(e).__name__}: {e}"
                    )

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")

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
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            return_dict=False,
        )[0].to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype), return_dict=False)[0]
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = latents.to(self.weights_dtype)
        if self._vae_decode_compiled is not None:
            try:
                image = self._vae_decode_compiled(latents)
            except Exception as e:
                threestudio.warn(
                    f"Compiled VAE decode failed, falling back to eager: {type(e).__name__}: {e}"
                )
                self._vae_decode_compiled = None
                image = self._vae_decode_tensor(latents)
        else:
            image = self._vae_decode_tensor(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def _vae_decode_tensor(self, z: Tensor) -> Tensor:
        if getattr(self.vae, "use_slicing", False) and z.shape[0] > 1:
            decoded_slices = [
                self.vae._decode(z_slice, return_dict=False)[0]
                for z_slice in z.split(1)
            ]
            return torch.cat(decoded_slices)
        return self.vae._decode(z, return_dict=False)[0]

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        current_step_ratio=0,
        adjusted_guidance_scale=None,
    ):
        batch_size = elevation.shape[0]

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            guidance_scale = adjusted_guidance_scale if adjusted_guidance_scale is not None else self.cfg.guidance_scale
            noise_pred = noise_pred_uncond + guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            guidance_scale = adjusted_guidance_scale if adjusted_guidance_scale is not None else self.cfg.guidance_scale
            noise_pred = noise_pred_text + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2, alphas t:[0, 1000] -> [1, 0]
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)

        elif self.cfg.weighting_strategy == "fantasia3d_1":
            w = (1.0 / (1 - self.alphas[t])).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "fantasia3d_2":
            # print(current_step_ratio)
            if current_step_ratio <= 0.2:
                w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
            else:
                w = (1.0 / (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_added": noise,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        adjusted_guidance_scale=None,
    ):
        batch_size = elevation.shape[0]

        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                y = latents
                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)
                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                y = latents

                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)

                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                
                # Use adjusted_guidance_scale if provided, otherwise use default
                guidance_scale = adjusted_guidance_scale if adjusted_guidance_scale is not None else self.cfg.guidance_scale
                noise_pred = noise_pred_text + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

        Ds = zs - sigma * noise_pred

        if self.cfg.var_red:
            grad = -(Ds - y) / sigma
        else:
            grad = -(Ds - zs) / sigma

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": scaled_zs,
            "noise_added": noise,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        current_step_ratio=0,
        adjusted_guidance_scale=None,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if self.cfg.use_sjc:
            grad, guidance_eval_utils = self.compute_grad_sjc(
                latents, t, prompt_utils, elevation, azimuth, camera_distances,
                adjusted_guidance_scale=adjusted_guidance_scale,
            )
        else:
            grad, guidance_eval_utils = self.compute_grad_sds(
                latents,
                t,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                current_step_ratio,
                adjusted_guidance_scale=adjusted_guidance_scale,
            )

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
            "timestep": t,
        }

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            return_noise = kwargs.get(
                "guidance_eval_return_noise", self.cfg.guidance_eval_return_noise
            )
            if return_noise:
                def noise_to_color_vis(noise_src: Tensor, bs_vis: int) -> Tensor:
                    noise_src = noise_src[:bs_vis].detach().float()
                    noise_mag = torch.linalg.norm(noise_src, ord=2, dim=1, keepdim=True)
                    noise_flat = noise_mag.reshape(bs_vis, -1)
                    noise_min = noise_flat.min(dim=1, keepdim=True)[0].reshape(
                        bs_vis, 1, 1, 1
                    )
                    noise_max = noise_flat.max(dim=1, keepdim=True)[0].reshape(
                        bs_vis, 1, 1, 1
                    )
                    noise_norm = (noise_mag - noise_min) / (noise_max - noise_min + 1e-6)
                    x = noise_norm[:, 0, ...]  # [B, H, W], in [0, 1]
                    r = torch.clamp(1.5 - torch.abs(4.0 * x - 3.0), 0.0, 1.0)
                    g = torch.clamp(1.5 - torch.abs(4.0 * x - 2.0), 0.0, 1.0)
                    b = torch.clamp(1.5 - torch.abs(4.0 * x - 1.0), 0.0, 1.0)
                    return torch.stack([r, g, b], dim=-1)

                noise_added_vis_src = guidance_eval_utils.get("noise_added", None)
                noise_pred_vis_src = guidance_eval_utils.get("noise_pred", None)
                bs = guidance_eval_out["bs"]
                if noise_added_vis_src is not None and bs > 0:
                    guidance_eval_out.update(
                        {"noise_added_vis": noise_to_color_vis(noise_added_vis_src, bs)}
                    )
                if noise_pred_vis_src is not None and bs > 0:
                    guidance_eval_out.update(
                        {"noise_pred_vis": noise_to_color_vis(noise_pred_vis_src, bs)}
                    )
            bs = guidance_eval_out["bs"]
            if bs > 0:
                guidance_input_vis = rgb[:bs].detach().float()
                if guidance_input_vis.shape[-1] == 1:
                    guidance_input_vis = guidance_input_vis.repeat(1, 1, 1, 3)
                elif guidance_input_vis.shape[-1] > 3:
                    # Keep the first 3 channels for visualization.
                    guidance_input_vis = guidance_input_vis[..., :3]
                if guidance_input_vis.min() < 0.0:
                    guidance_input_vis = (guidance_input_vis + 1.0) / 2.0
                guidance_input_vis = guidance_input_vis.clamp(0.0, 1.0)
                guidance_eval_out.update({"guidance_input_vis": guidance_input_vis})
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
        adjusted_guidance_scale=None,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            # Use adjusted_guidance_scale if provided, otherwise use default
            guidance_scale = adjusted_guidance_scale if adjusted_guidance_scale is not None else self.cfg.guidance_scale
            
            noise_pred = noise_pred_uncond + guidance_scale * (
                e_pos + accum_grad
            )
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            
            # Use adjusted_guidance_scale if provided, otherwise use default
            guidance_scale = adjusted_guidance_scale if adjusted_guidance_scale is not None else self.cfg.guidance_scale
            noise_pred = noise_pred_text + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        noise_added=None,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
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
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
