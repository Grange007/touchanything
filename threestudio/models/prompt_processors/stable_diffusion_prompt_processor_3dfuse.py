import json
import os
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from transformers import BertForMaskedLM

import threestudio
from threestudio.models.prompt_processors.base import (PromptProcessor,
                                                       hash_prompt,)
from threestudio.utils.misc import barrier, cleanup, get_rank
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-prompt-processor-3dfuse")
class StableDiffusionPromptProcessor3DFuse(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pretrained_model_name_or_path_aux: str = "stabilityai/stable-diffusion-2-1-base"
    

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]

        return text_embeddings, uncond_text_embeddings

    # 在 StableDiffusionPromptProcessor3DFuse 类中添加这两个方法

    def load_text_embeddings(self):
        # 同步所有进程，确保 spawn_func 已经把所有缓存文件都写完了
        barrier()
        
        # --- 加载主编码 ---
        self.text_embeddings = self.load_from_cache(self.prompt, self.cfg.pretrained_model_name_or_path)[None, ...]
        self.uncond_text_embeddings = self.load_from_cache(self.negative_prompt, self.cfg.pretrained_model_name_or_path)[None, ...]
        self.text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt, self.cfg.pretrained_model_name_or_path) for prompt in self.prompts_vd], dim=0
        )
        self.uncond_text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt, self.cfg.pretrained_model_name_or_path) for prompt in self.negative_prompts_vd], dim=0
        )

        # --- 加载辅助编码 (这是新增的关键部分) ---
        self.text_embeddings_aux = self.load_from_cache(self.prompt, self.cfg.pretrained_model_name_or_path_aux)[None, ...]
        self.uncond_text_embeddings_aux = self.load_from_cache(self.negative_prompt, self.cfg.pretrained_model_name_or_path_aux)[None, ...]
        self.text_embeddings_vd_aux = torch.stack(
            [self.load_from_cache(prompt, self.cfg.pretrained_model_name_or_path_aux) for prompt in self.prompts_vd], dim=0
        )
        self.uncond_text_embeddings_vd_aux = torch.stack(
            [self.load_from_cache(prompt, self.cfg.pretrained_model_name_or_path_aux) for prompt in self.negative_prompts_vd], dim=0
        )

        threestudio.debug(f"Loaded both primary and auxiliary text embeddings.")

    def load_from_cache(self, prompt: str, pretrained_model_name_or_path: str):
        # 使用模型路径和提示词共同生成唯一的哈希值
        cache_path = os.path.join(
            self._cache_dir,
            f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Text embedding file {cache_path} for model {pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )
        return torch.load(cache_path, map_location=self.device)

    ###
    def get_multi_text_embeddings(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> Tuple[Float[Tensor, "BB N Nf"], Float[Tensor, "BB N Nf_aux"]]:
        batch_size = elevation.shape[0]

        if view_dependent_prompting:
            # Get direction index based on camera angles
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in self.directions:
                direction_idx[
                    d.condition(elevation, azimuth, camera_distances)
                ] = self.direction2idx[d.name]

            # Get view-dependent text embeddings for both primary and auxiliary encoders
            text_embeddings = self.text_embeddings_vd[direction_idx]
            uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]
            
            text_embeddings_aux = self.text_embeddings_vd_aux[direction_idx]
            uncond_text_embeddings_aux = self.uncond_text_embeddings_vd_aux[direction_idx]
        else:
            # Get view-independent text embeddings
            text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)
            uncond_text_embeddings = self.uncond_text_embeddings.expand(batch_size, -1, -1)

            text_embeddings_aux = self.text_embeddings_aux.expand(batch_size, -1, -1)
            uncond_text_embeddings_aux = self.uncond_text_embeddings_aux.expand(batch_size, -1, -1)

        # Concatenate conditional and unconditional embeddings for both sets
        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        # print( "text_embeddings shape:", text_embeddings.shape )
        # print( "text_embeddings_aux shape:", text_embeddings_aux.shape )
        # print( "uncond_text_embeddings shape:", uncond_text_embeddings.shape )
        # print( "uncond_text_embeddings_aux shape:", uncond_text_embeddings_aux.shape )
        embeddings = torch.cat([text_embeddings, uncond_text_embeddings], dim=0)
        embeddings_aux = torch.cat([text_embeddings_aux, uncond_text_embeddings_aux], dim=0)
        
        return embeddings, embeddings_aux

    # 将此方法添加到 StableDiffusionPromptProcessor3DFuse 类中

    @rank_zero_only
    def prepare_text_embeddings(self):
        os.makedirs(self._cache_dir, exist_ok=True)

        all_prompts = (
            [self.prompt]
            + [self.negative_prompt]
            + self.prompts_vd
            + self.negative_prompts_vd
        )
        prompts_to_process = []
        
        # 分别为两个模型检查缓存
        for prompt in all_prompts:
            # 检查主模型的缓存
            cache_path_primary = os.path.join(
                self._cache_dir,
                f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
            )
            # 检查辅助模型的缓存
            cache_path_aux = os.path.join(
                self._cache_dir,
                f"{hash_prompt(self.cfg.pretrained_model_name_or_path_aux, prompt)}.pt",
            )
            # 只要有一个缓存不存在，就需要处理这个prompt
            if not os.path.exists(cache_path_primary) or not os.path.exists(cache_path_aux):
                prompts_to_process.append(prompt)

        # 去重，避免重复处理
        prompts_to_process = sorted(list(set(prompts_to_process)))

        if len(prompts_to_process) > 0:
            threestudio.info(f"Preparing text embeddings for {len(prompts_to_process)} prompts.")
            if self.cfg.spawn:
                ctx = mp.get_context("spawn")
                subprocess = ctx.Process(
                    target=self.spawn_func,
                    args=(
                        self.cfg.pretrained_model_name_or_path,
                        self.cfg.pretrained_model_name_or_path_aux,
                        prompts_to_process,
                        self._cache_dir,
                        self.device,
                    ),
                )
                subprocess.start()
                subprocess.join()
            else:
                self.spawn_func(
                    self.cfg.pretrained_model_name_or_path,
                    self.cfg.pretrained_model_name_or_path_aux,
                    prompts_to_process,
                    self._cache_dir,
                    self.device,
                )
            cleanup()
        

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, pretrained_model_name_or_path_aux, prompts, cache_dir, device):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            # --- 1. 加载主编码器 (例如来自 SD v1.5) ---
            print(f"Spawning process: Loading primary text encoder from {pretrained_model_name_or_path}...")
            tokenizer_primary = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, subfolder="tokenizer"
            )
            text_encoder_primary = CLIPTextModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder"
            ).to(device)

            # --- 2. 加载辅助编码器 (例如来自 SD v2.1) ---
            print(f"Spawning process: Loading auxiliary text encoder from {pretrained_model_name_or_path_aux}...")
            # 辅助编码器的分词器可能与主编码器不同
            tokenizer_aux = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path_aux, subfolder="tokenizer"
            )
            text_encoder_aux = CLIPTextModel.from_pretrained(
                pretrained_model_name_or_path_aux, subfolder="text_encoder"
            ).to(device)
            
            with torch.no_grad():
                # --- 3. 使用主编码器进行编码 ---
                tokens_primary = tokenizer_primary(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer_primary.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                embeddings_primary = text_encoder_primary(
                    tokens_primary.input_ids.to(device)
                )[0]

                # --- 4. 使用辅助编码器进行编码 ---
                tokens_aux = tokenizer_aux(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer_aux.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                embeddings_aux = text_encoder_aux(
                    tokens_aux.input_ids.to(device)
                )[0]

            # --- 5. 遍历所有提示词，并将两种编码分别存入缓存 ---
            for i, prompt in enumerate(prompts):
                # 保存主编码
                cache_path_primary = os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                )
                torch.save(embeddings_primary[i], cache_path_primary)

                # 保存辅助编码
                cache_path_aux = os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path_aux, prompt)}.pt",
                )
                torch.save(embeddings_aux[i], cache_path_aux)
                
            print("Spawning process: All embeddings have been cached.")

        finally:
            # --- 6. 清理内存 ---
            del text_encoder_primary, text_encoder_aux
            cleanup()
