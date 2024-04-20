# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List, Optional
import argparse
import ast
from pathlib import Path
import gc, os
import numpy as np

import torch
from tqdm import tqdm
from PIL import Image



import train_util
import random
import model_util
import prompt_util
from prompt_util import (
    PromptEmbedsCache,
    PromptEmbedsPair,
    PromptSettings,
    PromptEmbedsXL,
)
import debug_util
import config_util
from config_util import RootConfig

import wandb
from compel import Compel, ReturnedEmbeddingsType

NUM_IMAGES_PER_PROMPT = 1
from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV

def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(
    config: RootConfig,
    prompts: list[PromptSettings],
    device,
    folder_main: str,
    folders,
    scales,
    folder_caption
):
    scales = np.array(scales)
    folders = np.array(folders)
    scales_unique = list(scales)

    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    save_path = Path(config.save.path)

    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    if config.logging.verbose:
        print(metadata)

    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    save_suffix = config.save.save_suffix

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    (
        tokenizers,
        text_encoders,
        unet,
        noise_scheduler,
        vae
    ) = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )

    for text_encoder in text_encoders:
        text_encoder.to(device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
    
    
    compel = Compel(tokenizer=[tokenizers[0], tokenizers[1]] , text_encoder=[text_encoders[0], text_encoders[1]], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])


    unet.to(device, dtype=weight_dtype)
    if config.other.use_xformers:
        unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()
    
    vae.to(device)
    vae.requires_grad_(False)
    vae.eval()
    
    network = LoRANetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)

    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    #optimizer_args
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
            
    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs)
    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()

    print("Prompts")
    for settings in prompts:
        print(settings)

    # debug
    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    img_ext = '.webp'
    with torch.no_grad():
        for settings in prompts:
            print(settings)
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                if cache[prompt] == None:
                    tex_embs, pool_embs = train_util.encode_prompts_xl(
                            tokenizers,
                            text_encoders,
                            [prompt],
                            num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                        )
                    cache[prompt] = PromptEmbedsXL(
                        tex_embs,
                        pool_embs
                    )

            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        del tokenizer, text_encoder

    flush()

    # pbar = tqdm(range(config.train.iterations))
    captions = os.listdir(folder_caption)
    
    print('shuffle captions')
    random.shuffle(captions)

    total_step = len(captions) * config.train.iterations

    # loop all image
    pbar = tqdm(range(total_step))

    loss = None


    for i in pbar:
        if config.save.break_after_steps > 0 and i >= config.save.break_after_steps:
            break;

        scale_to_look = abs(random.choice(list(scales_unique)))
        folder1 = folders[scales==-scale_to_look][0]
        folder2 = folders[scales==scale_to_look][0]
        
        ims = os.listdir(f'{folder_main}/{folder1}/')
        ims = [im_ for im_ in ims if '.png' in im_ or '.jpg' in im_ or '.jpeg' in im_ or '.webp' in im_]
        # random_sampler = random.randint(0, len(ims)-1)

        caption_index = i
        # get caption_index if i >= len(captions)
        if i >= len(captions):
            caption_index = i % len(captions)
        # get filename from image
        caption_file = captions[caption_index]
        caption_name = caption_file.split('.')[0]
        
        caption_path = f'{folder_caption}/{caption_file}'

        caption = ""
        if random.random() > config.train.caption_drop_out:
            # read caption from file
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read()
        else:
            print(f"caption drop out")
        
        print(f'caption:{caption}')
        for repeat in range(config.train.repeat):
            # print(f"repeat:{repeat}")
            with torch.no_grad():
                noise_scheduler.set_timesteps(
                    config.train.max_denoising_steps, device=device
                )

                optimizer.zero_grad()

                # prompt_pair: PromptEmbedsPair = prompt_pairs[
                #     torch.randint(0, len(prompt_pairs), (1,)).item()
                # ]
                # tex_embs, pool_embs = train_util.encode_prompts_xl(
                #         tokenizers,
                #         text_encoders,
                #         [caption],
                #         num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                #     )
                
                # change to use compel
                tex_embs, pool_embs = compel(caption)
                
                caption_embs = PromptEmbedsXL(
                    tex_embs,
                    pool_embs
                )

                empty_tex_embs, empty_pool_embs = train_util.encode_prompts_xl(
                            tokenizers,
                            text_encoders,
                            [''],
                            num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                        )
                empty_embs = PromptEmbedsXL(
                    empty_tex_embs,
                    empty_pool_embs
                )
                
                for settings in prompts:
                    print(settings)
                    # use caption embs instead of settings
                    prompt_pair: PromptEmbedsPair = PromptEmbedsPair(
                        criteria,
                        caption_embs,
                        caption_embs,
                        empty_embs,
                        empty_embs,
                        settings,
                    )

                # 1 ~ 49 からランダム
                timesteps_to = torch.randint(
                    1, config.train.max_denoising_steps, (1,)
                ).item()

                # height, width = prompt_pair.resolution, prompt_pair.resolution
                # if prompt_pair.dynamic_resolution:
                #     height, width = train_util.get_random_resolution_in_bucket(
                #         prompt_pair.resolution
                #     )

                if config.logging.verbose:
                    print("guidance_scale:", prompt_pair.guidance_scale)
                    print("resolution:", prompt_pair.resolution)
                    print("dynamic_resolution:", prompt_pair.dynamic_resolution)
                    if prompt_pair.dynamic_resolution:
                        print("bucketed resolution:", (height, width))
                    print("batch_size:", prompt_pair.batch_size)
                    print("dynamic_crops:", prompt_pair.dynamic_crops)

                img1 = Image.open(f'{folder_main}/{folder1}/{caption_name}{img_ext}')
                img2 = Image.open(f'{folder_main}/{folder2}/{caption_name}{img_ext}')
                # width,height = (img2.width//2,img2.height//2)
                # print(f'resize_width_height:{width},{height}')
                # img2.resize(width,height)
                # # image1 resize to image2 width height
                # img1.resize(width,height)
                width = img2.width
                height = img2.height

                seed = random.randint(0,2*15)
                
                generator = torch.manual_seed(seed)
                denoised_latents_low, low_noise = train_util.get_noisy_image(
                    img1,
                    vae,
                    generator,
                    unet,
                    noise_scheduler,
                    start_timesteps=0,
                    total_timesteps=timesteps_to)
                denoised_latents_low = denoised_latents_low.to(device, dtype=weight_dtype)
                low_noise = low_noise.to(device, dtype=weight_dtype)
                
                generator = torch.manual_seed(seed)
                denoised_latents_high, high_noise = train_util.get_noisy_image(
                    img2,
                    vae,
                    generator,
                    unet,
                    noise_scheduler,
                    start_timesteps=0,
                    total_timesteps=timesteps_to)
                denoised_latents_high = denoised_latents_high.to(device, dtype=weight_dtype)
                high_noise = high_noise.to(device, dtype=weight_dtype)
                noise_scheduler.set_timesteps(1000)

                add_time_ids = train_util.get_add_time_ids(
                    height,
                    width,
                    dynamic_crops=prompt_pair.dynamic_crops,
                    dtype=weight_dtype,
                ).to(device, dtype=weight_dtype)


                current_timestep = noise_scheduler.timesteps[
                    int(timesteps_to * 1000 / config.train.max_denoising_steps)
                ]
                try:
                    # with network: の外では空のLoRAのみが有効になる
                    high_latents = train_util.predict_noise_xl(
                        unet,
                        noise_scheduler,
                        current_timestep,
                        denoised_latents_high,
                        text_embeddings=train_util.concat_embeddings(
                            prompt_pair.unconditional.text_embeds,
                            prompt_pair.positive.text_embeds,
                            prompt_pair.batch_size,
                        ),
                        add_text_embeddings=train_util.concat_embeddings(
                            prompt_pair.unconditional.pooled_embeds,
                            prompt_pair.positive.pooled_embeds,
                            prompt_pair.batch_size,
                        ),
                        add_time_ids=train_util.concat_embeddings(
                            add_time_ids, add_time_ids, prompt_pair.batch_size
                        ),
                        guidance_scale=1,
                    ).to(device, dtype=torch.float32)
                except:
                    flush()
                    print(f'Error Occured!: {np.array(img1).shape} {np.array(img2).shape}')
                    continue
                # with network: の外では空のLoRAのみが有効になる
                
                low_latents = train_util.predict_noise_xl(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    denoised_latents_low,
                    text_embeddings=train_util.concat_embeddings(
                        prompt_pair.unconditional.text_embeds,
                        prompt_pair.neutral.text_embeds,
                        prompt_pair.batch_size,
                    ),
                    add_text_embeddings=train_util.concat_embeddings(
                        prompt_pair.unconditional.pooled_embeds,
                        prompt_pair.neutral.pooled_embeds,
                        prompt_pair.batch_size,
                    ),
                    add_time_ids=train_util.concat_embeddings(
                        add_time_ids, add_time_ids, prompt_pair.batch_size
                    ),
                    guidance_scale=1,
                ).to(device, dtype=torch.float32)
                
                
                
                # if config.logging.verbose:
                #     print("positive_latents:", positive_latents[0, 0, :5, :5])
                #     print("neutral_latents:", neutral_latents[0, 0, :5, :5])
                #     print("unconditional_latents:", unconditional_latents[0, 0, :5, :5])
                    
            network.set_lora_slider(scale=scale_to_look)
            with network:
                target_latents_high = train_util.predict_noise_xl(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    denoised_latents_high,
                    text_embeddings=train_util.concat_embeddings(
                        prompt_pair.unconditional.text_embeds,
                        prompt_pair.positive.text_embeds,
                        prompt_pair.batch_size,
                    ),
                    add_text_embeddings=train_util.concat_embeddings(
                        prompt_pair.unconditional.pooled_embeds,
                        prompt_pair.positive.pooled_embeds,
                        prompt_pair.batch_size,
                    ),
                    add_time_ids=train_util.concat_embeddings(
                        add_time_ids, add_time_ids, prompt_pair.batch_size
                    ),
                    guidance_scale=1,
                ).to(device, dtype=torch.float32)

            high_latents.requires_grad = False
            low_latents.requires_grad = False
            
            loss_high = criteria(target_latents_high, high_noise.to(torch.float32))
            pbar.set_description(f"Loss*1k: {loss_high.item()*1000:.4f}")
            if config.logging.use_wandb:
                wandb.log(
                    {"loss_high": loss_high, "iteration": i, "lr": lr_scheduler.get_last_lr()[0]}
                )
            loss_high.backward()
            
            # opposite
            network.set_lora_slider(scale=-scale_to_look)
            with network:
                target_latents_low = train_util.predict_noise_xl(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    denoised_latents_low,
                    text_embeddings=train_util.concat_embeddings(
                        prompt_pair.unconditional.text_embeds,
                        prompt_pair.neutral.text_embeds,
                        prompt_pair.batch_size,
                    ),
                    add_text_embeddings=train_util.concat_embeddings(
                        prompt_pair.unconditional.pooled_embeds,
                        prompt_pair.neutral.pooled_embeds,
                        prompt_pair.batch_size,
                    ),
                    add_time_ids=train_util.concat_embeddings(
                        add_time_ids, add_time_ids, prompt_pair.batch_size
                    ),
                    guidance_scale=1,
                ).to(device, dtype=torch.float32)


            high_latents.requires_grad = False
            low_latents.requires_grad = False
            
            loss_low = criteria(target_latents_low, low_noise.to(torch.float32))
            pbar.set_description(f"Loss*1k: {loss_low.item()*1000:.4f}")
            if config.logging.use_wandb:
                wandb.log(
                    {"loss_low": loss_low, "iteration": i, "lr": lr_scheduler.get_last_lr()[0]}
                )
            loss_low.backward()
            
            
            optimizer.step()
            lr_scheduler.step()

            del (
                high_latents,
                low_latents,
                target_latents_low,
                target_latents_high,
            )
            flush()
            
            if (
                i % config.save.per_steps == 0
                and i != 0
                # and i != config.train.iterations - 1
                and i != total_step - 1
                # only save after certian steps
                and i > config.save.save_after_steps
            ):
                print("Saving...")
                save_path.mkdir(parents=True, exist_ok=True)
                network.save_weights(
                    save_path / f"{config.save.name}_{i}_{save_suffix}.safetensors",
                    dtype=save_weight_dtype,
                )

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_{i}_{save_suffix}.safetensors",
        dtype=save_weight_dtype,
    )

    del (
        unet,
        noise_scheduler,
        loss,
        optimizer,
        network,
    )

    flush()

    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_util.load_config_from_yaml(config_file)
    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(',')
        attributes = [a.strip() for a in attributes]
    
    config.network.alpha = args.alpha
    config.network.rank = args.rank
    # config.save.name += f'_alpha{args.alpha}'
    # config.save.name += f'_rank{config.network.rank }'
    # config.save.name += f'_{config.network.training_method}'
    config.save.path += f'/{config.save.name}'
    
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    
    device = torch.device(f"cuda:{args.device}")
    
    folders = args.folders.split(',')
    folders = [f.strip() for f in folders]
    scales = args.scales.split(',')
    scales = [f.strip() for f in scales]
    scales = [int(s) for s in scales]
    
    print(folders, scales)
    if len(scales) != len(folders):
        raise Exception('the number of folders need to match the number of scales')
    
    if args.stylecheck is not None:
        check = args.stylecheck.split('-')
        
        for i in range(int(check[0]), int(check[1])):
            folder_main = args.folder_main+ f'{i}'
            config.save.name = f'{os.path.basename(folder_main)}'
            # config.save.name += f'_alpha{args.alpha}'
            # config.save.name += f'_rank{config.network.rank }'
            config.save.path = f'models/{config.save.name}'
            train(config=config, prompts=prompts, device=device, folder_main = folder_main, folders = folders, scales = scales, folder_caption = args.folder_caption)
    else:
        train(config=config, prompts=prompts, device=device, folder_main = args.folder_main, folders = folders, scales = scales, folder_caption = args.folder_caption)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Config file for training.",
    )
    # config_file 'data/config.yaml'
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="LoRA weight.",
    )
    # --alpha 1.0
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=4,
    )
    # --rank 4
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to train on.",
    )
    # --device 0
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Device to train on.",
    )
    # --name 'eyesize_slider'
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default=None,
        help="attritbutes to disentangle (comma seperated string)",
    )
    parser.add_argument(
        "--folder_main",
        type=str,
        required=True,
        help="The folder to check",
    )
    
    parser.add_argument(
        "--stylecheck",
        type=str,
        required=False,
        default = None,
        help="The folder to check",
    )
    
    parser.add_argument(
        "--folders",
        type=str,
        required=False,
        default = 'verylow, low, high, veryhigh',
        help="folders with different attribute-scaled images",
    )
    parser.add_argument(
        "--scales",
        type=str,
        required=False,
        default = '-2, -1, 1, 2',
        help="scales for different attribute-scaled images",
    )
    
    parser.add_argument(
        "--folder_caption",
        type=str,
        required=True,
        help="The folder contains image caption",
    )
    
    args = parser.parse_args()

    main(args)
