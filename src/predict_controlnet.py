import os
from typing import List, Dict, Optional
import requests
import random
import string
import hashlib

import torch
torch.backends.cuda.matmul.allow_tf32 = True

from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler
)
from diffusers.utils import load_image
from diffusers.models.attention_processor import AttnProcessor2_0

from compel import Compel

import cv2
import numpy as np
from PIL import Image

# MODEL_ID refers to a diffusers-compatible model on HuggingFace
# e.g. prompthero/openjourney-v2, wavymulder/Analog-Diffusion, etc
MODEL_ID = "SdValar/deliberate2"
MODEL_CACHE = "diffusers-cache"

pipe = None
pipe_img2img = None
compel = None

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
if torch.cuda.is_available():
    device = "cuda"

def setup():
    """Load the model into memory to make running multiple predictions efficient"""
    print("Loading pipeline...")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE,
        local_files_only=True,
    )

    global pipe
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        controlnet=controlnet,
        cache_dir=MODEL_CACHE,
        local_files_only=True,
        safety_checker=None,
    ).to(device)
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.controlnet.to(memory_format=torch.channels_last)
    # pipe.enable_vae_tiling()

    # pipe.unet = torch.compile(pipe.unet)
    # pipe.controlnet = torch.compile(pipe.controlnet)

    # global pipe_img2img
    # pipe_img2img = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    #     MODEL_ID,
    #     controlnet=controlnet,
    #     cache_dir=MODEL_CACHE,
    #     local_files_only=True,
    #     safety_checker=None,
    #     # torch_dtype=torch.float16
    # ).to(device)

    global compel
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

@torch.inference_mode()
def predict(
    image_url: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    controlnet_scale: float,
    control_guidance_start: float,
    control_guidance_end: float,
    num_outputs: int,
    num_inference_steps: int,
    guidance_scale: float,
    scheduler: str,
    karras: bool,
    seed: int,
    guess_mode: bool,
    img2img: Dict,
    use_compel: bool,
    lora_a1111_url: Optional[str] = None,
    lora_diffusers_url: Optional[str] = None,
    lora_scale: float = None,
):
    """Run a single prediction on the model"""
    if seed is None or seed == -1:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    pipe.scheduler.config.use_karras_sigmas = karras
    pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

    generator = torch.Generator(device).manual_seed(seed)

    image = load_image(image_url)
    image, width, height = resize(image)
    np_image = np.array(image)

    # get canny image
    canny_image = cv2.Canny(np_image, 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    if lora_diffusers_url is not None:
        lora_path = load_lora_a1111_url(lora_diffusers_url)
        print(f"Loading LoRA Diffusers weights from {lora_diffusers_url}")
        pipe.unet.load_attn_procs(lora_path)

    if lora_a1111_url is not None:
        lora_path = load_lora_a1111_url(lora_a1111_url)
        print(f"Loading LoRA A1111 weights from {lora_a1111_url}")
        pipe.unload_lora_weights()
        pipe.load_lora_weights(lora_path)

    cross_attention_kwargs = {"scale": lora_scale} if lora_scale is not None else None

    if lora_scale is not None:
        pipe._lora_scale = lora_scale

    conditioning = None
    conditioning_negative = None
    if use_compel:
        # why no_grad: https://github.com/damian0815/compel/issues/24#issuecomment-1620786740
        with torch.no_grad():
            conditioning = compel.build_conditioning_tensor(prompt)
            conditioning_negative = compel.build_conditioning_tensor(negative_prompt) if negative_prompt is not None else None

    # if device == "cuda":
    #     pipe.enable_xformers_memory_efficient_attention()

    output = None
    if img2img:
        print("Running img2img")
        output = pipe_img2img(
            image=image,
            strength=img2img['strength'],
            control_image=canny_image,
            prompt=prompt if conditioning is None else None,
            negative_prompt=negative_prompt if conditioning_negative is None else None,
            prompt_embeds=conditioning if conditioning is not None else None,
            negative_prompt_embeds=conditioning_negative if conditioning_negative is not None else None,
            width=width,
            height=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=controlnet_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            generator=generator,
            guess_mode=guess_mode,
            cross_attention_kwargs=cross_attention_kwargs,
        )
    else:
        print("Running canny")
        output = pipe(
            image=canny_image,
            prompt=prompt if conditioning is None else None,
            negative_prompt=negative_prompt if conditioning_negative is None else None,
            prompt_embeds=conditioning if conditioning is not None else None,
            negative_prompt_embeds=conditioning_negative if conditioning_negative is not None else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=controlnet_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            generator=generator,
            guess_mode=guess_mode,
            cross_attention_kwargs=cross_attention_kwargs,
        )

    # if device == "cuda":
    #     pipe.disable_xformers_memory_efficient_attention()

    if lora_a1111_url is not None:
        pipe.load_lora_weights(lora_path)

    output_paths = []
    for i, sample in enumerate(output.images):
        output_path = f"/tmp/out-{i}.png"
        sample.save(output_path)
        # output_paths.append(Path(output_path))
        output_paths.append(output_path)

    return output_paths

def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPM": DPMSolverMultistepScheduler.from_config(config),
        "UNI": UniPCMultistepScheduler.from_config(config),
    }[name]

def resize(
    image: Image.Image,
) -> Image.Image:
    height = image.height - image.height % 8
    width = image.width - image.width % 8

    image = image.resize((width, height))

    return (image, width, height)

lora_by_url = {}

def load_lora_a1111_url(url):
    url_md5 = hashlib.md5(url.encode()).hexdigest()
    path = f"/tmp/{url_md5}.safetensors"

    if os.path.isfile(path):
        return path

    print(f"Downloading {url} to {path}")
    r = requests.get(url)
    with open(path, 'wb') as f:
        f.write(r.content)
    return path

def load_lora_diffusers_url(url):
    if url in lora_by_url:
        return lora_by_url[url]
    random_name = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    path = f"/tmp/{random_name}.bin"
    print(f"Downloading {url} to {path}")
    r = requests.get(url)
    with open(path, 'wb') as f:
        f.write(r.content)
    lora_by_url[url] = path
    return path
