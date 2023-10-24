import os
import shutil
import sys

import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)

# append project directory to path so predict.py can be imported
# sys.path.append('src')
#
# from predict import MODEL_CACHE, MODEL_ID
MODEL_ID = "SdValar/deliberate2"
MODEL_CACHE = "diffusers-cache"

# if os.path.exists(MODEL_CACHE):
#     shutil.rmtree(MODEL_CACHE)
os.makedirs(MODEL_CACHE, exist_ok=True)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16,
    cache_dir=MODEL_CACHE,
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    MODEL_ID,
    controlnet=controlnet,
    cache_dir=MODEL_CACHE,
    torch_dtype=torch.float16
)
