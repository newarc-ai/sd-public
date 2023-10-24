import sys
import os
from typing import List
from cog import BasePredictor, Input, Path

sys.path.append(os.path.dirname(__file__))
import predict_controlnet

class Predictor(BasePredictor):
    def setup(self):
        predict_controlnet.setup()

    def predict(
        self,
        image_url: str = Input(
            description="Input image",
        ),
        prompt: str = Input(
            description="Input prompt",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image.",
        ),
        height: int = Input(
            description="Height of output image.",
        ),
        controlnet_scale: float = Input(),
        control_guidance_start: float = Input(default=0.0),
        control_guidance_end: float = Input(default=1.0),
        steps: int = Input(),
        guidance_scale: float = Input(),
        scheduler: str = Input(),
        karras: bool = Input(default=True),
        seed: int = Input(),
        guess_mode: bool = Input(default=False),
        lora_a1111_url: str = Input(default=None),
        lora_diffusers_url: str = Input(default=None),
        lora_scale: float = Input(default=None),
        compel: bool = Input(default=False),
    ) -> List[Path]:
        output = predict_controlnet.predict(
            image_url=image_url,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            controlnet_scale=controlnet_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            scheduler=scheduler,
            karras=karras,
            seed=seed,
            guess_mode=guess_mode,
            num_outputs=1,
            img2img=None,
            lora_a1111_url=lora_a1111_url,
            lora_diffusers_url=lora_diffusers_url,
            lora_scale=lora_scale,
            use_compel=compel,
        )

        return [Path(x) for x in output]
