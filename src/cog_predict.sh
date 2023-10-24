#!/bin/env bash

cog predict \
-i prompt="stylish sneaker, (sky background)+, lineart" \
-i image_url=https://www.newarc.ai/s3/all/SJoAMyWT-AlbvBKKfRRfX.jpg \
-i negative_prompt='' \
-i width=751 \
-i height=521 \
-i controlnet_scale=0.7 \
-i steps=20 \
-i guidance_scale=9 \
-i scheduler=DPM \
-i karass=true \
-i seed=1337 \
-i lora_a1111_url=https://civitai.com/api/download/models/62833 \
-i lora_scale=-2 \
-i compel=true

