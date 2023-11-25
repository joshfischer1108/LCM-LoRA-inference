
from sys import platform
import os
import random
from os import path
import torch
from contextlib import nullcontext
from diffusers import LCMScheduler, StableDiffusionPipeline
is_mac = platform == "darwin"

cache_path = path.join(path.dirname(path.abspath(__file__)), "models")

os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

def should_use_fp16():
    if is_mac:
        return True

    gpu_props = torch.cuda.get_device_properties("cuda")

    if gpu_props.major < 6:
        return False

    nvidia_16_series = ["1660", "1650", "1630"]

    for x in nvidia_16_series:
        if x in gpu_props.name:
            return False

    return True

def load(model_id="CompVis/stable-diffusion-v1-4"):


    if not is_mac:
        torch.backends.cuda.matmul.allow_tf32 = True

    use_fp16 = should_use_fp16()

    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

    if use_fp16:
        print('this would be a mac, low precision')
        pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, cache_dir=cache_path, safety_checker=None)

    else:
        print("this would be cuda, fine precision")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp32", torch_dtype=torch.float32, cache_dir=cache_path, safety_checker=None)


    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.load_lora_weights(lcm_lora_id)
    pipe.fuse_lora()

    device = "mps" if is_mac else "cuda"

    pipe.to(device=device)

    generator = torch.Generator()

    def infer_no_image(
            prompt,
            num_inference_steps=4,
            guidance_scale=1,
            strength=0.9,
            seed=random.randrange(0, 2**63)
    ):
        with torch.inference_mode():
            with torch.autocast("cuda") if device == "cuda" else nullcontext():
                return pipe(
                    prompt=prompt,
                    generator=generator.manual_seed(seed),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength
                ).images[0]

    return infer_no_image
