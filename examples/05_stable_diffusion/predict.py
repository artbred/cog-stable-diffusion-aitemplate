import os, sys, torch

from PIL import Image
from pathlib import Path
from typing import List
from os import path

from cog import BasePredictor, Input, Path
from diffusers import EulerDiscreteScheduler, PNDMScheduler


sys.path.append("../")
from src.pipeline_stable_diffusion_img2img_ait import StableDiffusionImg2ImgAITPipeline
from src.pipeline_stable_diffusion_ait import StableDiffusionAITPipeline

LOCAL_DIR = "./tmp/diffusers-pipeline/polymorphia/ft-every-sd1-archmain-v2"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.txt2img_pipe = StableDiffusionAITPipeline.from_pretrained(
            LOCAL_DIR,
            scheduler=EulerDiscreteScheduler.from_pretrained(
                LOCAL_DIR, subfolder="scheduler"
            ),
            torch_dtype=torch.float16,
        ).to("cuda")

        self.img2img_pipe = StableDiffusionImg2ImgAITPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=PNDMScheduler.from_config(self.txt2img_pipe.scheduler.config),
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")
        

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[512],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[512],
            default=512,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=6,
            le=6,
            default=6,
        ),
        init_image: Path = Input(
            description="Inital image to generate variations of. Will be resized to the specified width and height",
            default=None,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        extra_kwargs = {}
        if init_image:
            pipe = self.img2img_pipe
            extra_kwargs = {
                "init_image": Image.open(init_image).convert("RGB").resize((height, width)),
                "strength": prompt_strength,
            }
        else:
            pipe = self.txt2img_pipe

        generator = torch.Generator("cuda").manual_seed(seed)
        output = pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            **extra_kwargs,
        )

        samples = [
            output.images[i]
            for i, nsfw_flag in enumerate(output.nsfw_content_detected)
            if not nsfw_flag
        ]

        if len(samples) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        if num_outputs > len(samples):
            print(
                f"NSFW content detected in {num_outputs - len(samples)} outputs, showing the rest {len(samples)} images..."
            )

        output_paths = []
        for i, sample in enumerate(samples):
            output_path = f"/tmp/out-{i}.jpg"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths


