import os
import pandas as pd
import torch
import numpy as np
import random

from diffusers import StableDiffusionPipeline
from diffusers.utils import export_to_video

# Specify the GPU to use (adjust as needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed
set_seed(42)

def generate_image(pipeline, prompt: str, output_path: str):
    """
    Generate an image using the Stable Diffusion model and save it
    """
    with torch.autocast("cuda"):
        image = pipeline(prompt).images[0]
    image.save(output_path)

import torch
from diffusers.utils import export_to_video  # Ensure these methods are correctly imported

def generate_video(pipeline, pipeline_type: str, prompt: str, output_path: str, **kwargs):
    """
    Generate a video using different video generation pipelines and save as mp4 or gif

    Parameters:
      pipeline: Loaded video generation pipeline
      pipeline_type: Type of video model, options are "cogvideo", "ltx", "hunyuan", "animatediff"
      prompt: Text description
      output_path: Output video path (animatediff defaults to gif, others to mp4)
      kwargs: Hyperparameter settings, e.g., width, height, num_frames, num_inference_steps, fps, guidance_scale, etc.
    """
    if pipeline_type == "cogvideo":
        # Example call for CogVideoX (some hyperparameters may only apply to this pipeline)
        video = pipeline(
            prompt=prompt,
            num_videos_per_prompt=kwargs.get("num_videos_per_prompt", 1),
            num_inference_steps=kwargs.get("num_inference_steps", 50),
            num_frames=kwargs.get("num_frames", 49),
            guidance_scale=kwargs.get("guidance_scale", 6),
            generator=kwargs.get("generator", torch.Generator(device="cuda").manual_seed(42))
        ).frames[0]
        export_to_video(video, output_path, fps=kwargs.get("fps", 8))
    elif pipeline_type == "ltx":
        # Example call for LTXPipeline
        video = pipeline(
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"),
            width=kwargs.get("width", 704),
            height=kwargs.get("height", 480),
            num_frames=kwargs.get("num_frames", 161),
            num_inference_steps=kwargs.get("num_inference_steps", 50),
        ).frames[0]
        export_to_video(video, output_path, fps=kwargs.get("fps", 15))
    elif pipeline_type == "hunyuan":
        # Example call for HunyuanVideoPipeline
        video = pipeline(
            prompt=prompt,
            width=kwargs.get("width", 512),
            height=kwargs.get("height", 320),
            num_frames=kwargs.get("num_frames", 61),
            num_inference_steps=kwargs.get("num_inference_steps", 30),
        ).frames[0]
        export_to_video(video, output_path, fps=kwargs.get("fps", 15))
    elif pipeline_type == "animatediff":
        # Example call for AnimateDiff-Lightning (defaults to generating gif)
        video = pipeline(
            prompt=prompt,
            guidance_scale=kwargs.get("guidance_scale", 1.0),
            num_inference_steps=kwargs.get("num_inference_steps", 4)  # Default step is 4, options are 1,2,4,8
        ).frames[0]
        export_to_video(video, output_path)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

def load_video_pipeline(pipeline_type: str):
    """
    Load the corresponding video generation model based on pipeline_type

    Parameters:
      pipeline_type: Options are "cogvideo", "ltx", "hunyuan", "animatediff"
    Returns:
      Loaded and initialized video generation pipeline
    """
    if pipeline_type == "cogvideo":
        from diffusers import CogVideoXPipeline
        print("Loading video generation model (CogVideoX-5b)...")
        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-5b",
            torch_dtype=torch.bfloat16
        )
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.to("cuda")
        return pipe
    elif pipeline_type == "ltx":
        from diffusers import LTXPipeline
        print("Loading video generation model (LTX-Video)...")
        pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        return pipe
    elif pipeline_type == "hunyuan":
        from diffusers import BitsAndBytesConfig, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
        from diffusers.hooks import apply_layerwise_casting
        from transformers import LlamaModel
        print("Loading video generation model (HunyuanVideo)...")
        model_id = "hunyuanvideo-community/HunyuanVideo"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        text_encoder = LlamaModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
        apply_layerwise_casting(text_encoder, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float16)
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.float16
        )
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()
        return pipe
    elif pipeline_type == "animatediff":
        from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        print("Loading video generation model (AnimateDiff-Lightning)...")
        device = "cuda"
        dtype = torch.float16
        step = 4  # Options: [1,2,4,8], default is 4
        repo = "ByteDance/AnimateDiff-Lightning"
        ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
        base = "emilianJR/epiCRealism"  # Choose base model as preferred
        adapter = MotionAdapter().to(device, dtype)
        # Download and load weights
        adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
        pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear"
        )
        return pipe
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

def main():
    # ============ 1. Load/Initialize Models ============
    # (1) Image generation model: Stable Diffusion
    print("Loading image generation model (Stable Diffusion)...")
    pipe_image = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe_image.to("cuda")
    # Enable xformers acceleration if needed
    # pipe_image.enable_xformers_memory_efficient_attention()

    # (2) Video generation model: Choose "cogvideo", "ltx", or "hunyuan"
    video_pipeline_type = "ltx"  # Change here to select other models: "ltx" or "hunyuan" animatediff

    # ============ 2. Define Task List ============
    tasks1 = [
        {
            "csv_file": "output_prompt_rag_more/prompt_ai_concrete_rag_10_testset.csv",
            "image_dir": "output_ai_covers_concrete_rag_10_testset",
            "video_dir": "output_ai_videos_concrete_rag_10_testset_ltx"
        },
        {
            "csv_file": "output_prompt_rag_more/prompt_ai_abstract_rag_10_testset.csv",
            "image_dir": "output_ai_covers_abstract_rag_10_testset",
            "video_dir": "output_ai_videos_abstract_rag_10_testset_ltx"
        }

    ]

    
    # Only the first task is used in the example
    #tasks = [tasks[-4],tasks[-2]]
    #tasks=tasks_ablation_abstract_5b+tasks_ablation_concrete_5b
    #tasks= tasks_ablation_concrete2
    tasks = tasks1
    pipe_video = load_video_pipeline(video_pipeline_type)

    # ============ 3. Iterate over CSV files to generate images and videos ============
    for task in tasks:
        csv_file = task["csv_file"]
        image_dir = task["image_dir"]
        video_dir = task["video_dir"]
        os.makedirs(image_dir, exist_ok=True)
        print(f"Ensuring directory exists: {image_dir}")
        os.makedirs(video_dir, exist_ok=True)
        print(f"Ensuring directory exists: {video_dir}")

        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} not found, please check the path.")
            continue

        df = pd.read_csv(csv_file)
        for idx, row in df.iterrows():
            user_prompt = str(row["user prompt"])
            title = str(row["title"])
            cover_prompt = str(row["cover prompt"])
            video_prompt = str(row["video prompt"])

            # Generate filenames
            image_filename = os.path.join(image_dir, f"{user_prompt}.png")
            video_filename = os.path.join(video_dir, f"{user_prompt}.mp4")

            print("-" * 50)
            print(f"[CSV: {csv_file}] - [{idx}] Starting generation: {user_prompt}")
            print(f"Title: {title}")
            print(f"Cover Prompt: {cover_prompt}")
            print(f"Video Prompt: {video_prompt}")

            if os.path.exists(image_filename) and os.path.exists(video_filename):
                print(f"File already exists, skipping generation: {video_filename}")
                continue

            # 4. Generate image
            try:
                generate_image(pipe_image, cover_prompt, image_filename)
                print(f"Image saved to {image_filename}")
            except Exception as e:
                print(f"Image generation failed: {e}")

            # 5. Generate video (customize hyperparameters by passing additional arguments)
            try:
                generate_video(
                    pipe_video,
                    pipeline_type=video_pipeline_type,
                    prompt=video_prompt,
                    output_path=video_filename
                    # To modify hyperparameters, pass them here, e.g.:
                    # num_inference_steps=60, num_frames=50, fps=10, width=640, height=360, guidance_scale=7, ...
                )
                print(f"Video saved to {video_filename}")
            except Exception as e:
                print(f"Video generation failed: {e}")

    print("All generation tasks completed!")

if __name__ == "__main__":
    main()
