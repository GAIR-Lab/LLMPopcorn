import gradio as gr
import torch
import json
import os
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import InferenceClient, hf_hub_download
from safetensors.torch import load_file
import spaces

# 1. 初始化 LLM 客户端 (使用 Hugging Face 免费的 Serverless API)
client = InferenceClient("meta-llama/Llama-3.3-70B-Instruct")

# 2. 初始化视频生成 Pipeline (AnimateDiff-Lightning)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

step = 4  # 4-step inference, fast and good quality
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism"

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear"
)

def generate_llm_content(query):
    system_prompt = "You are a talented video creator. Generate a response in JSON format with 'title', 'cover_prompt', and 'video_prompt' (3s)."
    user_prompt = f"User Query: {query}\n\nRequirements:\n- title: MAX 50 chars\n- cover_prompt: image description\n- video_prompt: 3s motion description\n\nReturn JSON ONLY."
    
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

@spaces.GPU(duration=60) # 申请 ZeroGPU A100 资源
def create_popcorn(query):
    # Step 1: LLM 生成内容
    content = generate_llm_content(query)
    title = content.get("title", "Untitled Video")
    video_prompt = content.get("video_prompt", query)
    
    # Step 2: 生成视频
    print(f"Generating video for: {video_prompt}")
    output = pipe(prompt=video_prompt, guidance_scale=1.0, num_inference_steps=4, num_frames=16)
    
    # 保存为 GIF (Gradio 支持展示)
    video_path = "output_video.gif"
    export_to_gif(output.frames[0], video_path)
    
    return title, content.get("cover_prompt"), video_prompt, video_path

# 3. 构建 Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍿 LLMPopcorn Demo")
    gr.Markdown("Input a topic, and LLMPopcorn will generate the **Title**, **Prompts**, and a **3-second AI Video**.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Enter your video idea", placeholder="e.g., A futuristic city with flying cars")
            btn = gr.Button("Generate Popcorn!", variant="primary")
        
        with gr.Column():
            output_title = gr.Textbox(label="Generated Title")
            output_video = gr.Image(label="Generated 3s Video (GIF)")
            
    with gr.Accordion("Prompt Details", open=False):
        output_cover_prompt = gr.Textbox(label="Cover Prompt")
        output_video_prompt = gr.Textbox(label="Video Prompt")

    btn.click(
        fn=create_popcorn,
        inputs=[input_text],
        outputs=[output_title, output_cover_prompt, output_video_prompt, output_video]
    )

if __name__ == "__main__":
    demo.launch()
