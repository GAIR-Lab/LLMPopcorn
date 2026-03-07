import gradio as gr
import torch
import json
import numpy as np
import pandas as pd
import faiss
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import InferenceClient, hf_hub_download
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import spaces

# ──────────────────────────────────────────
# 1. Shared: LLM client
# ──────────────────────────────────────────
client = InferenceClient("meta-llama/Llama-3.3-70B-Instruct")

# ──────────────────────────────────────────
# 2. Shared: Video generation pipeline (AnimateDiff-Lightning)
# ──────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

step = 4
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism"

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear"
)

# ──────────────────────────────────────────
# 3. PE only: Load MicroLens RAG dataset + SentenceTransformer
# ──────────────────────────────────────────
print("Loading MicroLens RAG dataset from Hugging Face...")
rag_df = load_dataset("junchenfu/microlens_rag", split="train").to_pandas()
rag_df['comment_count'] = rag_df['comment_count'].fillna(0)

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Pre-compute partition embeddings for fast retrieval
unique_partitions = rag_df['partition'].unique().tolist()
partition_embeddings = embed_model.encode(unique_partitions)
print(f"RAG dataset loaded: {len(rag_df)} videos, {len(unique_partitions)} categories.")

# ──────────────────────────────────────────
# 4. Basic LLMPopcorn: direct LLM generation
# ──────────────────────────────────────────
def generate_basic(query):
    system_prompt = (
        "You are a talented video creator. "
        "Generate a response in JSON format with 'title', 'cover_prompt', and 'video_prompt' (3s)."
    )
    user_prompt = (
        f"User Query: {query}\n\n"
        "Requirements:\n- title: MAX 50 chars\n"
        "- cover_prompt: image description\n"
        "- video_prompt: 3s motion description\n\n"
        "Return JSON ONLY."
    )
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

# ──────────────────────────────────────────
# 5. PE: RAG + CoT generation
# ──────────────────────────────────────────
def build_rag_context(user_prompt, selected_videos_num=10, num_tags=1, ratio=0.1):
    # Find top matching partition(s) by cosine similarity
    prompt_emb = embed_model.encode([user_prompt])[0]
    sims = [
        np.dot(prompt_emb, pe) / (np.linalg.norm(prompt_emb) * np.linalg.norm(pe))
        for pe in partition_embeddings
    ]
    top_partitions = [unique_partitions[i] for i in np.argsort(sims)[::-1][:num_tags]]

    filtered = rag_df[rag_df['partition'].isin(top_partitions)].copy()
    filtered = filtered.sort_values('comment_count', ascending=False)

    n_neg = int(len(filtered) * ratio)
    n_pos = len(filtered) - n_neg
    positive_videos = filtered.head(n_pos).drop_duplicates(subset=['video_id'])
    negative_videos = filtered.iloc[n_pos:].tail(n_neg).drop_duplicates(subset=['video_id'])
    combined = pd.concat([positive_videos, negative_videos]).drop_duplicates(subset=['video_id'])

    # FAISS retrieval within the filtered pool
    texts = (combined['title_en'] + " " + combined['cover_desc'] + " " + combined['caption_en']).tolist()
    combined_embs = embed_model.encode(texts).astype('float32')
    index = faiss.IndexFlatL2(combined_embs.shape[1])
    index.add(combined_embs)
    query_emb = embed_model.encode([user_prompt]).astype('float32')
    _, I = index.search(query_emb, len(combined))
    retrieved = combined.iloc[I[0]]

    n_final_neg = int(selected_videos_num * ratio)
    n_final_pos = selected_videos_num - n_final_neg
    pos_ids = set(positive_videos['video_id'].tolist())
    neg_ids = set(negative_videos['video_id'].tolist())
    final_pos = retrieved[retrieved['video_id'].isin(pos_ids)].head(n_final_pos)
    final_neg = retrieved[retrieved['video_id'].isin(neg_ids)].head(n_final_neg)

    pos_ctx = "\n".join([
        f"Reference Video {i+1} (Popular):\nTitle: {row['title_en']}\nDesc: {row['caption_en']}\nComments: {int(row['comment_count'])}"
        for i, (_, row) in enumerate(final_pos.iterrows())
    ])
    neg_ctx = "\n".join([
        f"Reference Video {i+1} (Unpopular):\nTitle: {row['title_en']}\nDesc: {row['caption_en']}\nComments: {int(row['comment_count'])}"
        for i, (_, row) in enumerate(final_neg.iterrows())
    ])
    return pos_ctx + "\n" + neg_ctx, top_partitions[0]

def generate_pe(query, vid_num=10):
    rag_context, matched_tag = build_rag_context(query, selected_videos_num=vid_num)

    cot_prompt = f"""You are a talented video creator. Think step-by-step using the reference videos below, then generate the most popular title, cover prompt, and 3-second video prompt.

User Query: {query}

Reference Videos (from category: {matched_tag}):
{rag_context}

Reasoning Chain:
1. Analyze what makes the popular videos successful and what makes unpopular ones fail.
2. Brainstorm original ideas inspired by (but not copying) the popular references.
3. Verify the idea matches the user's query and popular trends.
4. Generate the final output.

Return JSON ONLY with keys: title (max 50 chars), cover_prompt, video_prompt (3s).
"""
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a talented video creator. Return JSON only."},
            {"role": "user", "content": cot_prompt},
        ],
        max_tokens=800,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    result["_matched_tag"] = matched_tag
    return result

# ──────────────────────────────────────────
# 6. Shared: Video generation
# ──────────────────────────────────────────
@spaces.GPU(duration=60)
def run_video_generation(video_prompt):
    output = pipe(prompt=video_prompt, guidance_scale=1.0, num_inference_steps=4, num_frames=16)
    gif_path = "output_video.gif"
    export_to_gif(output.frames[0], gif_path)
    return gif_path

# ──────────────────────────────────────────
# 7. Combined entrypoints for Gradio
# ──────────────────────────────────────────
def run_basic(query):
    content = generate_basic(query)
    title = content.get("title", "Untitled")
    cover = content.get("cover_prompt", "")
    video_prompt = content.get("video_prompt", query)
    gif = run_video_generation(video_prompt)
    return title, cover, video_prompt, gif

def run_pe(query, vid_num):
    content = generate_pe(query, int(vid_num))
    title = content.get("title", "Untitled")
    cover = content.get("cover_prompt", "")
    video_prompt = content.get("video_prompt", query)
    matched_tag = content.get("_matched_tag", "N/A")
    gif = run_video_generation(video_prompt)
    return title, cover, video_prompt, f"Matched category: **{matched_tag}**", gif

# ──────────────────────────────────────────
# 8. Gradio UI
# ──────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍿 LLMPopcorn Demo")
    gr.Markdown(
        "Compare **Basic LLMPopcorn** (direct LLM generation) vs "
        "**PE — Prompt Enhancement** (RAG + Chain-of-Thought using MicroLens reference videos)."
    )

    with gr.Tabs():
        # ── Tab 1: Basic ──────────────────
        with gr.Tab("⚡ Basic LLMPopcorn"):
            gr.Markdown("### Direct LLM Generation\nThe LLM generates title and prompts directly from your query without any external reference.")
            with gr.Row():
                with gr.Column():
                    basic_input = gr.Textbox(
                        label="Enter your video idea",
                        placeholder="e.g., A futuristic city with flying cars"
                    )
                    basic_btn = gr.Button("Generate!", variant="primary")
                with gr.Column():
                    basic_title = gr.Textbox(label="Generated Title")
                    basic_video = gr.Image(label="Generated 3s Video (GIF)")
            with gr.Accordion("Prompt Details", open=False):
                basic_cover = gr.Textbox(label="Cover Prompt")
                basic_vprompt = gr.Textbox(label="Video Prompt")

            basic_btn.click(
                fn=run_basic,
                inputs=[basic_input],
                outputs=[basic_title, basic_cover, basic_vprompt, basic_video],
            )

        # ── Tab 2: PE ─────────────────────
        with gr.Tab("🚀 PE — Prompt Enhancement (RAG + CoT)"):
            gr.Markdown(
                "### RAG-Enhanced Generation\n"
                "Retrieves similar popular/unpopular reference videos from **MicroLens** "
                "and uses Chain-of-Thought reasoning to generate higher-quality prompts."
            )
            with gr.Row():
                with gr.Column():
                    pe_input = gr.Textbox(
                        label="Enter your video idea",
                        placeholder="e.g., A futuristic city with flying cars"
                    )
                    pe_vid_num = gr.Slider(
                        minimum=5, maximum=50, value=10, step=5,
                        label="Number of RAG reference videos"
                    )
                    pe_btn = gr.Button("Generate with PE!", variant="primary")
                with gr.Column():
                    pe_title = gr.Textbox(label="Generated Title")
                    pe_matched = gr.Markdown()
                    pe_video = gr.Image(label="Generated 3s Video (GIF)")
            with gr.Accordion("Prompt Details", open=False):
                pe_cover = gr.Textbox(label="Cover Prompt")
                pe_vprompt = gr.Textbox(label="Video Prompt")

            pe_btn.click(
                fn=run_pe,
                inputs=[pe_input, pe_vid_num],
                outputs=[pe_title, pe_cover, pe_vprompt, pe_matched, pe_video],
            )

if __name__ == "__main__":
    demo.launch()
