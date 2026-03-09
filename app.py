# spaces MUST be imported before torch / any CUDA package (ZeroGPU requirement)
import spaces

import gradio as gr
import torch
import json
import os
import re
import threading
import numpy as np
import pandas as pd
import faiss
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

HF_TOKEN = os.environ.get("HF_TOKEN")

# --- 1. LLM: Qwen2.5-7B-Instruct loaded on GPU via ZeroGPU
_LLM_ID = "Qwen/Qwen2.5-7B-Instruct"
_llm_pipe = None
_llm_lock = threading.Lock()

# --- 2. Lazy globals with threading locks ---
_pipe = None
_rag_df = None
_embed_model = None
_unique_partitions = None
_partition_embeddings = None
_pipe_lock = threading.Lock()
_rag_lock = threading.Lock()

def get_pipe():
    """Lazy-load Wan2.1-T2V-1.3B inside a ZeroGPU context."""
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                print("Loading Wan2.1-T2V-1.3B pipeline...")
                _pipe = WanPipeline.from_pretrained(
                    "Wan-AI/Wan2.1-T2V-1.3B",
                    torch_dtype=torch.bfloat16,
                )
                _pipe.enable_model_cpu_offload()
                _pipe.vae.enable_tiling()
                print("Wan2.1-T2V pipeline ready.")
    return _pipe

def get_rag():
    global _rag_df, _embed_model, _unique_partitions, _partition_embeddings
    if _rag_df is None:
        with _rag_lock:
            if _rag_df is None:
                print("Loading MicroLens RAG dataset (first use)...")
                _rag_df = load_dataset("junchenfu/microlens_rag", split="train").to_pandas()
                _rag_df["comment_count"] = _rag_df["comment_count"].fillna(0)
                _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
                _unique_partitions = _rag_df["partition"].unique().tolist()
                _partition_embeddings = _embed_model.encode(_unique_partitions)
                print(f"RAG ready: {len(_rag_df)} videos, {len(_unique_partitions)} categories.")
    return _rag_df, _embed_model, _unique_partitions, _partition_embeddings

# Pre-warm in background so the first user request is faster
def _preload():
    try:
        get_rag()
    except Exception as e:
        print(f"Background preload warning: {e}")

threading.Thread(target=_preload, daemon=True).start()

def _extract_json(text):
    """Extract the first JSON object from a string, with regex fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"No JSON found in response: {text[:200]}")

@spaces.GPU(duration=60)
def _llm_generate(messages: list, max_new_tokens: int = 500) -> str:
    """Run Qwen2.5-7B-Instruct on ZeroGPU. Model is lazy-loaded on first call."""
    global _llm_pipe
    if _llm_pipe is None:
        with _llm_lock:
            if _llm_pipe is None:
                print(f"Loading LLM {_LLM_ID} ...")
                _llm_pipe = hf_pipeline(
                    "text-generation",
                    model=_LLM_ID,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                print("LLM ready.")
    out = _llm_pipe(messages, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
    generated = out[0]["generated_text"]
    if isinstance(generated, list):
        return generated[-1].get("content", "")
    return str(generated)

# --- 3. Basic LLMPopcorn ---
def generate_basic(query):
    messages = [
        {"role": "system", "content": (
            "You are a talented video creator. "
            "Respond ONLY with a JSON object containing keys: title (max 50 chars), cover_prompt, video_prompt (3s clip)."
        )},
        {"role": "user", "content": (
            f"User Query: {query}\n\n"
            "Return JSON ONLY, no extra text, no markdown fences."
        )},
    ]
    raw = _llm_generate(messages, max_new_tokens=500)
    return _extract_json(raw)

# --- 4. PE: RAG + CoT ---
def build_rag_context(user_prompt, selected_videos_num=10, num_tags=1, ratio=0.1):
    rag_df, embed_model, unique_partitions, partition_embeddings = get_rag()
    prompt_emb = embed_model.encode([user_prompt])[0]
    sims = [
        np.dot(prompt_emb, pe) / (np.linalg.norm(prompt_emb) * np.linalg.norm(pe))
        for pe in partition_embeddings
    ]
    top_partitions = [unique_partitions[i] for i in np.argsort(sims)[::-1][:num_tags]]

    filtered = rag_df[rag_df["partition"].isin(top_partitions)].copy()
    filtered = filtered.sort_values("comment_count", ascending=False)

    n_neg = int(len(filtered) * ratio)
    n_pos = len(filtered) - n_neg
    positive_videos = filtered.head(n_pos).drop_duplicates(subset=["video_id"])
    negative_videos = filtered.iloc[n_pos:].tail(n_neg).drop_duplicates(subset=["video_id"])
    combined = pd.concat([positive_videos, negative_videos]).drop_duplicates(subset=["video_id"])

    texts = (combined["title_en"] + " " + combined["cover_desc"] + " " + combined["caption_en"]).tolist()
    combined_embs = embed_model.encode(texts).astype("float32")
    index = faiss.IndexFlatL2(combined_embs.shape[1])
    index.add(combined_embs)
    query_emb = embed_model.encode([user_prompt]).astype("float32")
    _, I = index.search(query_emb, len(combined))
    retrieved = combined.iloc[I[0]]

    n_final_neg = int(selected_videos_num * ratio)
    n_final_pos = selected_videos_num - n_final_neg
    pos_ids = set(positive_videos["video_id"].tolist())
    neg_ids = set(negative_videos["video_id"].tolist())
    final_pos = retrieved[retrieved["video_id"].isin(pos_ids)].head(n_final_pos)
    final_neg = retrieved[retrieved["video_id"].isin(neg_ids)].head(n_final_neg)

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
3. Verify the idea matches the user query and popular trends.
4. Generate the final output.

Return JSON ONLY with keys: title (max 50 chars), cover_prompt, video_prompt (3s).
"""
    pe_messages = [
        {"role": "system", "content": "You are a talented video creator. Return JSON only."},
        {"role": "user", "content": cot_prompt},
    ]
    raw = _llm_generate(pe_messages, max_new_tokens=800)
    result = _extract_json(raw)
    result["_matched_tag"] = matched_tag
    return result

# --- 5. Video generation (Wan2.1-T2V-1.3B inside ZeroGPU context) ---
@spaces.GPU(duration=180)
def run_video_generation(video_prompt):
    pipe = get_pipe()
    output = pipe(
        prompt=video_prompt,
        negative_prompt="blurry, ugly, bad quality, distorted, low resolution",
        width=512,
        height=288,
        num_frames=33,
        num_inference_steps=24,
        guidance_scale=5.0,
    )
    mp4_path = "output_video.mp4"
    export_to_video(output.frames[0], mp4_path, fps=16)
    return mp4_path

# --- 6. Gradio entrypoints ---
def run_basic(query):
    content = generate_basic(query)
    title = content.get("title", "Untitled")
    cover = content.get("cover_prompt", "")
    video_prompt = content.get("video_prompt", query)
    mp4 = run_video_generation(video_prompt)
    return title, cover, video_prompt, mp4

def run_pe(query, vid_num):
    content = generate_pe(query, int(vid_num))
    title = content.get("title", "Untitled")
    cover = content.get("cover_prompt", "")
    video_prompt = content.get("video_prompt", query)
    matched_tag = content.get("_matched_tag", "N/A")
    mp4 = run_video_generation(video_prompt)
    return title, cover, video_prompt, f"Matched category: **{matched_tag}**", mp4

# --- 7. Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Popcorn LLMPopcorn Demo")
    gr.Markdown(
        "Compare **Basic LLMPopcorn** (direct LLM generation) vs "
        "**PE - Prompt Enhancement** (RAG + Chain-of-Thought using MicroLens reference videos)."
    )

    with gr.Tabs():
        with gr.Tab("Basic LLMPopcorn"):
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
                    basic_video = gr.Video(label="Generated 3s Video")
            with gr.Accordion("Prompt Details", open=False):
                basic_cover = gr.Textbox(label="Cover Prompt")
                basic_vprompt = gr.Textbox(label="Video Prompt")

            basic_btn.click(
                fn=run_basic,
                inputs=[basic_input],
                outputs=[basic_title, basic_cover, basic_vprompt, basic_video],
            )

        with gr.Tab("PE - Prompt Enhancement (RAG + CoT)"):
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
                    pe_video = gr.Video(label="Generated 3s Video")
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
