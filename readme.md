---
license: mit
tags:
- text-to-video
- prompt-engineering
- video-generation
- llm
- rag
- research
datasets:
- junchenfu/llmpopcorn_prompts
- junchenfu/microlens_rag
pipeline_tag: text-generation
---

# LLMPopcorn (ICASSP'26)

![Micro-video Generation](https://img.shields.io/badge/Task-Micro--video_Generation-red)
![Prompt Engineering](https://img.shields.io/badge/Task-Prompt_Engineering-red)
![RAG](https://img.shields.io/badge/Task-RAG-red)
![LLM](https://img.shields.io/badge/Task-LLM-red)
<a href="https://arxiv.org/abs/2502.12945" alt="arXiv"><img src="https://img.shields.io/badge/arXiv-2502.12945-FAA41F.svg?style=flat" /></a>
<a href="https://huggingface.co/spaces/junchenfu/llmpopcorn-demo" alt="Demo"><img src="https://img.shields.io/badge/🤗_Demo-Spaces-blue.svg?style=flat" /></a>
<a href="https://huggingface.co/datasets/junchenfu/llmpopcorn_prompts" alt="Dataset"><img src="https://img.shields.io/badge/🤗_Dataset-Prompts-green.svg?style=flat" /></a>
<a href="https://huggingface.co/datasets/junchenfu/microlens_rag" alt="RAG Dataset"><img src="https://img.shields.io/badge/🤗_Dataset-MicroLens_RAG-green.svg?style=flat" /></a>
<a href="https://github.com/GAIR-Lab/LLMPopcorn" alt="GitHub"><img src="https://img.shields.io/badge/GitHub-LLMPopcorn-black.svg?style=flat&logo=github" /></a>

**LLMPopcorn** is a research framework for generating popular short video titles, cover image prompts, and 3-second video prompts using Large Language Models (LLMs). It supports both a **Basic** direct-generation mode and a **PE (Prompt Enhancement)** mode that uses Retrieval-Augmented Generation (RAG) with Chain-of-Thought reasoning over the MicroLens dataset.

> **Paper:** [LLMPopcorn: Exploring LLMs as Assistants for Popular Micro-video Generation](https://arxiv.org/abs/2502.12945) (ICASSP 2026)

---

## Prerequisites

### Install Required Python Packages

```bash
pip install torch transformers diffusers tqdm numpy pandas sentence-transformers faiss-cpu openai huggingface_hub safetensors accelerate datasets
```

> **Note:** `bitsandbytes` (for 4-bit quantization) is Linux-only. Install separately on GPU servers:
> ```bash
> pip install bitsandbytes
> ```

### Download the MicroLens Dataset

Download the following files from the [MicroLens dataset](https://github.com/westlake-repl/MicroLens) and place them in the `Microlens/` folder.

> **Alternatively**, use the pre-processed HuggingFace version — see [RAG Reference Dataset](#rag-reference-dataset-microlens) below.

| File | Description |
|------|-------------|
| `MicroLens-100k_likes_and_views.txt` | Video engagement stats (tab-separated) |
| `MicroLens-100k_title_en.csv` | Cover image descriptions (comma-separated) |
| `Microlens100K_captions_en.csv` | Video captions in English (tab-separated) |
| `MicroLens-100k_comment_en.txt` | User comments (tab-separated) |
| `tags_to_summary.csv` | Video category tags (comma-separated) |

Directory structure:
```
LLMPopcorn/
├── Microlens/
│   ├── MicroLens-100k_likes_and_views.txt
│   ├── MicroLens-100k_title_en.csv
│   ├── Microlens100K_captions_en.csv
│   ├── MicroLens-100k_comment_en.txt
│   └── tags_to_summary.csv
├── PE.py
├── pipline.py
└── ...
```

---

## Step 1: Generate Video Titles and Prompts

**Basic mode** — direct LLM generation:
```bash
python LLMPopcorn.py
```

**PE mode** — RAG + Chain-of-Thought enhanced generation:
```bash
python PE.py
```

---

## Step 2: Generate AI Videos

```bash
python generating_images_videos_three.py
```

---

## Step 3: Evaluate

Following the instructions in the [MMRA repository](https://github.com/westlake-repl/MicroLens), you can evaluate the generated videos.

---

## Interactive Demo

Try the live demo on Hugging Face Spaces:

**[🍿 LLMPopcorn Demo](https://huggingface.co/spaces/junchenfu/llmpopcorn-demo)**

The demo lets you compare:
- **Basic LLMPopcorn** — direct LLM generation from your query
- **PE (Prompt Enhancement)** — RAG + CoT using MicroLens reference videos

To run the demo locally:
```bash
pip install gradio diffusers sentence-transformers faiss-cpu datasets spaces safetensors
python app.py
```

---

## Datasets on Hugging Face

### Prompts Dataset

```python
from datasets import load_dataset

dataset = load_dataset("junchenfu/llmpopcorn_prompts")
for item in dataset["train"]:
    print(f"Type: {item['type']}, Prompt: {item['prompt']}")
```

Contains **200** abstract and concrete video prompts used as input queries.

### RAG Reference Dataset: MicroLens

For the RAG-enhanced pipeline (`PE.py` + `pipline.py`), a pre-processed MicroLens dataset is available so you don't need to download raw files:

**[junchenfu/microlens_rag](https://huggingface.co/datasets/junchenfu/microlens_rag)**

Contains **19,560** video entries across **22 categories**:

| Column | Description |
|--------|-------------|
| `video_id` | Unique video identifier |
| `title_en` | Cover image description (used as title) |
| `cover_desc` | Cover image description |
| `caption_en` | Full video caption in English |
| `partition` | Video category (e.g., Anime, Game, Delicacy) |
| `likes` | Number of likes |
| `views` | Number of views |
| `comment_count` | Number of comments (popularity signal) |

```python
from datasets import load_dataset

df = load_dataset("junchenfu/microlens_rag", split="train").to_pandas()
print(f"Total: {len(df)} videos, {df['partition'].nunique()} categories")
```
