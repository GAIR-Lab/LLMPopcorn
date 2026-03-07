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

# LLMPopcorn Usage Instructions

Welcome to LLMPopcorn! This guide will help you generate video titles and prompts, as well as create AI-generated videos based on those prompts.

## Prerequisites

### Install Required Python Packages

Before running the scripts, ensure that you have installed the necessary Python packages. You can do this by executing the following command:

```bash
pip install torch transformers diffusers tqdm numpy pandas sentence-transformers faiss-cpu openai huggingface_hub safetensors
```

**Download the MicroLens Dataset**:  
Download the following files from the [MicroLens dataset](https://github.com/westlake-repl/MicroLens) and place them in the `Microlens/` folder:

| File | Description |
|------|-------------|
| `MicroLens-100k_likes_and_views.txt` | Video engagement stats (tab-separated) |
| `MicroLens-100k_title_en.csv` | Cover image descriptions (comma-separated) |
| `Microlens100K_captions_en.csv` | Video captions in English (tab-separated) |
| `MicroLens-100k_comment_en.txt` | User comments (tab-separated) |
| `tags_to_summary.csv` | Video category tags (comma-separated) |

Your directory structure should look like:
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

## Step 1: Generate Video Titles and Prompts

To generate video titles and prompts, run the `LLMPopcorn.py` script:
```bash
python LLMPopcorn.py
```

To enhance LLMPopcorn, execute the `PE.py` script:
```bash
python PE.py
```

## Step 2: Generate AI Videos

To create AI-generated videos, execute the `generating_images_videos_three.py` script:
```bash
python generating_images_videos_three.py
```

## Step 3: Clone the Evaluation Code

Then, following the instructions in the MMRA repository, you can evaluate the generated videos.

## Tutorial: Using the Prompts Dataset

You can easily download and use the structured prompts directly from Hugging Face:

### 1. Install `datasets`
```bash
pip install datasets
```

### 2. Load the Dataset in Python
```python
from datasets import load_dataset

# Load the LLMPopcorn prompts
dataset = load_dataset("junchenfu/llmpopcorn_prompts")

# Access the data (abstract or concrete)
for item in dataset["train"]:
    print(f"Type: {item['type']}, Prompt: {item['prompt']}")
```

This dataset contains both abstract and concrete prompts, which you can use as input for the video generation scripts in Step 2.

## RAG Reference Dataset: MicroLens

For the RAG-enhanced pipeline (`PE.py` + `pipline.py`), we provide a pre-processed version of the MicroLens dataset on Hugging Face so you don't need to download and process the raw files manually.

The dataset is available at: [**junchenfu/microlens_rag**](https://huggingface.co/datasets/junchenfu/microlens_rag)

It contains **19,560** video entries across **22 categories** with the following fields:

| Column | Description |
|--------|-------------|
| `video_id` | Unique video identifier |
| `title_en` | Cover image description (used as title) |
| `cover_desc` | Cover image description |
| `caption_en` | Full video caption in English |
| `partition` | Video category (e.g., Anime, Game, Delicacy) |
| `likes` | Number of likes |
| `views` | Number of views |
| `comment_count` | Number of comments (used as popularity signal) |

### Load the RAG Dataset in Python

```python
from datasets import load_dataset

rag_dataset = load_dataset("junchenfu/microlens_rag")

# Access as a pandas DataFrame
df = rag_dataset["train"].to_pandas()
print(df.head())
print(f"Total: {len(df)} videos, {df['partition'].nunique()} categories")
```
