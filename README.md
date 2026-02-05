# LLMPopcorn (ICASSP'26)

Welcome to LLMPopcorn! This guide will help you generate video titles and prompts, as well as create AI-generated videos based on those prompts.

## Prerequisites

### Install Required Python Packages

Before running the scripts, ensure that you have installed the necessary Python packages. You can do this by executing the following command:

```bash
pip install torch transformers diffusers tqdm numpy pandas sentence-transformers faiss-cpu openai huggingface_hub safetensors
```

**Download the Dataset**:  
Download the Microlens dataset and place it in the `Microlens` folder for use with `PE.py`.

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

Then, following the instructions in the [MMRA](https://github.com/ICDM-UESTC/MMRA) repository, you can evaluate the generated videos.
