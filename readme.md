# LLMPopcorn Usage Instructions

Welcome to LLMPopcorn! This guide will help you generate video titles and prompts, and create AI-generated videos based on those prompts.

## Prerequisites

### Install Required Python Packages

Before running the scripts, ensure you have the necessary Python packages installed. You can install them using the following command:

```bash
pip install torch transformers diffusers tqdm numpy pandas sentence-transformers faiss-cpu openai huggingface_hub safetensors
```

**Download the Dataset**:  
Download the dataset from [this link](https://github.com/westlake-repl/MicroLens) and place it in the `Microlens` folder for evaluation.

## Step 1: Generate Video Titles and Prompts

To generate video titles and prompts, execute the `LLMPopcorn.py` script:
```bash
python LLMPopcorn.py
```

For enhancing LLMPopcorn, execute the `PE.py` script:
```bash
python PE.py
```

## Step 2: Generate AI Videos

To create AI-generated videos, run the `generating_images_videos_three.py` script:
```bash
python generating_images_videos_three.py
```

## Step 3: Clone the Evaluation Code 
Use the following command to clone the evaluation code repository:
```bash
git clone https://github.com/ICDM-UESTC/MMRA.git
```

