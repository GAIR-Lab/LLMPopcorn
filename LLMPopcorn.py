import os
import re
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set random seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Input file and output directory
input_file = "abstract_prompts.txt"
output_dir = "baseline_concrete_outputsf"
os.makedirs(output_dir, exist_ok=True)

# Model name (example)
LLAMA_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model_llama = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME, 
    device_map="auto", 
    torch_dtype=torch.bfloat16, 
    quantization_config=quantization_config
)

# Set up pipeline
llama_pipeline = pipeline(
    "text-generation",
    model=model_llama,
    tokenizer=tokenizer,
    max_new_tokens=5000,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True
)

# Define a function to generate a valid filename from a query
def sanitize_filename(filename: str) -> str:
    # Remove characters not suitable for filenames, truncate if too long
    filename = filename.strip()
    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
    # For safety, truncate filename if query is too long
    if len(filename) > 100:
        filename = filename[:100]
    return filename

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Process each line
for line in tqdm(lines):
    query = line.strip()
    if not query:
        continue

    # Prepare the LLM input prompt
    messages = [
        {
            "role": "system",
            "content": (
                "Now that you're a talented video creator with a wealth of ideas, you need to think from the user's perspective and after that generate the most popular video title, "
                "an AI-generated cover prompt, and a 3-second AI-generated video prompt."
            )
        },
        {
            "role": "user",
            "content": (
                f"Below is the user query:\n\n{query}\n\n"
                "Final Answer Requirements:\n"
                "- A single line for the final generated Title (MAX_length = 50).\n"
                "- A single paragraph for the Cover Prompt.\n"
                "- A single paragraph for the Video Prompt (3-second).\n\n"
                "Now, based on the above reasoning, generate the response in JSON format. Here is an example:\n"
                "{\n"
                '  "title": "Unveiling the Legacy of Ancient Rome: Rise, Glory, and Downfall.",\n'
                '  "cover_prompt": "Generate an image of a Roman Emperor standing proudly in front of the Colosseum, with a subtle sunset backdrop, highlighting the contrast between the ancient structure.",\n'
                '  "video_prompt": "Open with a 3-second aerial shot of the Roman Forum, showcasing the sprawling ancient ruins against a clear blue sky, before zooming in on a singular, imposing structure like the Colosseum."\n'
                "}\n"
                "Please provide your answer following this exact JSON template for the response."
            )
        }
    ]

    # Call the LLM for inference
    response = llama_pipeline(messages, num_return_sequences=1)
    final_output = response[0]["generated_text"]

    # Determine output file name and save
    output_filename = sanitize_filename(query) + ".txt"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(final_output[2]['content'])

    print(f"Processed query: {query} -> {output_path}")
