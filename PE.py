import os
from tqdm import tqdm
import time
# Input files and directories

typ = "abstract"

input_file = "abstract_prompts.txt"

#selected_videos_nums = [0,20,40,60,80,120,140,150] #,,80,120,140,150

selected_videos_nums = [10,50,100] #,,80,120,140,150
tags = [1]
for selected_videos_num in selected_videos_nums:
    for tag in tags:
        rag_prompt_dir = f"rag_cot_prompt_ai_{typ}_rag_{selected_videos_num}_tags_{tag}_testset/" # rag_cot_prompt_ai_abstract_rag_50_tags_2_testset
        output_dir = f"creation_rag_cot_prompt_ai_{typ}_rag_{selected_videos_num}_tags_{tag}_testset/"
        script_file = "pipline.py"
        # Create directories if they do not exist
        os.makedirs(rag_prompt_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        # Read UserPrompts.txt
        with open(input_file, "r", encoding="utf-8") as f:
            prompts = f.readlines()

        # # Process each prompt
        for i, prompt in enumerate(tqdm(prompts,desc="Generating Prompts")):
            prompt = prompt.strip()
            output_path = os.path.join(rag_prompt_dir, f"{prompt}.txt")
            if not prompt or os.path.exists(output_path):  # Skip empty lines
                continue

            safe_filename = f"{prompt[:].replace(' ', '_').replace('/', '_')}.txt"
            rag_prompt_file = os.path.join(rag_prompt_dir, safe_filename)

            # Generate RAG Prompt
            command = f"python {script_file} --USER_PROMPT \'{prompt}\' --OUTPUT_DIR {rag_prompt_dir} --MODE generate --MODEL llama --VID_NUM {selected_videos_num} --TAGS_NUM {tag}"
            result = os.system(command)  # Run the command
            if result == 0:
                print(f"Generated RAG prompt for: {prompt[:50]} -> Saved to {rag_prompt_file}")
            else:
                print(f"Error generating RAG prompt for: {prompt}")
                continue

# Process all RAG prompt files
# Run inference
        command = f"python {script_file} --INPUT_DIR {rag_prompt_dir} --OUTPUT_DIR {output_dir} --MODE infer --MODEL llama --VID_NUM {selected_videos_num} --TAGS_NUM {tag}"
        result = os.system(command)  # Run the command


print("All prompts have been processed and saved.")
